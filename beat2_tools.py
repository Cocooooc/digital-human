"""
Shared pieces for the root-cause work: BEAT2 access, EMAGE with controllable
decoding, face landmarks from the SMPL-X mesh, audio features, face renders.

Python 3.9 / pantomatrix env.
"""
import os
import sys
from pathlib import Path

import numpy as np

FPS = 30
L = lambda k: 76 + k - 17          # iBUG-68 index -> SMPL-X output joint (face landmarks)
EYE_PAIRS = [(L(37), L(41)), (L(38), L(40)), (L(43), L(47)), (L(44), L(46))]
BROWS = [L(k) for k in range(17, 27)]
EYE_CORNERS = [L(36), L(39), L(42), L(45)]     # corners do not move when the lids close
INNER_BROWS = (L(21), L(22))
LIP = (L(62), L(66))
UPPER = [13, 14, 16, 17, 18, 19, 20, 21]


# ---------------------------------------------------------------------------
# BEAT2
# ---------------------------------------------------------------------------
def fetch_beat2(data_dir, speakers=8, clips_per_speaker=2, prefer_heldout=True, skip=()):
    """[(stem, motion_npz, wav, held_out)] for a few speakers, picked by pattern."""
    from huggingface_hub import HfApi, hf_hub_download
    repo = "H-Liu1997/BEAT2"
    try:
        files = HfApi().list_repo_files(repo, repo_type="dataset")
    except Exception as e:
        print(f"  huggingface.co failed ({type(e).__name__}); trying hf-mirror.com")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        files = HfApi(endpoint="https://hf-mirror.com").list_repo_files(repo, repo_type="dataset")
    motion = [f for f in files if "smplxflame_30" in f and f.endswith(".npz") and "english" in f.lower()]
    wavs = {Path(f).stem: f for f in files if "wave16k" in f and f.endswith(".wav") and "english" in f.lower()}
    test = set()
    split = [f for f in files if f.endswith("train_test_split.csv") and "english" in f.lower()]
    if split:
        import csv
        p = hf_hub_download(repo, split[0], repo_type="dataset", local_dir=data_dir)
        with open(p) as fh:
            for row in csv.DictReader(fh):
                if row.get("type", "").strip() in ("test", "val"):
                    test.add(row.get("id", "").strip())
    by_spk = {}
    for f in sorted(motion):
        stem = Path(f).stem
        if stem in wavs and stem not in skip:
            by_spk.setdefault(stem.split("_")[1], []).append((stem in test, stem, f))
    out = []
    for spk in sorted(by_spk, key=lambda s: int(by_spk[s][0][1].split("_")[0]))[:speakers]:
        items = sorted(by_spk[spk], key=lambda x: ((not x[0]) if prefer_heldout else x[0], x[1]))
        for held, stem, f in items[:clips_per_speaker]:
            mp = hf_hub_download(repo, f, repo_type="dataset", local_dir=data_dir)
            wp = hf_hub_download(repo, wavs[stem], repo_type="dataset", local_dir=data_dir)
            out.append((stem, mp, wp, held))
    return out


def speaker_number(stem):
    return int(stem.split("_")[0])


# ---------------------------------------------------------------------------
# EMAGE with controllable decoding
# ---------------------------------------------------------------------------
class Emage:
    """gen(audio, speaker=0, margin=0.0, sample=False)

    margin > 0: "sticky" decoding for the body codebooks — keep the previous
    frame's code unless the best code beats it by more than `margin` in
    log-probability. margin 0 is the official argmax."""

    def __init__(self, repo, weights):
        import torch
        sys.path.insert(0, str(repo))
        from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel
        self.torch = torch
        self.dev = "cuda"
        src = str(weights)
        self.model = EmageAudioModel.from_pretrained(src).to(self.dev).eval()
        parts = {k: EmageVQVAEConv.from_pretrained(src, subfolder=f"emage_vq/{k}").to(self.dev).eval()
                 for k in ("face", "upper", "hands", "lower")}
        self.vq = EmageVQModel(face_model=parts["face"], upper_model=parts["upper"],
                               hands_model=parts["hands"], lower_model=parts["lower"],
                               global_model=EmageVAEConv.from_pretrained(src, subfolder="emage_vq/global")
                               .to(self.dev).eval()).to(self.dev).eval()
        self.cfg = self.model.cfg
        self.sr = self.cfg.audio_sr
        embs = [(nm, m) for nm, m in self.model.named_modules() if isinstance(m, torch.nn.Embedding)]
        self.embedding_sizes = {nm: m.num_embeddings for nm, m in embs}
        spk = [m.num_embeddings for nm, m in embs if "speaker" in nm.lower() or "spk" in nm.lower()]
        # Only sweep speakers when the table is identified by name: an id past
        # the end of an embedding is a CUDA assert that kills the process.
        self.n_speakers = spk[0] if spk else 1
        self.last_switch_rate = {}

    def _sticky(self, logits, margin):
        torch = self.torch
        lp_t = torch.log_softmax(logits, dim=2)[0]          # (T, V)
        if margin <= 0:
            return lp_t.argmax(1).unsqueeze(0)
        lp = lp_t.float().cpu().numpy()
        best = lp.argmax(1)
        idx = best.copy()
        for t in range(1, len(lp)):
            prev = idx[t - 1]
            if lp[t, prev] >= lp[t, best[t]] - margin:
                idx[t] = prev
        return torch.from_numpy(idx).to(logits.device).unsqueeze(0)

    def gen(self, audio, speaker=0, margin=0.0, sample=False, temperature=1.0):
        torch, c = self.torch, self.cfg
        F = torch.nn.functional
        a = torch.from_numpy(np.ascontiguousarray(audio, np.float32)).to(self.dev).unsqueeze(0)
        spk = torch.full((1, 1), int(speaker), dtype=torch.long, device=self.dev)
        with torch.no_grad():
            lat = self.model.inference(a, spk, self.vq, masked_motion=None, mask=None)

            def pick(k):
                if sample:
                    p = F.softmax(lat[k] / temperature, dim=2)
                    b, t, v = p.shape
                    idx = torch.multinomial(p.reshape(-1, v), 1).reshape(b, t)
                else:
                    idx = self._sticky(lat[k], margin)
                self.last_switch_rate[k] = float((idx[0, 1:] != idx[0, :-1]).float().mean())
                return idx

            pred = self.vq.decode(
                face_latent=lat["rec_face"] if c.lf > 0 and c.cf == 0 else None,
                upper_latent=lat["rec_upper"] if c.lu > 0 and c.cu == 0 else None,
                hands_latent=lat["rec_hands"] if c.lh > 0 and c.ch == 0 else None,
                lower_latent=lat["rec_lower"] if c.ll > 0 and c.cl == 0 else None,
                face_index=pick("cls_face") if c.cf > 0 else None,
                upper_index=pick("cls_upper") if c.cu > 0 else None,
                hands_index=pick("cls_hands") if c.ch > 0 else None,
                lower_index=pick("cls_lower") if c.cl > 0 else None,
                get_global_motion=True, ref_trans=torch.zeros(1, 3).to(self.dev))
        return pred["motion_axis_angle"][0].cpu().numpy(), pred["expression"][0].cpu().numpy()


# ---------------------------------------------------------------------------
# face landmarks (head frame), mm
# ---------------------------------------------------------------------------
_SMPL = {}


def face_mesh(model_folder, poses, expr, keep_verts=False):
    import torch
    import smplx
    dev = "cuda"
    n = len(poses)
    out_j, out_v = [], []
    for s in range(0, n, 600):
        p, e = poses[s:s + 600], expr[s:s + 600]
        b = len(p)
        m = smplx.create(model_folder, model_type="smplx", gender="neutral", ext="npz", num_betas=300,
                         num_expression_coeffs=100, use_pca=False, use_face_contour=False,
                         flat_hand_mean=False, batch_size=b).to(dev).eval()
        z = lambda k: torch.zeros(b, k, device=dev)
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev)
        with torch.no_grad():
            o = m(betas=z(300), transl=z(3), global_orient=z(3), body_pose=z(63), jaw_pose=T(p[:, 66:69]),
                  leye_pose=T(p[:, 69:72]), reye_pose=T(p[:, 72:75]), expression=T(e),
                  left_hand_pose=z(45), right_hand_pose=z(45))
        if "faces" not in _SMPL:
            _SMPL["faces"] = m.faces.astype(np.int32)
            W = m.lbs_weights.detach().cpu().numpy()
            _SMPL["eyes"] = [np.where((W.argmax(1) == j) & (W.max(1) > 0.5))[0] for j in (23, 24)]
            _SMPL["vt"] = m.v_template.detach().cpu().numpy()
        out_j.append(o.joints.cpu().numpy() * 1000)
        if keep_verts:
            out_v.append(o.vertices.cpu().numpy())
        del m
    return np.concatenate(out_j), (np.concatenate(out_v) if keep_verts else None)


def face_signals(J):
    aperture = np.mean([np.linalg.norm(J[:, a] - J[:, b], axis=1) for a, b in EYE_PAIRS], axis=0)
    brow = J[:, BROWS, 1].mean(1) - J[:, EYE_CORNERS, 1].mean(1)
    inner = np.linalg.norm(J[:, INNER_BROWS[0]] - J[:, INNER_BROWS[1]], axis=1)
    gap = np.linalg.norm(J[:, LIP[0]] - J[:, LIP[1]], axis=1)
    return {"aperture": aperture, "brow": brow, "inner": inner, "gap": gap}


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def runs(mask, lo=1, hi=10 ** 9):
    out, s = [], None
    for i, x in enumerate(np.append(mask, False)):
        if x and s is None:
            s = i
        elif not x and s is not None:
            if lo <= i - s <= hi:
                out.append((s, i))
            s = None
    return out


def rolling_median(x, w):
    pad = np.pad(x, (w // 2, w // 2), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(pad, w), axis=1)[:len(x)]


def blinks(aperture):
    """[(start, end, min_frame)] — lid closes below 55% of the local open level for 2-12 frames."""
    base = rolling_median(aperture, 31)
    ev = []
    for s, e in runs(aperture < 0.55 * base, 2, 12):
        ev.append((s, e, s + int(np.argmin(aperture[s:e]))))
    return ev


def brow_events(brow, inner, exclude=()):
    """Raises (brow up) and frowns (inner brows together), against a 3 s local
    baseline, at least 3 frames, away from blinks (lid motion is not a brow)."""
    n = len(brow)
    ok = np.ones(n, bool)
    for s, e, _ in exclude:
        ok[max(0, s - 3):e + 3] = False
    rb = brow - rolling_median(brow, 91)
    ri = inner - rolling_median(inner, 91)
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6
    t_r = max(1.0, 3 * mad(rb))
    t_f = max(0.5, 3 * mad(ri))
    raise_ = [(s, e, s + int(np.argmax(rb[s:e]))) for s, e in runs((rb > t_r) & ok, 3, 90)]
    frown = [(s, e, s + int(np.argmin(ri[s:e]))) for s, e in runs((ri < -t_f) & ok, 3, 90)]
    return raise_, frown, (t_r, t_f)


def template(expr, events, near=(3, 10)):
    """Mean expression change at the event peak relative to just before the
    event starts and just after it ends."""
    a, b = near
    d = []
    for s, e, p in events:
        if s - b < 0 or e + b >= len(expr):
            continue
        ctx = np.concatenate([expr[s - b:s - a], expr[e + a:e + b]])
        d.append(expr[p] - ctx.mean(0))
    return (np.mean(d, 0), len(d)) if d else (np.zeros(expr.shape[1]), 0)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def audio_feats(audio, sr, n):
    import librosa
    hop = sr / FPS
    rms = np.array([np.sqrt(np.mean(audio[int(i * hop):int(i * hop) + int(2 * hop)] ** 2) + 1e-12)
                    for i in range(n)])
    db = 20 * np.log10(rms + 1e-9)
    sp = db > db.max() - 35
    idx = np.where(sp)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 6:
            sp[a:b] = True
    f0, voiced, _ = librosa.pyin(audio, fmin=65, fmax=500, sr=sr, frame_length=1024, hop_length=int(hop))
    f = np.full(n, np.nan)
    m = min(n, len(f0))
    f[:m] = np.where(voiced[:m], f0[:m], np.nan)
    return db, sp, f


def pitch_accents(f0, min_prom=1.5):
    """Frames of local pitch peaks that stand out (semitones) from the surrounding 0.6 s."""
    ok = ~np.isnan(f0)
    if ok.sum() < 10:
        return []
    st = np.full(len(f0), np.nan)
    st[ok] = 12 * np.log2(f0[ok] / np.nanmedian(f0))
    peaks = []
    for t in range(9, len(st) - 9):
        if np.isnan(st[t]):
            continue
        win = st[t - 9:t + 10]
        if st[t] >= np.nanmax(win) and st[t] - np.nanmin(win) >= min_prom:
            if not peaks or t - peaks[-1] > 6:
                peaks.append(t)
    return peaks


# ---------------------------------------------------------------------------
# body
# ---------------------------------------------------------------------------
def body_stats(poses):
    from scipy.spatial.transform import Rotation as R
    sp = np.mean([np.degrees((R.from_rotvec(poses[:-1, 3 * j:3 * j + 3]).inv()
                              * R.from_rotvec(poses[1:, 3 * j:3 * j + 3])).magnitude()) for j in UPPER], 0) * FPS
    x = np.concatenate([poses[:, 3 * j:3 * j + 3] for j in UPPER], 1)
    x = x - x.mean(0)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / FPS)
    hf = float(spec[fr > 4].sum() / (spec[fr > 0.2].sum() + 1e-12))
    feat = x[::15]
    var = float(np.mean([np.linalg.norm(feat[i] - feat[j]) for i in range(len(feat))
                         for j in range(i + 1, len(feat))])) if len(feat) > 2 else 0.0
    amp = float(np.degrees(np.linalg.norm(x.reshape(len(x), -1, 3), axis=2)).std())
    return {"speed_dps": float(np.median(sp)), "hf_ratio": hf, "variety": var, "amplitude_deg": amp}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_faces(path, rows, labels=None, size=256):
    """rows: [(row_label, [verts, verts, ...])] -> one PNG grid."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh
    import imageio
    faces, vt = _SMPL["faces"], _SMPL["vt"]
    vcol = np.ones((vt.shape[0], 4), np.float32)
    for idx in _SMPL["eyes"]:
        d = vt[idx] - vt[idx].mean(0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        ang = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))
        col = np.tile(np.float32([1.5, 1.45, 1.35, 1]), (len(idx), 1))
        col[ang < 30] = [0.10, 0.06, 0.04, 1]
        col[ang < 13] = [0.004, 0.004, 0.004, 1]
        vcol[idx] = col
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.52, 0.54, 0.60, 1], metallicFactor=0,
                                             roughnessFactor=0.7)
    v0 = rows[0][1][0]
    head = (v0[_SMPL["eyes"][0]].mean(0) + v0[_SMPL["eyes"][1]].mean(0)) / 2
    cam = np.eye(4)
    cam[:3, 3] = [head[0], head[1] - 0.02, head[2] + 0.40]
    key = np.eye(4)
    key[:3, 3] = [head[0] + 0.3, head[1] + 0.4, head[2] + 0.8]
    r = pyrender.OffscreenRenderer(size, size)
    img_rows = []
    for _, vs in rows:
        row = []
        for v in vs:
            sc = pyrender.Scene(bg_color=[0.09, 0.09, 0.11, 1], ambient_light=[0.12] * 3)
            tm = trimesh.Trimesh(v, faces, process=False)
            sc.add(pyrender.Mesh([pyrender.Primitive(positions=tm.vertices, normals=tm.vertex_normals,
                                                     indices=faces, color_0=vcol, material=mat, mode=4)]))
            sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 8), pose=cam)
            sc.add(pyrender.DirectionalLight(intensity=2.2), pose=key)
            sc.add(pyrender.DirectionalLight(intensity=1.0), pose=cam)
            row.append(r.render(sc)[0])
        img_rows.append(np.hstack(row))
    r.delete()
    w = max(x.shape[1] for x in img_rows)
    img = np.vstack([np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0))) for x in img_rows])
    try:
        import cv2
        for i, (lab, _) in enumerate(rows):
            cv2.putText(img, lab, (6, i * size + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2,
                        cv2.LINE_AA)
        if labels:
            for j, lab in enumerate(labels):
                cv2.putText(img, lab, (j * size + 6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1, cv2.LINE_AA)
    except ImportError:
        pass
    imageio.imwrite(path, img)
