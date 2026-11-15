"""
AI screening of EMAGE: is each problem in the DATA or in the MODEL?

For every measure, the same number is computed three times, on the same audio:

    data     BEAT2 — the motion-capture EMAGE was trained on
    argmax   EMAGE as we run it (the official inference: at every step take
             the single most likely code from each body-part codebook)
    sample   EMAGE, same weights, but drawing codes from the predicted
             distribution instead of always taking the top one

and judged:

    data has it,  argmax lacks it,  sample has it   -> DECODING  (argmax averages it away)
    data has it,  argmax lacks it,  sample lacks it -> MODEL     (lost in training / codebook)
    data lacks it                                    -> DATA      (never there to learn)
    all close                                        -> OK

Measures (face ones are taken from the SMPL-X mesh, in the head's own frame):
    blinks/min            eyelid aperture dips (iBUG 37/38 vs 41/40, 43/44 vs 47/46)
    brow movement         std of brow height above the eyes (raise / lower)
    frown movement        std of the distance between inner brow ends
    expression energy     summed std of the 100 FLAME coefficients
    lip movement          std of upper-lower inner-lip gap while speaking
    mouth open in pauses  median lip gap in silence
    lip-sync corr         loudness vs lip gap correlation (best lag within 150 ms)
    eye movement          std of eyeball rotation
    head movement         std of head+neck rotation
    gesture speed         upper-body angular speed, deg/s
    gesture variety       mean distance between upper-body poses 0.5 s apart
                          sampled across the clip (low = the same few poses)

Also writes, for an AI (or a person) to look at:
    plots/<clip>.png      eyelid / brow / lip traces, data vs argmax vs sample
    faces/<clip>.png      face close-ups at 8 moments, one row per source

Run (GPU box, pantomatrix env) — audit_emage.sh does all of it:
    python audit_emage.py --out /root/audit
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

FPS = 30
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/root/audit")
ap.add_argument("--speakers", type=int, default=4)
ap.add_argument("--clips-per-speaker", type=int, default=2)
ap.add_argument("--seconds", type=float, default=45.0, help="crop each clip to this length")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--repo", default="/root/PantoMatrix")
ap.add_argument("--weights", default="/root/emage_weights")
ap.add_argument("--model-folder", default="/root/PantoMatrix/emage_evaltools/smplx_models")
ap.add_argument("--data-dir", default="/root/beat2_subset")
args = ap.parse_args()
out = Path(args.out)
for d in ("plots", "faces"):
    (out / d).mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. a small slice of BEAT2
# =============================================================================
def fetch_beat2():
    """Lists the dataset repo and pulls a few speakers' motion + matching audio.
    Picks files by pattern rather than hard-coded paths, so a reorganised repo
    still works."""
    import os
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
    split = [f for f in files if f.endswith("train_test_split.csv") and "english" in f.lower()]
    test = set()
    if split:
        import csv
        p = hf_hub_download(repo, split[0], repo_type="dataset", local_dir=args.data_dir)
        with open(p) as fh:
            for row in csv.DictReader(fh):
                if row.get("type", "").strip() in ("test", "val"):
                    test.add(row.get("id", "").strip())
    by_spk = {}
    for f in sorted(motion):
        stem = Path(f).stem
        if stem not in wavs:
            continue
        spk = stem.split("_")[1] if "_" in stem else "?"
        by_spk.setdefault(spk, []).append((stem in test, stem, f))
    print(f"  BEAT2: {len(motion)} motion files, {len(by_spk)} speakers"
          + (f", {len(test)} held-out (test/val) ids" if test else ""))
    chosen = []
    for spk in sorted(by_spk)[:args.speakers]:
        # prefer held-out clips: the model never saw them in training
        for _, stem, f in sorted(by_spk[spk], key=lambda x: (not x[0], x[1]))[:args.clips_per_speaker]:
            mp = hf_hub_download(repo, f, repo_type="dataset", local_dir=args.data_dir)
            wp = hf_hub_download(repo, wavs[stem], repo_type="dataset", local_dir=args.data_dir)
            chosen.append((stem, mp, wp, stem in test))
    return chosen


print("== 1. data ==")
clips = fetch_beat2()
if not clips:
    raise SystemExit("no BEAT2 clips found — check the dataset listing printed above")
for stem, _, _, held in clips:
    print(f"  {stem}{'  (held-out)' if held else '  (training set)'}")


# =============================================================================
# 2. EMAGE, argmax and sampled
# =============================================================================
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import librosa  # noqa: E402

sys.path.insert(0, str(args.repo))
from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel  # noqa: E402

dev = "cuda"
src = str(args.weights)
model = EmageAudioModel.from_pretrained(src).to(dev).eval()
parts = {k: EmageVQVAEConv.from_pretrained(src, subfolder=f"emage_vq/{k}").to(dev).eval()
         for k in ("face", "upper", "hands", "lower")}
vq = EmageVQModel(face_model=parts["face"], upper_model=parts["upper"], hands_model=parts["hands"],
                  lower_model=parts["lower"],
                  global_model=EmageVAEConv.from_pretrained(src, subfolder="emage_vq/global").to(dev).eval()
                  ).to(dev).eval()
c = model.cfg
SR = c.audio_sr
mode = {p: ("codes" if getattr(c, "c" + p[0]) > 0 else "continuous" if getattr(c, "l" + p[0]) > 0 else "off")
        for p in ("face", "upper", "hands", "lower")}
print(f"\n== 2. model ==\n  how each part is decoded: {mode}")
print("  ('codes' parts are where argmax vs sampling makes a difference)")
torch.manual_seed(args.seed)


def gen(audio, sample=False):
    a = torch.from_numpy(audio).float().to(dev).unsqueeze(0)
    with torch.no_grad():
        lat = model.inference(a, torch.zeros(1, 1).long().to(dev), vq, masked_motion=None, mask=None)

        def pick(k):
            logits = lat[k]
            if not sample:
                return torch.max(F.log_softmax(logits, dim=2), dim=2)[1]
            probs = F.softmax(logits / args.temperature, dim=2)
            b, t, v = probs.shape
            return torch.multinomial(probs.reshape(-1, v), 1).reshape(b, t)

        pred = vq.decode(
            face_latent=lat["rec_face"] if c.lf > 0 and c.cf == 0 else None,
            upper_latent=lat["rec_upper"] if c.lu > 0 and c.cu == 0 else None,
            hands_latent=lat["rec_hands"] if c.lh > 0 and c.ch == 0 else None,
            lower_latent=lat["rec_lower"] if c.ll > 0 and c.cl == 0 else None,
            face_index=pick("cls_face") if c.cf > 0 else None,
            upper_index=pick("cls_upper") if c.cu > 0 else None,
            hands_index=pick("cls_hands") if c.ch > 0 else None,
            lower_index=pick("cls_lower") if c.cl > 0 else None,
            get_global_motion=True, ref_trans=torch.zeros(1, 3).to(dev))
    return pred["motion_axis_angle"][0].cpu().numpy(), pred["expression"][0].cpu().numpy()


# =============================================================================
# 3. measures
# =============================================================================
import smplx  # noqa: E402

# SMPL-X output joints: 55 skeleton + 21 extra + 51 face landmarks = iBUG-68
# points 17..67, so iBUG k -> joint 76 + (k - 17).
L = lambda k: 76 + k - 17
EYE_PAIRS = [(L(37), L(41)), (L(38), L(40)), (L(43), L(47)), (L(44), L(46))]
BROWS = [L(k) for k in range(17, 27)]
EYES = [L(k) for k in range(36, 48)]
INNER_BROWS = (L(21), L(22))
LIP = (L(62), L(66))
_smpl = {}


def face_landmarks(poses, expr):
    """Face landmarks in the head's own frame: body, head and global
    orientation left at zero, so heights are not confounded with head pitch."""
    n = len(poses)
    m = smplx.create(args.model_folder, model_type="smplx", gender="neutral", ext="npz", num_betas=300,
                     num_expression_coeffs=100, use_pca=False, use_face_contour=False,
                     flat_hand_mean=False, batch_size=n).to(dev).eval()
    z = lambda k: torch.zeros(n, k, device=dev)
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev)
    with torch.no_grad():
        o = m(betas=z(300), transl=z(3), global_orient=z(3), body_pose=z(63), jaw_pose=T(poses[:, 66:69]),
              leye_pose=T(poses[:, 69:72]), reye_pose=T(poses[:, 72:75]), expression=T(expr),
              left_hand_pose=z(45), right_hand_pose=z(45))
    if "faces" not in _smpl:
        _smpl["faces"] = m.faces.astype(np.int32)
        W = m.lbs_weights.detach().cpu().numpy()
        _smpl["eyes"] = [np.where((W.argmax(1) == j) & (W.max(1) > 0.5))[0] for j in (23, 24)]
        _smpl["vt"] = m.v_template.detach().cpu().numpy()
    return o.joints.cpu().numpy() * 1000, o.vertices.cpu().numpy()      # mm, m


def speech_mask(audio, n):
    hop = SR / FPS
    rms = np.array([np.sqrt(np.mean(audio[int(i * hop):int(i * hop) + int(2 * hop)] ** 2) + 1e-12)
                    for i in range(n)])
    db = 20 * np.log10(rms + 1e-9)
    sp = db > db.max() - 35
    idx = np.where(sp)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 6:
            sp[a:b] = True
    return sp, db


def runs(mask, lo, hi):
    out, s = [], None
    for i, x in enumerate(np.append(mask, False)):
        if x and s is None:
            s = i
        elif not x and s is not None:
            if lo <= i - s <= hi:
                out.append((s, i))
            s = None
    return out


def rot_deg(rv):
    return np.degrees(np.linalg.norm(rv, axis=-1))


def measures(poses, expr, audio):
    n = len(poses)
    J, V = face_landmarks(poses, expr)
    sp, db = speech_mask(audio, n)
    aperture = np.mean([np.linalg.norm(J[:, a] - J[:, b], axis=1) for a, b in EYE_PAIRS], axis=0)
    med = np.median(aperture)
    blinks = runs(aperture < 0.5 * med, 1, 12) if med > 0.3 else []
    brow_h = J[:, BROWS, 1].mean(1) - J[:, EYES, 1].mean(1)
    inner = np.linalg.norm(J[:, INNER_BROWS[0]] - J[:, INNER_BROWS[1]], axis=1)
    gap = np.linalg.norm(J[:, LIP[0]] - J[:, LIP[1]], axis=1)
    env = np.clip(db - (db.max() - 40), 0, None)
    best = -1.0
    for lag in range(-4, 5):
        a_, b_ = (env[:n - lag], gap[lag:]) if lag >= 0 else (env[-lag:], gap[:n + lag])
        if len(a_) > 30 and a_.std() > 0 and b_.std() > 0:
            best = max(best, float(np.corrcoef(a_, b_)[0, 1]))
    sil = ~sp
    sil[:6] = False
    up = [13, 14, 16, 17, 18, 19, 20, 21]
    from scipy.spatial.transform import Rotation as R
    speed = np.mean([np.degrees((R.from_rotvec(poses[:-1, 3 * j:3 * j + 3]).inv()
                                 * R.from_rotvec(poses[1:, 3 * j:3 * j + 3])).magnitude()) for j in up], 0) * FPS
    feat = np.concatenate([poses[:, 3 * j:3 * j + 3] for j in up], 1)[::15]
    var = float(np.mean([np.linalg.norm(feat[i] - feat[j]) for i in range(len(feat)) for j in range(i + 1, len(feat))])) \
        if len(feat) > 2 else 0.0
    head = np.concatenate([poses[:, 36:39], poses[:, 45:48]], 1)        # neck 12, head 15
    return {
        "blinks_per_min": len(blinks) / (n / FPS / 60),
        "eyelid_aperture_mm": float(med),
        "brow_movement_mm": float(brow_h.std()),
        "frown_movement_mm": float(inner.std()),
        "expression_energy": float(expr.std(0).sum()),
        "lip_movement_mm": float(gap[sp].std()) if sp.sum() > 30 else float(gap.std()),
        "mouth_open_in_pauses_mm": float(np.median(gap[sil])) if sil.sum() > 10 else float("nan"),
        "lip_sync_corr": best,
        "eye_movement_deg": float(rot_deg(poses[:, 69:75].reshape(n, 2, 3)).std()),
        "head_movement_deg": float(rot_deg(head.reshape(n, 2, 3)).std()),
        "gesture_speed_dps": float(np.median(speed)),
        "gesture_variety": var,
    }, {"aperture": aperture, "brow": brow_h, "inner": inner, "gap": gap, "speech": sp, "verts": V,
        "expr_std": expr.std(0)}


# =============================================================================
# 4. pictures for review
# =============================================================================
def plot_traces(path, name, tr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"data": "#2b6cb0", "argmax": "#c0504d", "sample": "#2f855a"}
    rows = [("aperture", "eyelid aperture (mm)\n dips = blinks"), ("brow", "brow height (mm)\n raise / lower"),
            ("inner", "inner-brow distance (mm)\n down = frown"), ("gap", "lip gap (mm)")]
    fig, ax = plt.subplots(len(rows), 1, figsize=(12, 9), sharex=True)
    for i, (k, lab) in enumerate(rows):
        for s, t in tr.items():
            x = np.arange(len(t[k])) / FPS
            ax[i].plot(x, t[k], color=col[s], lw=1.1, label=s, alpha=0.9)
        sp = tr["data"]["speech"]
        ax[i].fill_between(np.arange(len(sp)) / FPS, *ax[i].get_ylim(), where=~sp, color="#bbb", alpha=0.25, lw=0)
        ax[i].set_ylabel(lab, fontsize=8)
    ax[0].legend(loc="upper right", fontsize=8, ncol=3)
    ax[0].set_title(f"{name}   (grey = silence)", fontsize=10)
    ax[-1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def face_sheet(path, name, tr, k=8):
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh
    faces, vt = _smpl["faces"], _smpl["vt"]
    vcol = np.ones((vt.shape[0], 4), np.float32)
    for idx in _smpl["eyes"]:
        cen = vt[idx].mean(0)
        d = vt[idx] - cen
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        ang = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))
        col = np.tile(np.float32([1.5, 1.45, 1.35, 1]), (len(idx), 1))
        col[ang < 30] = [0.10, 0.06, 0.04, 1]
        col[ang < 13] = [0.004, 0.004, 0.004, 1]
        vcol[idx] = col
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.52, 0.54, 0.60, 1], metallicFactor=0,
                                             roughnessFactor=0.7)
    n = min(len(t["verts"]) for t in tr.values())
    times = np.linspace(0.08 * n, 0.92 * n, k).astype(int)
    v0 = tr["data"]["verts"][0]
    head = (v0[_smpl["eyes"][0]].mean(0) + v0[_smpl["eyes"][1]].mean(0)) / 2   # between the eyes
    S = 256
    r = pyrender.OffscreenRenderer(S, S)
    cam = np.eye(4)
    cam[:3, 3] = [head[0], head[1] - 0.03, head[2] + 0.45]       # in front, aimed at the nose
    key = np.eye(4)
    key[:3, 3] = [head[0] + 0.3, head[1] + 0.4, head[2] + 0.8]
    rows = []
    for s, t in tr.items():
        row = []
        for f in times:
            sc = pyrender.Scene(bg_color=[0.09, 0.09, 0.11, 1], ambient_light=[0.12] * 3)
            tm = trimesh.Trimesh(t["verts"][f], faces, process=False)
            sc.add(pyrender.Mesh([pyrender.Primitive(positions=tm.vertices, normals=tm.vertex_normals,
                                                     indices=faces, color_0=vcol, material=mat, mode=4)]))
            sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 8), pose=cam)
            sc.add(pyrender.DirectionalLight(intensity=2.2), pose=key)
            sc.add(pyrender.DirectionalLight(intensity=1.0), pose=cam)
            row.append(r.render(sc)[0])
        rows.append(np.hstack(row))
    r.delete()
    img = np.vstack(rows)
    try:
        import cv2
        for i, s in enumerate(tr):
            cv2.putText(img, s, (6, i * S + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2, cv2.LINE_AA)
        for j, f in enumerate(times):
            cv2.putText(img, f"{f / FPS:.1f}s", (j * S + 6, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
    except ImportError:
        pass
    import imageio
    imageio.imwrite(path, img)


# =============================================================================
# 5. run
# =============================================================================
print("\n== 3. measuring ==")
results, expr_std = {}, {"data": [], "argmax": [], "sample": []}
for stem, mp, wp, held in clips:
    d = np.load(mp, allow_pickle=True)
    n = int(min(len(d["poses"]), args.seconds * FPS))
    audio = librosa.load(wp, sr=SR)[0][:int(n / FPS * SR)]
    n = min(n, int(len(audio) / SR * FPS))
    srcs = {"data": (np.asarray(d["poses"][:n], np.float32), np.asarray(d["expressions"][:n], np.float32))}
    p, e = gen(audio)
    srcs["argmax"] = (p[:n], e[:n])
    p, e = gen(audio, sample=True)
    srcs["sample"] = (p[:n], e[:n])
    n = min(len(v[0]) for v in srcs.values())
    tr = {}
    results[stem] = {"held_out": held, "seconds": round(n / FPS, 1)}
    for s, (p, e) in srcs.items():
        m, t = measures(p[:n], e[:n], audio)
        results[stem][s] = m
        tr[s] = t
        expr_std[s].append(t["expr_std"])
    plot_traces(out / "plots" / f"{stem}.png", stem, tr)
    face_sheet(out / "faces" / f"{stem}.png", stem, tr)
    print(f"  {stem}: {n / FPS:.0f}s  blinks/min data {results[stem]['data']['blinks_per_min']:.1f} "
          f"argmax {results[stem]['argmax']['blinks_per_min']:.1f} sample {results[stem]['sample']['blinks_per_min']:.1f}")

# ---- aggregate and judge ---------------------------------------------------------
keys = list(next(iter(results.values()))["data"].keys())
agg = {k: {s: float(np.nanmean([results[c][s][k] for c in results])) for s in ("data", "argmax", "sample")}
       for k in keys}
# What a viewer would call "present": thresholds on the DATA side only, so the
# data verdict does not depend on the model.
PRESENT = {"blinks_per_min": 5.0, "eye_movement_deg": 1.0, "brow_movement_mm": 0.3, "frown_movement_mm": 0.2}
LOWER_IS_BETTER = {"mouth_open_in_pauses_mm"}


def verdict(k, v):
    d, a, s = v["data"], v["argmax"], v["sample"]
    if k in PRESENT and d < PRESENT[k]:
        return "DATA: barely present in the training data"
    if k in LOWER_IS_BETTER:
        if a > 1.5 * d:
            return "MODEL: worse than data" + (" (sampling helps)" if s < 0.8 * a else "")
        return "ok"
    if k == "lip_sync_corr":
        return "MODEL: weaker sync than data" if a < d - 0.15 else "ok"
    if d <= 1e-6:
        return "n/a"
    ra, rs = a / d, s / d
    if ra < 0.6:
        return f"DECODING: argmax keeps {ra:.0%} of the data's amount, sampling {rs:.0%}" if rs >= 0.8 else \
               f"MODEL: argmax keeps {ra:.0%}, sampling {rs:.0%} — lost in training/codebook"
    if ra > 1.6:
        return f"MODEL: exaggerated ({ra:.0%} of data)"
    return "ok"


report = {"clips": results, "summary": {k: {**{s: round(v[s], 3) for s in v}, "verdict": verdict(k, v)}
                                        for k, v in agg.items()},
          "decoding_mode": mode, "temperature": args.temperature}
# which FLAME components does the model lose most?
ed, ea = np.mean(expr_std["data"], 0), np.mean(expr_std["argmax"], 0)
ratio = ea / np.maximum(ed, 1e-6)
order = np.argsort(ratio)
report["flame_components_most_lost"] = [{"component": int(i), "data_std": round(float(ed[i]), 3),
                                         "argmax_std": round(float(ea[i]), 3), "kept": round(float(ratio[i]), 2)}
                                        for i in order[:10] if ed[i] > 0.05]
(out / "audit_report.json").write_text(json.dumps(report, indent=2))

names = {"blinks_per_min": "blinks / min", "eyelid_aperture_mm": "eyelid opening (mm)",
         "brow_movement_mm": "brow raise/lower (mm std)", "frown_movement_mm": "frown (mm std)",
         "expression_energy": "expression energy", "lip_movement_mm": "lip movement (mm std)",
         "mouth_open_in_pauses_mm": "mouth open in pauses (mm)", "lip_sync_corr": "lip-sync corr",
         "eye_movement_deg": "eye movement (deg std)", "head_movement_deg": "head movement (deg std)",
         "gesture_speed_dps": "gesture speed (deg/s)", "gesture_variety": "gesture variety"}
lines = ["# EMAGE audit: data vs model", "",
         f"{len(results)} BEAT2 clips ({sum(r['held_out'] for r in results.values())} held-out), "
         f"{sum(r['seconds'] for r in results.values()):.0f} s total. Decoding: {mode}.", "",
         "| measure | BEAT2 data | EMAGE argmax | EMAGE sampled | verdict |", "|---|---|---|---|---|"]
for k, v in report["summary"].items():
    lines.append(f"| {names.get(k, k)} | {v['data']:.2f} | {v['argmax']:.2f} | {v['sample']:.2f} | {v['verdict']} |")
lines += ["", "FLAME components the model loses most (std kept vs data):", ""]
lines += [f"- #{x['component']}: data {x['data_std']}, model {x['argmax_std']} ({x['kept']:.0%} kept)"
          for x in report["flame_components_most_lost"]]
lines += ["", "Pictures: plots/<clip>.png (traces), faces/<clip>.png (close-ups, one row per source)."]
(out / "audit_report.md").write_text("\n".join(lines))
print("\n" + "\n".join(lines))
