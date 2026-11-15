"""
Fine-tune EMAGE on one other BEAT2 speaker — the fast form of "retrain from
the data source".

Why: the released EMAGE was trained on BEAT2 speaker 2 (Scott) only
(configs/emage_audio.yaml: beat2_s20_l64_speaker2.json, speaker_dims 1), and
its output reproduces his style: widest mouth, fastest gestures, almost no
blinks among the 8 speakers measured. Re-training on someone else's data is
the direct test of that diagnosis and, if it holds, the fix.

How it is kept fast:
  - start from the released weights, a few thousand steps, not 100k
  - the four motion codebooks (VQ-VAEs) stay frozen — the official training
    freezes them too; stage `check` measures whether they can represent
    other speakers at all before relying on that
  - wav and motion files are read once and kept in memory (the stock loader
    re-reads a whole wav file for every 64-frame sample)

Stages (pantomatrix env, from /root/digital-human):
  check              codebook round-trip error for each speaker in /root/beat2_subset
  prep  SPEAKER      download that speaker's BEAT2 standard set, foot contact, json, config
  train SPEAKER      fine-tune  (--steps, --lr, --bs)
  eval  SPEAKER      original vs fine-tuned vs real data on held-out clips + videos
SPEAKER = BEAT2 number, e.g. 6 (carla), 4 (lawrence), 5 (stewart).
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["check", "prep", "train", "eval"])
ap.add_argument("speaker", nargs="?", type=int, default=0)
ap.add_argument("--repo", default="/root/PantoMatrix")
ap.add_argument("--weights", default="/root/emage_weights")
ap.add_argument("--subset", default="/root/beat2_subset")
ap.add_argument("--root", default="/root/beat2_target")
ap.add_argument("--out", default="/root/ft")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--save-every", type=int, default=1000)
ap.add_argument("--model-folder", default="/root/PantoMatrix/emage_evaltools/smplx_models")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, args.repo)
os.chdir(args.repo)                      # the repo's own imports use relative paths
sys.path.insert(1, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

DEV = "cuda"
FPS = 30


def load_vq():
    from models.emage_audio import EmageVQVAEConv, EmageVAEConv, EmageVQModel
    src = args.weights
    parts = {k: EmageVQVAEConv.from_pretrained(src, subfolder=f"emage_vq/{k}").to(DEV).eval()
             for k in ("face", "upper", "hands", "lower")}
    vq = EmageVQModel(face_model=parts["face"], upper_model=parts["upper"], hands_model=parts["hands"],
                      lower_model=parts["lower"],
                      global_model=EmageVAEConv.from_pretrained(src, subfolder="emage_vq/global").to(DEV).eval()
                      ).to(DEV).eval()
    for p in vq.parameters():
        p.requires_grad = False
    return vq


def to_rot6d(poses):
    from emage_utils import rotation_conversions as rc
    t = torch.from_numpy(np.ascontiguousarray(poses, np.float32)).to(DEV).reshape(len(poses), 55, 3)
    return rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(t)).reshape(1, len(poses), 330)


# =============================================================================
# check: can Scott's codebooks represent other speakers?
# =============================================================================
if args.stage == "check":
    import inspect as _inspect
    from scipy.spatial.transform import Rotation as R
    import beat2_tools as bt
    vq = load_vq()
    dec_params = set(_inspect.signature(vq.decode).parameters)
    rows = []
    for npz in sorted(Path(args.subset).rglob("smplxflame_30/*.npz")):
        d = np.load(npz, allow_pickle=True)
        n = min(len(d["poses"]), 64 * 20)
        n -= n % 64
        P = np.asarray(d["poses"][:n], np.float32)
        E = np.asarray(d["expressions"][:n], np.float32)
        with torch.no_grad():
            r6 = to_rot6d(P)
            ex = torch.from_numpy(E).to(DEV).unsqueeze(0)
            zc = torch.zeros(1, n, 4, device=DEV)
            zt = torch.zeros(1, n, 3, device=DEV)
            idx = vq.map2index(r6, ex, zc, zt)
            lat = vq.map2latent(r6, ex, zc, zt)
            if not rows:
                print(f"  map2index keys: {list(idx) if isinstance(idx, dict) else type(idx)}")
                print(f"  map2latent keys: {list(lat) if isinstance(lat, dict) else type(lat)}")
            if not isinstance(idx, dict) or not isinstance(lat, dict):
                raise SystemExit(f"map2index/map2latent returned {type(idx)}/{type(lat)}; send this line")
            kw = {}
            for k in ("upper", "hands", "lower"):
                for cand in (f"{k}_index", k):
                    if cand in idx:
                        kw[f"{k}_index"] = idx[cand]
                        break
            for cand in ("face_latent", "face"):
                if cand in lat:
                    kw["face_latent"] = lat[cand]
                    break
            kw = {k: v for k, v in kw.items() if k in dec_params}
            out = vq.decode(**kw, get_global_motion=True, ref_trans=torch.zeros(1, 3, device=DEV))
        Pr = out["motion_axis_angle"][0].cpu().numpy()[:n]
        Er = out["expression"][0].cpu().numpy()[:n]
        err = {}
        for name, joints in (("upper", [13, 14, 16, 17, 18, 19, 20, 21]), ("hands", list(range(25, 55))),
                             ("lower", [1, 2, 4, 5, 7, 8])):
            e = [np.degrees((R.from_rotvec(P[:, 3 * j:3 * j + 3]).inv() * R.from_rotvec(Pr[:, 3 * j:3 * j + 3])).magnitude())
                 for j in joints]
            err[name] = float(np.mean(e))
        J0, _ = bt.face_mesh(args.model_folder, P, E)
        J1, _ = bt.face_mesh(args.model_folder, Pr, Er)
        g0, g1 = bt.face_signals(J0)["gap"], bt.face_signals(J1)["gap"]
        err["lip_gap_err_mm"] = float(np.abs(g0 - g1).mean())
        err["blinks_kept"] = f"{len(bt.blinks(bt.face_signals(J1)['aperture']))}/{len(bt.blinks(bt.face_signals(J0)['aperture']))}"
        rows.append((npz.stem, err))
        print(f"  {npz.stem:<22} upper {err['upper']:5.2f}°  hands {err['hands']:5.2f}°  lower {err['lower']:5.2f}°  "
              f"lip gap err {err['lip_gap_err_mm']:.2f} mm  blinks kept {err['blinks_kept']}")
    (OUT / "codebook_check.json").write_text(json.dumps(dict(rows), indent=2))
    print("\n  Low and similar errors across speakers -> Scott's codebooks can represent the others,"
          "\n  so freezing them for fine-tuning is safe. A speaker far above the rest is a poor target.")
    raise SystemExit(0)


# =============================================================================
# prep: data for one speaker
# =============================================================================
if args.stage == "prep":
    import csv
    from huggingface_hub import HfApi, hf_hub_download
    repo = "H-Liu1997/BEAT2"
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    split_f = [f for f in files if f.endswith("train_test_split.csv") and "english" in f.lower()][0]
    split = {}
    with open(hf_hub_download(repo, split_f, repo_type="dataset", local_dir=args.root)) as fh:
        for row in csv.DictReader(fh):
            split[row["id"].strip()] = row["type"].strip()
    ids = sorted(i for i, t in split.items() if int(i.split("_")[0]) == args.speaker and t != "additional")
    print(f"  speaker {args.speaker}: {len(ids)} clips "
          f"({sum(split[i] == 'train' for i in ids)} train, {sum(split[i] == 'val' for i in ids)} val, "
          f"{sum(split[i] == 'test' for i in ids)} test)")
    have = set(files)
    base = Path(split_f).parent
    t0 = time.time()
    for k, i in enumerate(ids):
        for sub, ext in (("smplxflame_30", ".npz"), ("wave16k", ".wav")):
            f = str(base / sub / (i + ext))
            if f in have:
                hf_hub_download(repo, f, repo_type="dataset", local_dir=args.root)
        if k % 10 == 0:
            print(f"    downloaded {k + 1}/{len(ids)}  ({time.time() - t0:.0f}s)")
    root = Path(args.root) / base

    # foot contact, as datasets/foot_contact.py computes it
    import smplx
    sm = smplx.create("./emage_evaltools/smplx_models/", model_type="smplx", gender="NEUTRAL_2020",
                      use_face_contour=False, num_betas=300, num_expression_coeffs=100, ext="npz",
                      use_pca=False).eval().to(DEV)
    (root / "footcontact").mkdir(exist_ok=True)
    clips = []
    for i in ids:
        npz, wav = root / "smplxflame_30" / f"{i}.npz", root / "wave16k" / f"{i}.wav"
        if not (npz.exists() and wav.exists()):
            continue
        d = np.load(npz, allow_pickle=True)
        P = torch.from_numpy(d["poses"]).float().to(DEV)
        n = len(P)
        T = torch.from_numpy(d["trans"]).float().to(DEV)
        B = torch.from_numpy(np.tile(d["betas"].reshape(1, 300), (n, 1))).float().to(DEV)
        X = torch.from_numpy(d["expressions"] if "expressions" in d else np.zeros((n, 100))).float().to(DEV)
        js = []
        with torch.no_grad():
            for s in range(0, n, 128):
                sl = slice(s, s + 128)
                js.append(sm(betas=B[sl], transl=T[sl], expression=X[sl], jaw_pose=P[sl, 66:69],
                             global_orient=P[sl, :3], body_pose=P[sl, 3:66], left_hand_pose=P[sl, 75:120],
                             right_hand_pose=P[sl, 120:165], leye_pose=P[sl, 69:72], reye_pose=P[sl, 72:75],
                             return_joints=True)["joints"][:, (7, 8, 10, 11), :].cpu())
        j = torch.cat(js).permute(1, 0, 2)
        v = torch.zeros(4, n)
        v[:, :-1] = (j[:, 1:] - j[:, :-1]).norm(dim=-1)
        np.save(root / "footcontact" / f"{i}.npy", (v < 0.01).numpy().astype(float).T)
        for st in range(0, n - 64, 20):
            clips.append({"video_id": i, "motion_path": str(npz), "audio_path": str(wav),
                          "mode": split[i], "start_idx": st, "end_idx": st + 64})
    js_path = OUT / f"beat2_s20_l64_speaker{args.speaker}.json"
    js_path.write_text(json.dumps(clips))
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Path(args.repo) / "configs" / "emage_audio.yaml")
    cfg.data.meta_paths = [str(js_path)]
    cfg.data.test_meta_paths = [str(js_path)]
    cfg.output_dir = str(OUT)
    cfg.exp_name = f"ft_speaker{args.speaker}"
    OmegaConf.save(cfg, OUT / f"config_speaker{args.speaker}.yaml")
    print(f"  {len(clips)} training windows -> {js_path}\n  config -> {OUT}/config_speaker{args.speaker}.yaml")
    raise SystemExit(0)


# =============================================================================
# train
# =============================================================================
class _Stub(types.ModuleType):
    """Stand-in for an optional package (logging, schedulers): any attribute is
    a do-nothing callable, and it can have submodules (diffusers.optimization)."""
    __path__ = []

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return lambda *a, **kw: None


class _StubFinder:
    roots = set()

    def find_spec(self, name, path=None, target=None):
        # importlib.util is imported at the top of the file: importing it here
        # would re-enter this finder and recurse.
        if name.split(".")[0] in self.roots:
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        pass


def import_trainer():
    """Import train_emage_audio.train_val_fn without running its main and
    without the logging/eval extras this box may not have installed."""
    finder = _StubFinder()
    sys.meta_path.insert(0, finder)
    for _ in range(12):
        try:
            return importlib.import_module("train_emage_audio")
        except ModuleNotFoundError as e:
            root = (e.name or "").split(".")[0]
            if not root or root in finder.roots:
                raise
            print(f"  (stubbing missing module '{root}' — only used for logging/eval)")
            finder.roots.add(root)
            for k in [m for m in sys.modules if m == root or m.startswith(root + ".")]:
                del sys.modules[k]
            sys.modules.pop("train_emage_audio", None)
    raise SystemExit("could not import train_emage_audio")


if args.stage == "train":
    from omegaconf import OmegaConf
    from models.emage_audio import EmageAudioModel
    cfg = OmegaConf.load(OUT / f"config_speaker{args.speaker}.yaml")
    tr = import_trainer()
    if not hasattr(tr, "train_val_fn"):
        raise SystemExit("train_emage_audio has no train_val_fn — PantoMatrix changed; send the error")

    # read every wav / motion file once
    import librosa
    import datasets.beat2 as b2
    _wav, _mot = {}, {}
    _orig_load, _orig_beat = librosa.load, b2.beat_format_load

    def cached_load(path, sr=None, **kw):
        k = (str(path), sr)
        if k not in _wav:
            _wav[k] = _orig_load(path, sr=sr, **kw)
        return _wav[k]

    def cached_beat(path, mask=None):
        k = (str(path), str(mask))
        if k not in _mot:
            _mot[k] = _orig_beat(path, mask=mask)
        return _mot[k]
    b2.librosa.load = cached_load
    b2.beat_format_load = cached_beat

    ds = getattr(b2, cfg.data.class_name)(cfg, split="train")
    print(f"  {len(ds)} training windows")
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=0)
    vq = load_vq()
    model = EmageAudioModel.from_pretrained(args.weights).to(DEV)
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    cls_fn = torch.nn.NLLLoss()
    save_dir = OUT / f"speaker{args.speaker}"
    it, t0, hist = 0, time.time(), []
    print(f"  fine-tuning from the released weights: {args.steps} steps, bs {args.bs}, lr {args.lr}")
    while it < args.steps:
        for batch in loader:
            ld = tr.train_val_fn(cfg, batch, model, DEV, mode="train", motion_vq=vq, optimizer=opt,
                                 lr_scheduler=sched, ClsFn=cls_fn, iteration=it)
            hist.append({k: float(v) for k, v in ld.items()})
            it += 1
            if it % 50 == 0 or it == 1:
                avg = {k: np.mean([h[k] for h in hist[-50:]]) for k in hist[-1]}
                eta = (time.time() - t0) / it * (args.steps - it)
                print(f"  step {it}/{args.steps}  " + "  ".join(f"{k} {v:.3f}" for k, v in list(avg.items())[:6])
                      + f"  | {time.time() - t0:.0f}s, eta {eta / 60:.0f} min")
            if it % args.save_every == 0 or it == args.steps:
                model.save_pretrained(save_dir / f"step{it}")
                (save_dir / "loss_history.json").write_text(json.dumps(hist))
            if it >= args.steps:
                break
    model.save_pretrained(save_dir / "final")
    print(f"  saved -> {save_dir}/final")
    raise SystemExit(0)


# =============================================================================
# eval: original vs fine-tuned vs real data
# =============================================================================
if args.stage == "eval":
    import librosa
    import soundfile as sf
    import beat2_tools as bt
    from evaluate_shortcomings import save_emage_npz
    ft_dir = OUT / f"speaker{args.speaker}" / "final"
    orig = bt.Emage(args.repo, args.weights)
    ft = bt.Emage(args.repo, ft_dir, vq_weights=args.weights)
    root = next(Path(args.root).rglob("train_test_split.csv")).parent
    import csv
    test = []
    with open(root / "train_test_split.csv") as fh:
        for row in csv.DictReader(fh):
            if int(row["id"].split("_")[0]) == args.speaker and row["type"].strip() == "test":
                test.append(row["id"].strip())
    test = [i for i in sorted(test) if (root / "smplxflame_30" / f"{i}.npz").exists()][:4]
    if not test:
        raise SystemExit(f"no test clips for speaker {args.speaker} under {root} — run prep first")
    others = sorted(Path(args.subset).rglob("smplxflame_30/*.npz"))
    others = [p for p in others if not p.stem.startswith(f"{args.speaker}_")][:4]
    evals = [(i, root / "smplxflame_30" / f"{i}.npz", root / "wave16k" / f"{i}.wav", "target") for i in test]
    evals += [(p.stem, p, Path(str(p).replace("smplxflame_30", "wave16k").replace(".npz", ".wav")), "other")
              for p in others]
    res = {}
    vid_dir = OUT / f"eval_speaker{args.speaker}"
    vid_dir.mkdir(exist_ok=True)
    for name, npz, wav, kind in evals:
        d = np.load(npz, allow_pickle=True)
        n = min(len(d["poses"]), 60 * FPS)
        audio = librosa.load(str(wav), sr=orig.sr)[0][:int(n / FPS * orig.sr)]
        n = min(n, int(len(audio) / orig.sr * FPS))
        db, sp, _ = bt.audio_feats(audio, orig.sr, n)
        srcs = {"data": (np.asarray(d["poses"][:n], np.float32), np.asarray(d["expressions"][:n], np.float32)),
                "original": orig.gen(audio), "finetuned": ft.gen(audio)}
        res[name] = {"kind": kind}
        for s, (P, E) in srcs.items():
            P, E = P[:n], E[:n]
            J, _ = bt.face_mesh(args.model_folder, P, E)
            sig = bt.face_signals(J)
            bl = bt.blinks(sig["aperture"])
            ra, fr, _ = bt.brow_events(sig["brow"], sig["inner"], exclude=bl)
            g = sig["gap"]
            m = min(len(g), len(sp))
            res[name][s] = {"gap_p95": float(np.percentile(g[:m][sp[:m]], 95)),
                            "gap_std": float(g[:m][sp[:m]].std()),
                            "blinks_per_min": len(bl) / (n / FPS / 60),
                            "raises_per_min": len(ra) / (n / FPS / 60), "frowns_per_min": len(fr) / (n / FPS / 60),
                            **{f"body_{k}": v for k, v in bt.body_stats(P).items()}}
            if kind == "target" and name == test[0]:
                save_emage_npz(vid_dir / f"{s}.npz", P, E, np.zeros((len(P), 3)))
                sf.write(vid_dir / "audio.wav", audio, orig.sr)
        r = res[name]
        print(f"  {name:<22} [{kind}] mouth p95 data {r['data']['gap_p95']:.1f} orig {r['original']['gap_p95']:.1f} "
              f"ft {r['finetuned']['gap_p95']:.1f} | speed {r['data']['body_speed_dps']:.0f}/"
              f"{r['original']['body_speed_dps']:.0f}/{r['finetuned']['body_speed_dps']:.0f} | blinks "
              f"{r['data']['blinks_per_min']:.1f}/{r['original']['blinks_per_min']:.1f}/{r['finetuned']['blinks_per_min']:.1f}")
    keys = ["gap_p95", "gap_std", "blinks_per_min", "raises_per_min", "frowns_per_min",
            "body_speed_dps", "body_amplitude_deg", "body_variety"]
    lines = [f"# Fine-tune on speaker {args.speaker}: before / after", "",
             "Target speaker's held-out test clips (what the fine-tune should match):", "",
             "| measure | target data | original EMAGE | fine-tuned |", "|---|---|---|---|"]
    for kind, title in (("target", None), ("other", "Other speakers (generalisation, not the target):")):
        if title:
            lines += ["", title, "", "| measure | their data | original EMAGE | fine-tuned |", "|---|---|---|---|"]
        sel = [v for v in res.values() if v["kind"] == kind]
        for k in keys:
            vals = [np.mean([v[s][k] for v in sel]) for s in ("data", "original", "finetuned")]
            lines.append(f"| {k} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} |")
    (OUT / f"eval_speaker{args.speaker}.md").write_text("\n".join(lines))
    (OUT / f"eval_speaker{args.speaker}.json").write_text(json.dumps(res, indent=2))
    print("\n" + "\n".join(lines))
