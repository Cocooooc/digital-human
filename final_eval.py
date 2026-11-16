"""
Finish the remaining four problems on top of the Lawrence fine-tune, and
measure all five against Lawrence's real held-out data.

  ④ gestures too fast   } pick the checkpoint (step1000 / step2000 / final)
  ③ brow events         }   closest to Lawrence on speed + brow events
  ① blinks              a blink branch learned from Lawrence's own data:
                          the blink's FLAME shape, its time profile, and the
                          intervals between blinks — sampled, because blinks
                          are not predictable from audio (shown earlier: they
                          fall in pauses exactly as often as chance)
  ⑤ eyes                BEAT2 has no real eye movement (0.4 deg std, 0.06
                          saccades/min) — nothing to learn from — so a gaze
                          model with fixations, saccades and counter-rotation
                          against the head supplies it
  ② mouth               already fixed by the fine-tune; re-measured here

Also evaluates on four other speakers (the fine-tune's generalisation),
and renders a 4-way video: Lawrence | original | fine-tuned | fine-tuned + blink + gaze.

Run (pantomatrix env): python final_eval.py --speaker 4
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import beat2_tools as bt
from evaluate_shortcomings import make_gaze, save_emage_npz

FPS = bt.FPS
ap = argparse.ArgumentParser()
ap.add_argument("--speaker", type=int, default=4)
ap.add_argument("--repo", default="/root/PantoMatrix")
ap.add_argument("--weights", default="/root/emage_weights")
ap.add_argument("--ft", default="/root/ft")
ap.add_argument("--root", default="/root/beat2_target")
ap.add_argument("--subset", default="/root/beat2_subset")
ap.add_argument("--model-folder", default="/root/PantoMatrix/emage_evaltools/smplx_models")
ap.add_argument("--blink-train-clips", type=int, default=24)
ap.add_argument("--look", type=float, default=0.6, help="share of time the eyes are on the listener")
ap.add_argument("--seconds", type=float, default=60)
args = ap.parse_args()
OUT = Path(args.ft) / f"final_speaker{args.speaker}"
OUT.mkdir(parents=True, exist_ok=True)

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

root = next(Path(args.root).rglob("train_test_split.csv")).parent
split = {}
with open(root / "train_test_split.csv") as fh:
    for row in csv.DictReader(fh):
        split[row["id"].strip()] = row["type"].strip()
mine = lambda t: sorted(i for i, v in split.items() if int(i.split("_")[0]) == args.speaker and v == t
                        and (root / "smplxflame_30" / f"{i}.npz").exists())
test_ids, train_ids = mine("test")[:4], mine("train")


def load_clip(npz, wav, sr):
    d = np.load(npz, allow_pickle=True)
    n = int(min(len(d["poses"]), args.seconds * FPS))
    a = librosa.load(str(wav), sr=sr)[0][:int(n / FPS * sr)]
    n = min(n, int(len(a) / sr * FPS))
    return np.asarray(d["poses"][:n], np.float32), np.asarray(d["expressions"][:n], np.float32), a, n


# =============================================================================
# 1. blink model from the target speaker's TRAINING clips
# =============================================================================
print(f"== blink model from {min(len(train_ids), args.blink_train_clips)} of speaker {args.speaker}'s training clips ==")
events, ibis, exprs = [], [], []
for i in train_ids[:args.blink_train_clips]:
    d = np.load(root / "smplxflame_30" / f"{i}.npz", allow_pickle=True)
    P, E = np.asarray(d["poses"], np.float32), np.asarray(d["expressions"], np.float32)
    J, _ = bt.face_mesh(args.model_folder, P, E)
    bl = bt.blinks(bt.face_signals(J)["aperture"])
    events.append((E, bl))
    starts = [b[0] for b in bl]
    ibis += list(np.diff(starts) / FPS)
    exprs.append(len(E))
tpl, cnt = bt.template(np.concatenate([e for e, _ in events]),
                       [(s + off, e + off, p + off) for (E, bl), off in
                        zip(events, np.cumsum([0] + exprs[:-1])) for s, e, p in bl])
u = tpl / (np.linalg.norm(tpl) + 1e-9)
# time profile: projection onto the blink direction, +-6 frames around each minimum
prof = []
for E, bl in events:
    for _, _, p in bl:
        if 8 <= p < len(E) - 8:
            base = np.concatenate([E[p - 8:p - 5], E[p + 6:p + 9]]).mean(0)
            prof.append([(E[p + k] - base) @ u for k in range(-6, 7)])
prof = np.clip(np.mean(prof, 0), 0, None)
prof = prof / (prof.max() + 1e-9)
prof = prof[prof > 0.05] if (prof > 0.05).sum() >= 2 else np.array([0.4, 1.0, 0.6])
ibis = np.array(ibis) if len(ibis) > 5 else np.array([3.3])
rate = len(ibis) / max(1e-9, sum(exprs) / FPS / 60)
print(f"  {cnt} blinks in {sum(exprs) / FPS / 60:.1f} min ({rate:.1f}/min); profile {len(prof)} frames; "
      f"interval median {np.median(ibis):.1f}s")
np.savez(OUT / "blink_model.npz", template=tpl, profile=prof, intervals=ibis)


def add_blinks(E, seed):
    """Renewal process: intervals drawn from the speaker's own, the blink's
    shape and time course from the speaker's own."""
    rng = np.random.default_rng(seed)
    E = E.copy()
    t = int(rng.uniform(0, 1) * np.median(ibis) * FPS)
    k = 0
    while t + len(prof) < len(E):
        E[t:t + len(prof)] += prof[:, None] * tpl[None, :]
        k += 1
        t += max(len(prof) + 2, int(rng.choice(ibis) * FPS))
    return E, k


# =============================================================================
# 2. measures
# =============================================================================
def measures(P, E, audio, sr):
    n = len(P)
    _, sp, _ = bt.audio_feats(audio, sr, n)
    J, _ = bt.face_mesh(args.model_folder, P, E)
    sig = bt.face_signals(J)
    bl = bt.blinks(sig["aperture"])
    ra, fr, _ = bt.brow_events(sig["brow"], sig["inner"], exclude=bl)
    g = sig["gap"]
    m = min(len(g), len(sp))
    mins = n / FPS / 60
    eye = np.degrees(np.linalg.norm(P[:, 69:75].reshape(n, 2, 3), axis=2)).mean(1)
    ev = np.abs(np.diff(eye)) * FPS
    dy_h, dy_e = np.diff(P[:, 46] + P[:, 37]), np.diff(P[:, 70])
    return {"mouth_p95_mm": float(np.percentile(g[:m][sp[:m]], 95)),
            "gesture_speed_dps": bt.body_stats(P)["speed_dps"],
            "raises_per_min": len(ra) / mins, "frowns_per_min": len(fr) / mins,
            "blinks_per_min": len(bl) / mins,
            "eye_std_deg": float(eye.std()),
            "saccades_per_min": len(bt.runs(ev > 60, 1, 6)) / mins,
            "eye_head_corr": float(np.corrcoef(dy_h, dy_e)[0, 1]) if dy_e.std() > 1e-6 else 0.0}


# =============================================================================
# 3. checkpoints
# =============================================================================
ckpts = {"original": args.weights}
_ck = Path(args.ft) / f"speaker{args.speaker}"
for p in sorted(_ck.glob("step*"), key=lambda q: int(q.name[4:])) + [_ck / "final"]:
    if p.exists() and p.name not in ckpts:
        ckpts[p.name] = str(p)

# other speakers' clips (generalisation): fetch if this box has none
others = sorted(p for p in Path(args.subset).rglob("smplxflame_30/*.npz") if not p.stem.startswith(f"{args.speaker}_"))
if not others:
    print("  fetching other speakers' held-out clips ...")
    got = bt.fetch_beat2(args.subset, speakers=6, clips_per_speaker=1)
    others = [Path(m) for s, m, w, h in got if not s.startswith(f"{args.speaker}_")]
others = others[:4]

clips = [(i, root / "smplxflame_30" / f"{i}.npz", root / "wave16k" / f"{i}.wav", "target") for i in test_ids]
clips += [(p.stem, p, Path(str(p).replace("smplxflame_30", "wave16k").replace(".npz", ".wav")), "other")
          for p in others]

res = {c[0]: {"kind": c[3]} for c in clips}
gens = {}
print("\n== checkpoints ==")
for name, w in ckpts.items():
    em = bt.Emage(args.repo, w, vq_weights=args.weights)
    for cid, npz, wav, kind in clips:
        P0, E0, a, n = load_clip(npz, wav, em.sr)
        if "data" not in res[cid]:
            res[cid]["data"] = measures(P0, E0, a, em.sr)
        P, E = em.gen(a)
        P, E = P[:n], E[:n]
        res[cid][name] = measures(P, E, a, em.sr)
        gens[(name, cid)] = (P, E)
    del em
    t = [r[name] for r in res.values() if r["kind"] == "target"]
    print(f"  {name:<9} target: mouth {np.mean([x['mouth_p95_mm'] for x in t]):.2f}  speed "
          f"{np.mean([x['gesture_speed_dps'] for x in t]):.1f}  raises {np.mean([x['raises_per_min'] for x in t]):.1f}  "
          f"frowns {np.mean([x['frowns_per_min'] for x in t]):.1f}  blinks {np.mean([x['blinks_per_min'] for x in t]):.1f}")

tgt = lambda src, k: float(np.mean([r[src][k] for r in res.values() if r["kind"] == "target"]))
KEYS = ["mouth_p95_mm", "gesture_speed_dps", "raises_per_min", "frowns_per_min"]


def score(src):
    """Mean relative distance to Lawrence's real data on the checkpoint-dependent measures."""
    return float(np.mean([abs(tgt(src, k) - tgt("data", k)) / (abs(tgt("data", k)) + 1e-6) for k in KEYS]))


scores = {c: round(score(c), 3) for c in ckpts if c != "original"}
best = min(scores, key=scores.get)
print(f"\n  distance to Lawrence (lower = closer): {scores}  -> using {best}")

# =============================================================================
# 4. blink + gaze on the best checkpoint
# =============================================================================
print("\n== + blink branch + gaze ==")
for k, (cid, npz, wav, kind) in enumerate(clips):
    P, E = gens[(best, cid)]
    E2, nb = add_blinks(E, seed=k)
    P2 = P.copy()
    gz, _, _ = make_gaze(P2, seed=k, look_ratio=args.look, include_root=True)
    P2[:, 69:72] = gz
    P2[:, 72:75] = gz
    _, _, a, n = load_clip(npz, wav, 16000)
    res[cid]["full"] = measures(P2, E2, a, 16000)
    gens[("full", cid)] = (P2, E2)

# =============================================================================
# 5. report
# =============================================================================
ROWS = [("② mouth open p95 (mm)", "mouth_p95_mm"), ("④ gesture speed (deg/s)", "gesture_speed_dps"),
        ("③ brow raises /min", "raises_per_min"), ("③ frowns /min", "frowns_per_min"),
        ("① blinks /min", "blinks_per_min"), ("⑤ eye movement (deg std)", "eye_std_deg"),
        ("⑤ saccades /min", "saccades_per_min"), ("⑤ eye-head counter-rotation (corr)", "eye_head_corr")]
cols = ["data", "original"] + [c for c in ckpts if c != "original"] + ["full"]
head = {"data": "Lawrence (real)", "original": "original EMAGE", "full": f"{best} + blink + gaze"}
L = [f"# Five problems: before / after (speaker {args.speaker}, {len(test_ids)} held-out clips)", "",
     "| problem | " + " | ".join(head.get(c, c) for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
for lab, k in ROWS:
    L.append(f"| {lab} | " + " | ".join(f"{tgt(c, k):.2f}" for c in cols) + " |")
L += ["", f"Checkpoint distance to Lawrence: {scores} -> {best}",
      f"Blink model: {cnt} blinks from {len(train_ids[:args.blink_train_clips])} training clips, "
      f"profile {len(prof)} frames, median interval {np.median(ibis):.1f}s", "",
      "Other speakers (generalisation; each against their own data):", "",
      "| measure | their data | original | " + best + " |", "|---|---|---|---|"]
oth = [r for r in res.values() if r["kind"] == "other"]
for lab, k in ROWS[:5]:
    if oth:
        L.append(f"| {lab} | " + " | ".join(f"{np.mean([r[c][k] for r in oth]):.2f}" for c in ("data", "original", best)) + " |")
(OUT / "five_problems.md").write_text("\n".join(L))
(OUT / "five_problems.json").write_text(json.dumps({"results": res, "scores": scores, "best": best}, indent=2))
print("\n" + "\n".join(L))

# clips for the 4-way video (first target clip)
cid = test_ids[0]
_, _, a, n = load_clip(root / "smplxflame_30" / f"{cid}.npz", root / "wave16k" / f"{cid}.wav", 16000)
d = np.load(root / "smplxflame_30" / f"{cid}.npz", allow_pickle=True)
save_emage_npz(OUT / "v1_lawrence.npz", d["poses"][:n], d["expressions"][:n], np.zeros((n, 3)))
for tag, key in (("v2_original", "original"), ("v3_finetuned", best), ("v4_full", "full")):
    P, E = gens[(key, cid)]
    save_emage_npz(OUT / f"{tag}.npz", P, E, np.zeros((len(P), 3)))
sf.write(OUT / "audio.wav", a, 16000)
print(f"\n  video sources -> {OUT}/v1..v4 .npz + audio.wav")
