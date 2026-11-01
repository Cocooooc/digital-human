"""
Before / after for the three live-motion fixes in motion_post.py:
idle motion, smaller gestures, smooth sentence seams.

Builds one simulated conversation turn the way the live pipeline produces it
— the 28s PantoMatrix example cut into sentences at its quietest points, each
sentence sent to EMAGE on its own — with pauses where the avatar would be
listening or thinking:

    idle 2s | sentence 1 | sentence 2 | pause 1.5s | sentence 3 | ... | idle 3s

  before  what the system does now: sentences concatenated as generated, the
          avatar holding a frozen frame whenever nobody is speaking, gestures
          at full size. (Root drift is fixed in both, so the comparison is
          only about the three fixes.)
  after   motion_post.Session: idle motion in every pause, gestures x0.7
          around a fixed rest pose, each seam faded over ~0.3-0.5s.

Run (server, pantomatrix env):
    conda activate pantomatrix && source /root/.render_env
    cd /root/digital-human
    python demo_postprocess.py --out /root/eval
Then render the two .npz it prints, with --keep-root.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import motion_post as mp
from evaluate_shortcomings import load_emage, save_emage_npz

FPS = mp.FPS

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/root/eval")
ap.add_argument("--audio", default="/root/PantoMatrix/examples/audio/2_scott_0_103_103_28s.wav")
ap.add_argument("--repo", default="/root/PantoMatrix")
ap.add_argument("--weights", default="/root/emage_weights")
ap.add_argument("--scale", type=float, default=0.7)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

print("loading EMAGE…")
gen, sr = load_emage(Path(args.repo), Path(args.weights))
full, _ = librosa.load(args.audio, sr=sr)

# Rest pose: the mean of the whole clip generated in one pass, which is also
# what emage_server computes at start-up.
whole_poses, _, _ = gen(full)
rest = mp.rest_pose_from(whole_poses)

# Cut into sentences at the quietest 50ms within +-1s of every 6s.
hop = int(0.05 * sr)
rms = np.array([np.sqrt(np.mean(full[i:i + hop] ** 2)) for i in range(0, len(full) - hop, hop)])
cuts = [0]
for target in np.arange(6, len(full) / sr - 3, 6):
    lo, hi = int((target - 1) / 0.05), int((target + 1) / 0.05)
    cuts.append((lo + int(np.argmin(rms[lo:hi]))) * hop)
cuts.append(len(full))
sentences = []
for a, b in zip(cuts[:-1], cuts[1:]):
    poses, expr, _ = gen(full[a:b])
    n = len(poses)
    audio = np.zeros(int(round(n / FPS * sr)), np.float32)       # audio trimmed/padded to the motion
    seg = full[a:b][:len(audio)]
    audio[:len(seg)] = seg
    sentences.append((poses, expr, audio))
print("sentences:", [round(len(s[0]) / FPS, 1) for s in sentences], "s")

# The turn: (kind, seconds or sentence index)
plan = [("idle", 2.0), ("say", 0), ("say", 1), ("idle", 1.5)] + \
       [("say", i) for i in range(2, len(sentences))] + [("idle", 3.0)]

# --- before -----------------------------------------------------------------
b_poses, b_expr, audio, marks = [], [], [], []
last = None
for kind, v in plan:
    if kind == "idle":
        n = int(v * FPS)
        hold = mp.fix_root(sentences[0][0])[0] if last is None else last
        b_poses.append(np.tile(hold, (n, 1)))
        b_expr.append(np.zeros((n, 100), np.float32))
        audio.append(np.zeros(int(n / FPS * sr), np.float32))
    else:
        p, e, a = sentences[v]
        p = mp.fix_root(p)
        b_poses.append(p)
        b_expr.append(e)
        audio.append(a)
        last = p[-1]
    marks.append((kind, sum(len(x) for x in b_poses)))
before = np.concatenate(b_poses).astype(np.float32)
before_e = np.concatenate(b_expr).astype(np.float32)
wav = np.concatenate(audio)

# --- after ------------------------------------------------------------------
s = mp.Session(rest, scale=args.scale)
a_poses, a_expr = [], []
for kind, v in plan:
    p, e = s.idle(int(v * FPS)) if kind == "idle" else s.speak(sentences[v][0], sentences[v][1])
    a_poses.append(p)
    a_expr.append(e)
after = np.concatenate(a_poses)
after_e = np.concatenate(a_expr)
assert len(after) == len(before), (len(after), len(before))

# --- measure ----------------------------------------------------------------
bounds = [m[1] - 1 for m in marks[:-1]]
speech = np.zeros(len(before), bool)
steady_idle = np.zeros(len(before), bool)     # idle, after the 1s fade out of speech
start = 0
for kind, end in marks:
    if kind == "say":
        speech[start:end] = True
    else:
        steady_idle[start + FPS:end] = True
    start = end
st_b, st_a = mp.step_deg(before), mp.step_deg(after)
p99 = float(np.percentile(mp.step_deg(mp.fix_root(whole_poses)), 99))


def idle_speed(steps):
    return float(steps[steady_idle[1:]].mean() * FPS)


dev_b = mp.deviation_p95(before[speech], rest)
dev_a = mp.deviation_p95(after[speech], rest)
names = {13: "L collar", 14: "R collar", 16: "L shoulder", 17: "R shoulder",
         18: "L elbow", 19: "R elbow", 20: "L wrist", 21: "R wrist"}
report = {
    "sentences_s": [round(len(x[0]) / FPS, 2) for x in sentences],
    "boundary_step_deg": {"before": [round(float(st_b[i]), 1) for i in bounds],
                          "after": [round(float(st_a[i]), 1) for i in bounds],
                          "normal_p99": round(p99, 2)},
    "largest_step_near_boundaries_deg": {
        "before": round(float(max(st_b[i:i + 30].max() for i in bounds)), 1),
        "after": round(float(max(st_a[i:i + 30].max() for i in bounds)), 1)},
    "idle_upper_body_speed_dps": {"before": round(idle_speed(st_b), 2), "after": round(idle_speed(st_a), 2)},
    "speech_p95_deviation_from_rest_deg": {
        names[j]: [round(dev_b[j], 1), round(dev_a[j], 1)] for j in dev_b},
    "scale": args.scale,
}
(out / "convo_report.json").write_text(json.dumps(report, indent=2))

save_emage_npz(out / "convo_before.npz", before, before_e, np.zeros((len(before), 3)))
save_emage_npz(out / "convo_after.npz", after, after_e, np.zeros((len(after), 3)))
sf.write(out / "convo.wav", wav, sr)

print("\n== seams (deg moved between two frames at each boundary) ==")
print("  before", report["boundary_step_deg"]["before"])
print("  after ", report["boundary_step_deg"]["after"], f"   normal p99 {p99:.1f}")
print("  largest step within 1s of a boundary:",
      report["largest_step_near_boundaries_deg"]["before"], "->",
      report["largest_step_near_boundaries_deg"]["after"])
print("\n== idle (upper-body speed while nobody speaks, deg/s) ==")
print("  before", report["idle_upper_body_speed_dps"]["before"], "(frozen)")
print("  after ", report["idle_upper_body_speed_dps"]["after"])
print(f"\n== gesture size while speaking (p95 deg from rest; before -> after, x{args.scale}) ==")
for k, (x, y) in report["speech_p95_deviation_from_rest_deg"].items():
    print(f"  {k:<11} {x:5.1f} -> {y:5.1f}")
print(f"\n  wrote {out}/convo_before.npz  convo_after.npz  convo.wav  convo_report.json")
print("  render:")
print(f"    python face_export.py {out}/convo_before.npz --audio {out}/convo.wav --side-by-side --keep-root --label before")
print(f"    python face_export.py {out}/convo_after.npz --audio {out}/convo.wav --side-by-side --keep-root --label \"after: idle + 0.7x + smooth seams\"")
