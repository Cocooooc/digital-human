"""
Shortcomings check — the four questions the teacher asked, measured.

  1. Long input   — does EMAGE stay sane on 10-60s of audio, and what happens
                    at the seams when a reply is generated sentence by
                    sentence (which is how the live pipeline will run)?
  2. Exaggeration — how big and how fast are the gestures; plus damped
                    versions (x0.7, x0.5) to judge side by side.
  3. Gaze         — EMAGE drives no eyes (both eye joints are 0 on every
                    frame), so the avatar stares. A gaze model is added:
                    mostly on the listener with micro-saccades, occasional
                    look-aways, eyes counter-rotating against head motion,
                    plus blinks.
  4. Latency      — motion-generation time against audio length.

Whether motion looks exaggerated is finally a judgement made by eye. The
script gives the numbers and writes the damped variants so the judgement is
made comparing renders, not from memory of one clip.

Damping, done correctly: scaling axis-angle by 0.7 pulls every joint toward
the SMPL-X T-pose — arms swing UP toward horizontal. The damped variants
instead pull each joint toward this clip's own mean pose, so gestures get
smaller around how this person actually stands.

Found while building this: the "looking down" posture was ours, not EMAGE's.
EMAGE tilts the pelvis back ~19 degrees to balance the spine's forward lean.
Our renderers zeroed the whole global orientation to stop slow left-right
drift, which also threw away that tilt: head pitch went from -9 to -27
degrees and the face was more than 25 degrees off camera on 84% of frames,
out of reach of any eye movement. fix_root() removes only the slow heading
drift. Every variant starts from it.

Run (server, pantomatrix env):
    conda activate pantomatrix
    cd /root/digital-human
    python evaluate_shortcomings.py --generate --out /root/eval
  analysis + variants for an existing clip, no generation:
    python evaluate_shortcomings.py --analyze /root/render_test/out/clip5s_output.npz --out /root/eval
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

FPS = 30
UPPER = {13: "left_collar", 14: "right_collar", 16: "left_shoulder",
         17: "right_shoulder", 18: "left_elbow", 19: "right_elbow",
         20: "left_wrist", 21: "right_wrist"}
# Head orientation from the spine chain. Joint 0 (global orientation) is left
# out on purpose: both renderers and the VRM test bench zero it, so gaze has
# to be computed in the same frame or the eyes aim at the wrong place.
HEAD_CHAIN = [3, 6, 9, 12, 15]
BODY = list(range(22))
EYE_L, EYE_R = 69, 72
MAX_EYE_DEG = 25.0


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------
def rots(poses, j):
    return R.from_rotvec(poses[:, 3 * j:3 * j + 3])


def step_deg(poses, j):
    """Rotation between consecutive frames, degrees."""
    r = rots(poses, j)
    return np.degrees((r[:-1].inv() * r[1:]).magnitude())


def speed_dps(poses, joints):
    """Mean angular speed over `joints`, deg/s, per frame step."""
    return np.mean([step_deg(poses, j) for j in joints], axis=0) * FPS


def deviation_deg(poses, j):
    """How far each frame sits from the clip's mean pose for joint j."""
    r = rots(poses, j)
    return np.degrees((r.mean().inv() * r).magnitude())


def damp(poses, s, joints=BODY):
    """Shrink motion around the clip's own mean pose (NOT toward T-pose)."""
    out = poses.copy()
    for j in joints:
        r = rots(poses, j)
        m = r.mean()
        delta = (m.inv() * r).as_rotvec() * s
        out[:, 3 * j:3 * j + 3] = (m * R.from_rotvec(delta)).as_rotvec()
    return out


def head_world(poses, include_root=False):
    hw = R.identity(len(poses))
    for j in ([0] + HEAD_CHAIN if include_root else HEAD_CHAIN):
        hw = hw * rots(poses, j)
    return hw


def head_direction(poses, include_root):
    """Where the face points, degrees: yaw, pitch (+ = up), angle off camera."""
    f = head_world(poses, include_root).apply([0, 0, 1])
    return (np.degrees(np.arctan2(f[:, 0], f[:, 2])),
            np.degrees(np.arcsin(np.clip(f[:, 1], -1, 1))),
            np.degrees(np.arccos(np.clip(f[:, 2], -1, 1))))


def fix_root(poses, window_s=2.0):
    """Remove the slow left-right drift of the body, keep everything else.

    The renderers used to zero the global orientation to stop the body
    slowly turning away. But EMAGE's global orientation also carries a
    steady ~19 degree backward pelvis tilt that balances the spine's forward
    lean; zeroing it leaves the lean uncompensated and the head looks ~27
    degrees down. Here only the heading's slow component (a 2s moving
    average) is taken out, as a rotation about the vertical axis: pitch,
    roll and quick sways stay as generated.
    """
    n = len(poses)
    r = rots(poses, 0)
    f = r.apply([0, 0, 1])
    heading = np.unwrap(np.arctan2(f[:, 0], f[:, 2]))
    k = int(window_s * FPS) | 1
    slow = np.convolve(np.pad(heading, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    undo = R.from_rotvec(np.stack([np.zeros(n), -slow, np.zeros(n)], 1))
    out = poses.copy()
    out[:, 0:3] = (undo * r).as_rotvec()
    return out


# ---------------------------------------------------------------------------
# Gaze + blink model
# ---------------------------------------------------------------------------
def _dir(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.stack([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)], -1)


def make_gaze(poses, seed=0, look_ratio=0.7, include_root=True):
    """Eye rotations (n,3), with two kinds of fixation that use different frames.

    On the listener (70%): the target is the camera, fixed in world space and
    converted into the head's frame every frame, so when the head turns the
    eyes turn back the other way — the counter-rotation that holds a real
    gaze on someone. Small jitter, so it is never a dead stare.
    Looking away (30%): 6-12 degrees off the head's OWN forward direction,
    more often up than down. People avert their eyes relative to where their
    head is pointing; aiming the aversion in camera space instead stacks it
    on top of the head's own offset and drives the eyes into their limit.
    Fixations last 0.2-2s (lognormal around 0.6s); saccades take 3 frames.
    """
    n = len(poses)
    rng = np.random.default_rng(seed)
    hw_inv = head_world(poses, include_root).inv()
    want = np.zeros((n, 3))
    fixations = []
    t = 0
    while t < n:
        dur = int(np.clip(rng.lognormal(np.log(0.6 * FPS), 0.5), 0.2 * FPS, 2.0 * FPS))
        end = min(n, t + dur)
        if rng.random() < look_ratio:
            kind = "listener"
            d = _dir(rng.normal(0, 1.5), rng.normal(0, 1.0))
            want[t:end] = hw_inv[t:end].apply(np.tile(d, (end - t, 1)))
        else:
            kind = "away"
            d = _dir(rng.choice([-1, 1]) * rng.uniform(6, 12), rng.uniform(-3, 7))
            want[t:end] = d
        fixations.append({"start": t, "frames": end - t, "kind": kind})
        t = end

    # saccades: blend from where the eye was into each new fixation over 3 frames
    gaze = want.copy()
    for f in fixations[1:]:
        s0 = f["start"]
        for k in range(min(3, f["frames"])):
            w = (k + 1) / 3
            v = (1 - w) * gaze[s0 - 1] + w * want[s0 + k]
            gaze[s0 + k] = v / np.linalg.norm(v)

    axis = np.stack([-gaze[:, 1], gaze[:, 0], np.zeros(n)], 1)        # +Z x d
    sn = np.linalg.norm(axis, axis=1, keepdims=True)
    ang = np.arctan2(sn[:, 0], gaze[:, 2])
    lim = np.radians(MAX_EYE_DEG)
    clamped = ang > lim
    ang = np.minimum(ang, lim)
    rv = np.where(sn > 1e-9, axis / np.maximum(sn, 1e-9), 0.0) * ang[:, None]
    return rv.astype(np.float32), fixations, float(clamped.mean())


def make_blinks(n, seed=0):
    """0..1 per frame: a blink every 2-6s, 5 frames (~170ms) each."""
    rng = np.random.default_rng(seed + 1)
    b = np.zeros(n, np.float32)
    shape = np.array([0.3, 0.8, 1.0, 0.8, 0.3], np.float32)
    t = int(rng.uniform(1, 3) * FPS)
    count = 0
    while t + 5 <= n:
        b[t:t + 5] = shape
        count += 1
        t += int(rng.uniform(2, 6) * FPS)
    return b, count


# ---------------------------------------------------------------------------
# Analysis of one clip
# ---------------------------------------------------------------------------
def analyse(poses):
    n = len(poses)
    sp = speed_dps(poses, UPPER)
    win = 5 * FPS
    windows = [float(sp[i:i + win].mean()) for i in range(0, len(sp) - win // 2, win)]
    med = float(np.median(windows)) if windows else 0.0
    weak = [i for i, w in enumerate(windows) if med > 0 and w < 0.3 * med]
    dev = {UPPER[j]: float(np.percentile(deviation_deg(poses, j), 95)) for j in UPPER}
    steps = np.concatenate([step_deg(poses, j) for j in UPPER])
    eye_std = float(np.degrees(np.linalg.norm(poses[:, EYE_L:EYE_L + 3], axis=1)).std())
    root_dev = float(np.percentile(deviation_deg(poses, 0), 95))
    head = {}
    for label, inc in (("root_zeroed", False), ("root_kept", True)):
        _, pt, off = head_direction(poses, inc)
        head[label] = {"pitch_mean_deg": round(float(pt.mean()), 1),
                       "off_camera_median_deg": round(float(np.median(off)), 1),
                       "frames_over_25deg": round(float((off > 25).mean()), 3)}
    return {
        "frames": n, "seconds": round(n / FPS, 2),
        "upper_speed_dps": {"p50": float(np.percentile(sp, 50)), "p95": float(np.percentile(sp, 95))},
        "energy_per_5s_window_dps": [round(w, 1) for w in windows],
        "weak_windows": weak,
        "deviation_p95_deg": {k: round(v, 1) for k, v in dev.items()},
        "frame_step_deg": {"p50": float(np.percentile(steps, 50)), "p99": float(np.percentile(steps, 99))},
        "root_deviation_p95_deg": round(root_dev, 1),
        "eye_std_deg": round(eye_std, 3),
        "head_direction": head,
    }


def write_variant(src, poses, path, extra=None):
    out = {k: src[k] for k in src.files}
    out["poses"] = poses.astype(np.float32)
    if extra:
        out.update(extra)
    np.savez(path, **out)


# ---------------------------------------------------------------------------
# EMAGE (only for --generate)
# ---------------------------------------------------------------------------
def load_emage(repo, weights):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(repo))
    from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel
    dev = "cuda"
    src = str(weights)
    model = EmageAudioModel.from_pretrained(src).to(dev).eval()
    parts = {k: EmageVQVAEConv.from_pretrained(src, subfolder=f"emage_vq/{k}").to(dev).eval()
             for k in ("face", "upper", "hands", "lower")}
    glob = EmageVAEConv.from_pretrained(src, subfolder="emage_vq/global").to(dev).eval()
    vq = EmageVQModel(face_model=parts["face"], upper_model=parts["upper"],
                      hands_model=parts["hands"], lower_model=parts["lower"],
                      global_model=glob).to(dev).eval()

    def gen(audio_np):
        a = torch.from_numpy(audio_np).float().to(dev).unsqueeze(0)
        with torch.no_grad():
            lat = model.inference(a, torch.zeros(1, 1).long().to(dev), vq,
                                  masked_motion=None, mask=None)
            c = model.cfg
            pick = lambda k: torch.max(F.log_softmax(lat[k], dim=2), dim=2)[1]
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
        torch.cuda.synchronize()
        return (pred["motion_axis_angle"][0].cpu().numpy(),
                pred["expression"][0].cpu().numpy(),
                pred["trans"][0].cpu().numpy())

    gen(np.zeros(model.cfg.audio_sr, np.float32))     # warm-up, not timed
    return gen, model.cfg.audio_sr


def save_emage_npz(path, poses, expr, trans):
    np.savez(path, poses=poses.astype(np.float32), expressions=expr.astype(np.float32),
             trans=trans.astype(np.float32), betas=np.zeros(300, np.float32),
             mocap_frame_rate=np.int32(FPS))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--analyze", nargs="*", default=[])
    ap.add_argument("--out", default="/root/eval")
    ap.add_argument("--audio", default="/root/PantoMatrix/examples/audio/2_scott_0_103_103_28s.wav")
    ap.add_argument("--repo", default="/root/PantoMatrix")
    ap.add_argument("--weights", default="/root/emage_weights")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"clips": {}, "latency": [], "seams": None}
    clips = [Path(p) for p in args.analyze]

    if args.generate:
        import librosa
        import soundfile as sf
        print("loading EMAGE…")
        gen, sr = load_emage(Path(args.repo), Path(args.weights))
        full, _ = librosa.load(args.audio, sr=sr)
        # 56s = the 28s example twice: a stress test of length, not of content.
        sources = {f"{s}s": full[:s * sr] for s in (5, 10, 20, 28)}
        sources["56s"] = np.concatenate([full, full])

        print("\n== 1. length and latency ==")
        for name, a in sources.items():
            t0 = time.time()
            poses, expr, trans = gen(a)
            dt = time.time() - t0
            dur = len(a) / sr
            p = out / f"len_{name}.npz"
            save_emage_npz(p, poses, expr, trans)
            sf.write(out / f"len_{name}.wav", a, sr)
            clips.append(p)
            report["latency"].append({"audio_s": round(dur, 2), "infer_s": round(dt, 3),
                                      "rtf": round(dt / dur, 4)})
            print(f"  {name:>4}: {dt:.3f}s for {dur:.1f}s audio  RTF {dt/dur:.4f}")

        print("\n== 2. seams: 20s as four separately generated 5s sentences ==")
        whole = np.load(out / "len_20s.npz")["poses"]
        parts = [gen(full[i * 5 * sr:(i + 1) * 5 * sr])[0] for i in range(4)]
        jumps = []
        for a_, b_ in zip(parts[:-1], parts[1:]):
            jumps.append(float(np.mean([np.degrees((rots(a_[-1:], j).inv() * rots(b_[:1], j)).magnitude()[0])
                                        for j in UPPER])))
        steps = np.concatenate([step_deg(whole, j) for j in UPPER])
        p99 = float(np.percentile(steps, 99))
        report["seams"] = {"jump_deg": [round(x, 1) for x in jumps],
                           "within_clip_step_p50_deg": round(float(np.percentile(steps, 50)), 2),
                           "within_clip_step_p99_deg": round(p99, 2),
                           "jump_over_p99": [round(x / p99, 1) for x in jumps]}
        print(f"  seam jumps {report['seams']['jump_deg']} deg; normal frame step p99 {p99:.2f} deg "
              f"-> {report['seams']['jump_over_p99']} x")

    if not clips:
        sys.exit("nothing to analyse: pass --generate or --analyze <npz>")

    print("\n== 3. per clip: energy over time, amplitude, gaze ==")
    for p in clips:
        src = np.load(p, allow_pickle=True)
        poses = np.asarray(src["poses"], np.float32)
        a = analyse(poses)
        report["clips"][p.stem] = a
        print(f"\n  {p.stem}: {a['seconds']}s")
        print(f"    upper-body speed p50 {a['upper_speed_dps']['p50']:.0f} / p95 {a['upper_speed_dps']['p95']:.0f} deg/s")
        print(f"    energy per 5s window {a['energy_per_5s_window_dps']}  weak windows {a['weak_windows'] or 'none'}")
        print(f"    p95 deviation from own mean pose: {a['deviation_p95_deg']}")
        print(f"    root p95 deviation {a['root_deviation_p95_deg']} deg   eye std {a['eye_std_deg']} deg")

        # Variants. All start from the root fix, and are meant to be rendered
        # WITH the global orientation kept (face_export.py --keep-root). Gaze
        # is computed last, on the final body, because damping moves the head.
        base = fix_root(poses)
        _, pt, off = head_direction(base, True)
        a["root_fixed"] = {"pitch_mean_deg": round(float(pt.mean()), 1),
                           "frames_over_25deg": round(float((off > 25).mean()), 3)}
        write_variant(src, base, out / f"{p.stem}_rootfix.npz")
        blink, nb = make_blinks(len(poses))

        def with_gaze(body):
            eye, fix, cf = make_gaze(body, seed=0, include_root=True)
            g = body.copy()
            g[:, EYE_L:EYE_L + 3] = eye
            g[:, EYE_R:EYE_R + 3] = eye
            return g, fix, cf, eye

        for sc in (0.7, 0.5):
            d = damp(base, sc)
            write_variant(src, d, out / f"{p.stem}_scale{sc}.npz")
            a[f"scale{sc}_deviation_p95_deg"] = {UPPER[j]: round(float(np.percentile(deviation_deg(d, j), 95)), 1)
                                                 for j in (16, 18)}
        g, fix, clamp_frac, eye = with_gaze(base)
        write_variant(src, g, out / f"{p.stem}_gaze.npz", {"blink": blink})
        t, _, _, _ = with_gaze(damp(base, 0.7))
        write_variant(src, t, out / f"{p.stem}_tuned.npz", {"blink": blink})
        gaze_info = {"fixations": len(fix), "look_aways": sum(f["kind"] == "away" for f in fix),
                     "blinks": nb, "eye_clamped_frac": round(clamp_frac, 3),
                     "eye_std_deg": round(float(np.degrees(np.linalg.norm(eye, axis=1)).std()), 2)}
        a["gaze_variant"] = gaze_info
        hd = a["head_direction"]
        print(f"    head pitch: root zeroed {hd['root_zeroed']['pitch_mean_deg']} deg "
              f"({hd['root_zeroed']['frames_over_25deg']*100:.0f}% frames >25 off camera) | "
              f"root kept {hd['root_kept']['pitch_mean_deg']} deg | root fixed {a['root_fixed']['pitch_mean_deg']} deg "
              f"({a['root_fixed']['frames_over_25deg']*100:.0f}% >25)")
        print(f"    gaze variant: {gaze_info}")

    (out / "shortcomings_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  report -> {out/'shortcomings_report.json'}")
    print(f"  variants -> {out}/*_rootfix.npz *_scale0.7.npz *_scale0.5.npz *_gaze.npz *_tuned.npz")
    print("  render variants WITH the root kept:  python face_export.py <variant>.npz --audio <wav> --side-by-side --keep-root")


if __name__ == "__main__":
    main()
