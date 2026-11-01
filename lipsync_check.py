"""
Lip-sync check — the mouth problems the teacher listed, measured per clip.

  * mouth not moving while speaking  ("有时口型没有变")
  * drawn-out vowels                  ("拖音"): the sound holds steady, the mouth
                                       should hold open — not close, not flap
  * vibrato                           ("颤音"): a wobble in pitch should not
                                       make the jaw chatter
  * plus two the notes imply: mouth open during silence, and audio/mouth
    offset (is the mouth early or late)

How the mouth is measured: SMPL-X forward kinematics on EMAGE's jaw pose +
FLAME expression, then the distance between the inner-lip landmarks (upper
lip centre <-> lower lip centre) in millimetres. That is what a viewer sees;
jaw angle alone misses expression-driven lip movement.

How "too little" is decided: not by a fixed number. The normal-speech clips
set the baseline (how much the mouth moves in half a second of ordinary
speech) and every other clip is judged against it, so thresholds follow the
model rather than a guess.

Test material, all synthetic (no real voices — publishable): Kokoro sentences
at normal, fast and slow speed, a bilabial tongue-twister (p/b/m must close
the lips), pauses, plus two made here: a vowel time-stretched to a long hold,
and the same with vibrato added (5.5 Hz pitch wobble). --wav-dir adds any
folder of real recordings, e.g. an open dataset.

Two stages (Kokoro in base env, EMAGE in pantomatrix):
    python lipsync_check.py tts --out /root/lipsync
    python lipsync_check.py analyze --out /root/lipsync [--wav-dir /some/wavs]
lipsync_check.sh runs both.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FPS = 30
TTS_SR = 24000

SENTENCES = [
    # id, text, speed
    ("normal_1", "The weather is nice today, so let's take a walk in the park after lunch.", 1.0),
    ("normal_2", "Could you tell me how this system works, one step at a time?", 1.0),
    ("normal_3", "My name is Alex. I study computer science, and in my free time I like "
                 "hiking, cooking, and reading about the history of science.", 1.0),
    ("fast", "Could you tell me how this system works, one step at a time?", 1.45),
    ("slow_drawl", "Well, so, I was thinking, maybe we could go home now.", 0.6),
    ("pauses", "I think... the answer is... yes. But... let me check again.", 0.9),
    ("bilabial", "Peter Piper picked a peck of pickled peppers, maybe more, maybe a bit.", 1.0),
    ("vowel_src", "Ah.", 0.8),          # stretched into held_vowel / vibrato in stage 2
]
BASELINE = ("normal_1", "normal_2", "normal_3")

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["tts", "analyze"])
ap.add_argument("--out", default="/root/lipsync")
ap.add_argument("--voice", default="af_heart")
ap.add_argument("--wav-dir", default="", help="extra recordings to check (any sample rate)")
ap.add_argument("--repo", default="/root/PantoMatrix")
ap.add_argument("--weights", default="/root/emage_weights")
ap.add_argument("--model-folder", default="/root/PantoMatrix/emage_evaltools/smplx_models")
args = ap.parse_args()
out = Path(args.out)
(out / "wav").mkdir(parents=True, exist_ok=True)

if args.stage == "tts":
    import soundfile as sf
    from kokoro import KPipeline
    pipe = KPipeline(lang_code=args.voice[0])
    for cid, text, speed in SENTENCES:
        chunks = [np.asarray(a, np.float32) for _, _, a in pipe(text, voice=args.voice, speed=speed)]
        a = np.concatenate(chunks)
        sf.write(out / "wav" / f"{cid}.wav", a, TTS_SR)
        print(f"  {cid:<11} {len(a) / TTS_SR:5.1f}s  speed {speed}  '{text[:50]}'")
    raise SystemExit(0)


# =============================================================================
# analyze
# =============================================================================
import librosa  # noqa: E402
import soundfile as sf  # noqa: E402


# ---- derived test audio --------------------------------------------------------
def held_vowel(a, sr, target_s=2.5):
    """Stretch the voiced core of a short 'Ah' to ~target_s: a drawn-out vowel."""
    rms = librosa.feature.rms(y=a, frame_length=1024, hop_length=256)[0]
    on = np.where(rms > rms.max() * 0.3)[0]
    core = a[on[0] * 256:(on[-1] + 1) * 256]
    rate = max(0.05, len(core) / sr / target_s)
    held = librosa.effects.time_stretch(core, rate=rate)
    fade = int(0.03 * sr)
    held[:fade] *= np.linspace(0, 1, fade)
    held[-fade:] *= np.linspace(1, 0, fade)
    pad = np.zeros(int(0.4 * sr), np.float32)
    return np.concatenate([pad, held, pad]).astype(np.float32)


def add_vibrato(a, sr, rate_hz=5.5, depth_ms=0.6):
    """Pitch vibrato via a sinusoidally modulated delay line (+-~3% pitch)."""
    t = np.arange(len(a)) / sr
    d = depth_ms / 1000 * (1 + np.sin(2 * np.pi * rate_hz * t)) / 2
    return np.interp(t - d, t, a, left=0.0).astype(np.float32)


# ---- audio features at 30 fps ----------------------------------------------------
def audio_features(a, sr, n):
    hop = sr / FPS
    idx = (np.arange(n) * hop).astype(int)
    win = int(hop * 2)
    rms = np.array([np.sqrt(np.mean(a[i:i + win] ** 2) + 1e-12) for i in idx])
    db = 20 * np.log10(rms + 1e-9)
    # Speech = loud enough, with dips shorter than 0.2s (between syllables,
    # stop consonants) bridged: those are inside a word, not silence.
    speech = fill_gaps(db > db.max() - 35, int(0.2 * FPS))
    f0, voiced, _ = librosa.pyin(a, fmin=65, fmax=500, sr=sr,
                                 frame_length=2048, hop_length=int(hop))
    m = min(n, len(voiced))
    v = np.zeros(n, bool)
    v[:m] = voiced[:m]
    f = np.full(n, np.nan)
    f[:m] = f0[:m]
    return db, speech, v & speech, f


def fill_gaps(mask, max_gap):
    m = mask.copy()
    idx = np.where(mask)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap:
            m[a:b] = True
    return m


def runs(mask, min_len):
    """[start, end) of True runs at least min_len long."""
    out, s = [], None
    for i, x in enumerate(np.append(mask, False)):
        if x and s is None:
            s = i
        elif not x and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    return out


def window_std(x, w=15):
    return np.array([x[i:i + w].std() for i in range(0, max(1, len(x) - w + 1))])


# ---- mouth aperture from the mesh ------------------------------------------------
# SMPL-X output joints: 55 skeleton + 21 extra (nose, eyes, ears, feet,
# fingertips) + 51 face landmarks = iBUG-68 points 17..67. So iBUG k sits at
# joint 76 + (k - 17). Inner lips: 62 = upper centre, 66 = lower centre,
# 60/64 = inner corners. Checked at run time: the gap must open with the jaw.
UPPER_LIP, LOWER_LIP = 76 + 62 - 17, 76 + 66 - 17
LEFT_CORNER, RIGHT_CORNER = 76 + 64 - 17, 76 + 60 - 17


def mouth_aperture(poses, expr):
    import torch
    import smplx
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(poses)
    model = smplx.create(args.model_folder, model_type="smplx", gender="neutral", ext="npz",
                         num_betas=300, num_expression_coeffs=100, use_pca=False,
                         use_face_contour=False, flat_hand_mean=False, batch_size=n).to(dev).eval()
    z = lambda k: torch.zeros(n, k, device=dev)
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev)
    with torch.no_grad():
        o = model(betas=z(300), transl=z(3), global_orient=z(3), body_pose=z(63),
                  jaw_pose=T(poses[:, 66:69]), expression=T(expr),
                  leye_pose=z(3), reye_pose=z(3), left_hand_pose=z(45), right_hand_pose=z(45))
    j = o.joints.cpu().numpy()
    assert j.shape[1] >= 127, f"expected 127 joints with face landmarks, got {j.shape[1]}"
    gap = np.linalg.norm(j[:, UPPER_LIP] - j[:, LOWER_LIP], axis=1) * 1000
    width = np.linalg.norm(j[:, LEFT_CORNER] - j[:, RIGHT_CORNER], axis=1) * 1000
    return gap, width


# ---- per-clip checks ---------------------------------------------------------------
def best_lag(env, gap, max_ms=300):
    """Positive = mouth moves AFTER the sound."""
    e = (env - env.mean()) / (env.std() + 1e-9)
    g = (gap - gap.mean()) / (gap.std() + 1e-9)
    best = (0, -2.0)
    for L in range(-int(max_ms / 1000 * FPS), int(max_ms / 1000 * FPS) + 1):
        a, b = (e[:len(e) - L], g[L:]) if L >= 0 else (e[-L:], g[:len(g) + L])
        if len(a) > 10:
            c = float(np.corrcoef(a, b)[0, 1])
            if c > best[1]:
                best = (L, c)
    return round(best[0] / FPS * 1000), round(best[1], 2)


def check_clip(name, a, sr, gap, base):
    n = len(gap)
    db, speech, voiced, f0 = audio_features(a, sr, n)
    env = np.convolve(np.clip(db - (db.max() - 40), 0, None), np.ones(3) / 3, mode="same")
    lag_ms, corr = best_lag(env, gap)
    issues = []

    # 1. held vowels: steady pitch and loudness for >= 0.4 s
    held = []
    in_held = np.zeros(n, bool)
    for s, e in runs(voiced, int(0.4 * FPS)):
        ff = f0[s:e][~np.isnan(f0[s:e])]
        if len(ff) < 5:
            continue
        steady_pitch = np.std(np.log2(ff)) * 12 < 1.5          # semitones
        steady_level = db[s:e].std() < 6.0
        if not (steady_pitch and steady_level):
            continue
        g = gap[s + 3:e - 3] if e - s > 12 else gap[s:e]     # skip onset/release
        h = {"start_s": round(s / FPS, 2), "end_s": round(e / FPS, 2),
             "gap_mean_mm": round(float(g.mean()), 1), "gap_min_mm": round(float(g.min()), 1),
             "gap_std_mm": round(float(g.std()), 1)}
        held.append(h)
        in_held[s:e] = True
        if g.min() < base["closed_mm"]:
            issues.append(("closes during a held vowel", s, e,
                           f"lip gap dips to {g.min():.1f}mm while the sound holds"))
        if g.std() > base["win_std_p50"]:
            issues.append(("mouth flaps during a held vowel", s, e,
                           f"lip gap std {g.std():.1f}mm while the sound holds steady "
                           f"(typical speech window {base['win_std_p50']:.1f})"))
        if g.mean() < base["speech_gap_p25"]:
            issues.append(("held vowel with mouth barely open", s, e,
                           f"mean gap {g.mean():.1f}mm (normal speech p25 {base['speech_gap_p25']:.1f})"))

    # 2. mouth frozen while speaking: half-second speech windows in which the
    #    lips move less than 40% of what a typical normal-speech window does.
    #    Held vowels are excluded — a steady mouth is right for a steady sound.
    frozen = np.zeros(n, bool)
    for i in range(0, n - 15 + 1, 5):
        if (speech[i:i + 15].mean() > 0.8 and in_held[i:i + 15].mean() < 0.5
                and gap[i:i + 15].std() < base["frozen_std"]):
            frozen[i:i + 15] = True
    frozen_spans = runs(frozen, 10)
    sp = max(1, speech.sum())
    frozen_pct = round(100 * (frozen & speech).sum() / sp, 1)
    for s, e in frozen_spans:
        issues.append(("mouth static while speaking", s, e,
                       f"lip gap moved {gap[s:e].std():.1f}mm (typical speech {base['win_std_p50']:.1f})"))

    # 3. open during silence
    for s, e in runs(~speech, int(0.3 * FPS)):
        inner = gap[s + 6:e] if e - s > 9 else gap[s:e]    # let it close (~0.2s)
        if len(inner) and np.median(inner) > base["silence_gap_p90"]:
            issues.append(("mouth open during silence", s, e,
                           f"median gap {np.median(inner):.1f}mm (normal silence p90 {base['silence_gap_p90']:.1f})"))

    # 4. lip closures (p/b/m need the lips to meet)
    closures = int(np.sum((gap[1:-1] < base["closed_mm"]) & (gap[1:-1] <= gap[:-2]) &
                          (gap[1:-1] < gap[2:]) & speech[1:-1]))
    # 5. 4-8 Hz jaw chatter — vibrato should not show up here
    gs = gap[speech] - gap[speech].mean() if speech.sum() > 30 else gap - gap.mean()
    spec = np.abs(np.fft.rfft(gs * np.hanning(len(gs)))) ** 2
    fr = np.fft.rfftfreq(len(gs), 1 / FPS)
    chatter = float(spec[(fr >= 4) & (fr <= 8)].sum() / (spec[fr > 0.3].sum() + 1e-9))

    if abs(lag_ms) >= 100:
        issues.append(("audio/mouth offset", 0, n, f"mouth is {'late' if lag_ms > 0 else 'early'} by {abs(lag_ms)} ms"))

    return {
        "seconds": round(n / FPS, 2),
        "lag_ms": lag_ms, "corr": corr,
        "frozen_speech_pct": frozen_pct,
        "lip_closures_per_s": round(closures / max(1e-9, sp / FPS), 2),
        "chatter_4_8hz": round(chatter, 3),
        "held_vowels": held,
        "issues": [{"what": w, "start_s": round(s / FPS, 2), "end_s": round(e / FPS, 2), "detail": d}
                   for w, s, e, d in issues],
    }, (db, speech, voiced, env)


def plot_clip(path, name, gap, feats, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    db, speech, voiced, env = feats
    t = np.arange(len(gap)) / FPS
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)
    ax[0].plot(t, env, color="#555", lw=1)
    ax[0].fill_between(t, 0, env.max(), where=voiced, color="#9cc3e6", alpha=0.35, label="voiced")
    ax[0].set_ylabel("loudness")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(t, gap, color="#c0504d", lw=1.4)
    ax[1].set_ylabel("lip gap (mm)")
    ax[1].set_xlabel("seconds")
    for k in res["issues"]:
        if k["end_s"] - k["start_s"] < res["seconds"] - 0.1:
            ax[1].axvspan(k["start_s"], k["end_s"], color="#f2c14e", alpha=0.35)
    ax[0].set_title(f"{name}   lag {res['lag_ms']} ms   corr {res['corr']}   "
                    f"issues {len(res['issues'])}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---- run --------------------------------------------------------------------------
from evaluate_shortcomings import load_emage, save_emage_npz  # noqa: E402

print("loading EMAGE…")
gen, sr = load_emage(Path(args.repo), Path(args.weights))

clips = {}
for cid, _, _ in SENTENCES:
    p = out / "wav" / f"{cid}.wav"
    if not p.exists():
        raise SystemExit(f"{p} missing — run the tts stage first (base env)")
    clips[cid] = librosa.load(str(p), sr=sr)[0]
src = clips.pop("vowel_src")
clips["held_vowel"] = held_vowel(src, sr)
clips["held_vibrato"] = add_vibrato(clips["held_vowel"], sr)
clips["drawl_vibrato"] = add_vibrato(clips["slow_drawl"], sr)
if args.wav_dir:
    for p in sorted(Path(args.wav_dir).glob("*.wav"))[:40]:
        clips[f"ext_{p.stem}"] = librosa.load(str(p), sr=sr)[0]
for k in ("held_vowel", "held_vibrato", "drawl_vibrato"):
    sf.write(out / "wav" / f"{k}.wav", clips[k], sr)

gaps = {}
for cid, a in clips.items():
    poses, expr, _ = gen(a)
    gap, width = mouth_aperture(poses, expr)
    gaps[cid] = gap
    save_emage_npz(out / f"{cid}.npz", poses, expr, np.zeros((len(poses), 3)))
    jaw = np.degrees(np.linalg.norm(poses[:, 66:69], axis=1))
    print(f"  {cid:<14} {len(a) / sr:5.1f}s  lip gap {gap.min():4.1f}-{gap.max():4.1f}mm  "
          f"gap~jaw corr {np.corrcoef(gap, jaw)[0, 1]:.2f}")

# The landmark indices are only right if the lip gap follows the jaw.
c = np.corrcoef(np.concatenate([gaps[b] for b in BASELINE]),
                np.concatenate([np.degrees(np.linalg.norm(np.load(out / f"{b}.npz")["poses"][:, 66:69], axis=1))
                                for b in BASELINE]))[0, 1]
print(f"\n  landmark sanity: lip gap vs jaw angle correlation {c:.2f} "
      f"({'ok' if c > 0.5 else 'LOW — landmark indices may be wrong, treat results with caution'})")

# Baseline from normal speech.
bw, bs, bsil = [], [], []
for b in BASELINE:
    db, speech, _, _ = audio_features(clips[b], sr, len(gaps[b]))
    g = gaps[b]
    for i in range(0, len(g) - 15 + 1, 5):
        if speech[i:i + 15].mean() > 0.8:
            bw.append(g[i:i + 15].std())
    bs.extend(g[speech])
    for s, e in runs(~speech, int(0.3 * FPS)):
        bsil.extend(g[s + 6:e] if e - s > 9 else g[s:e])
bsil = bsil or [min(bs)]
base = {"win_std_p10": float(np.percentile(bw, 10)), "win_std_p50": float(np.percentile(bw, 50)),
        "frozen_std": 0.4 * float(np.percentile(bw, 50)),
        "speech_gap_p25": float(np.percentile(bs, 25)), "speech_gap_mean": float(np.mean(bs)),
        "closed_mm": float(np.percentile(bs, 5) + 0.15 * (np.percentile(bs, 50) - np.percentile(bs, 5))),
        "silence_gap_p90": float(max(np.percentile(bsil, 90), np.percentile(bs, 10)))}
print("  baseline (normal speech): " + ", ".join(f"{k} {v:.2f}" for k, v in base.items()))

report = {"baseline": {k: round(v, 2) for k, v in base.items()}, "landmark_sanity_corr": round(float(c), 2),
          "clips": {}}
(out / "plots").mkdir(exist_ok=True)
for cid, a in clips.items():
    res, feats = check_clip(cid, a, sr, gaps[cid], base)
    report["clips"][cid] = res
    plot_clip(out / "plots" / f"{cid}.png", cid, gaps[cid], feats, res)
(out / "lipsync_report.json").write_text(json.dumps(report, indent=2))

print("\n  clip            secs  lag ms  corr  frozen%  closures/s  chatter  issues")
for cid, r in report["clips"].items():
    print(f"  {cid:<14} {r['seconds']:5.1f}  {r['lag_ms']:6d}  {r['corr']:4.2f}  {r['frozen_speech_pct']:6.1f}"
          f"  {r['lip_closures_per_s']:9.2f}  {r['chatter_4_8hz']:7.3f}  {len(r['issues'])}")
print("\n  issues:")
worst = sorted(report["clips"].items(), key=lambda kv: -len(kv[1]["issues"]))
for cid, r in worst:
    for k in r["issues"]:
        print(f"    {cid:<14} {k['start_s']:5.2f}-{k['end_s']:5.2f}s  {k['what']}: {k['detail']}")
print(f"\n  wrote {out}/lipsync_report.json, plots/*.png, <clip>.npz, wav/*.wav")
top = [cid for cid, r in worst[:2] if r["issues"]]
if top:
    print("  render the worst clips (face close-up):")
    for cid in top:
        print(f"    python face_export.py {out}/{cid}.npz --audio {out}/wav/{cid}.wav --label {cid}")
