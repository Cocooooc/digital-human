"""
Mouth post-processing — fixes EMAGE's lip-sync problems from the audio,
without retraining anything.

What the first lip-sync check (lipsync_check.py, 2026-11-01) found, and what
each fix here does about it:

  problem (measured)                        fix
  ----------------------------------------  -----------------------------------
  mouth stays open in pauses                silence -> jaw closes, face relaxes
    (up to 14.7 mm, closed is ~3 mm)          to neutral; opens again ~0.1 s
                                              BEFORE the next word (people
                                              open their mouth before sound)
  drawn-out vowel: mouth opens ~0.2 s late,  held vowel -> jaw held open at a
    flaps while the sound is steady,          steady value for the whole hold,
    closes before the sound ends              reached at the vowel's onset
  mouth barely moves in 20-31% of speech    "frozen" half-seconds -> jaw driven
                                              by the loudness of each syllable
  vibrato                                   was not a problem (no jaw chatter);
                                              held/vibrato spans are low-passed
                                              anyway as a guard

Everything is decided from the audio: loudness for speech/silence and
syllables, pitch (pYIN) for held vowels and vibrato — the "音域分析" part.
No model, no training data, so it applies to any voice and any language,
which is the point of an engineering fix.

The jaw is SMPL-X joint 22 (poses[:, 66:69]); opening is a rotation about one
axis, which is estimated from the clip itself (the dominant direction of the
jaw rotation during speech) rather than assumed.

Python 3.9 / numpy 1.23 compatible. Needs librosa (pantomatrix env has it).
"""
import numpy as np

FPS = 30
JAW = slice(66, 69)


# ---------------------------------------------------------------------------
# audio analysis at 30 fps
# ---------------------------------------------------------------------------
def _fill_gaps(mask, max_gap):
    m = mask.copy()
    idx = np.where(mask)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap:
            m[a:b] = True
    return m


def _runs(mask, min_len=1):
    out, s = [], None
    for i, x in enumerate(np.append(mask, False)):
        if x and s is None:
            s = i
        elif not x and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    return out


def analyse_audio(audio, sr, n):
    """Per-frame loudness (dB), speech mask, voiced mask, f0 (Hz, nan when
    unvoiced), and the spans of held vowels and of vibrato."""
    import librosa
    hop = sr / FPS
    idx = (np.arange(n) * hop).astype(int)
    win = int(hop * 2)
    rms = np.array([np.sqrt(np.mean(audio[i:i + win] ** 2) + 1e-12) if i < len(audio) else 1e-6
                    for i in idx])
    db = 20 * np.log10(rms + 1e-9)
    speech = _fill_gaps(db > db.max() - 35, int(0.2 * FPS))
    f0, voiced, _ = librosa.pyin(audio, fmin=65, fmax=500, sr=sr,
                                 frame_length=2048, hop_length=int(hop))
    m = min(n, len(voiced))
    v = np.zeros(n, bool)
    v[:m] = voiced[:m]
    f = np.full(n, np.nan)
    f[:m] = f0[:m]
    v &= speech

    # Held vowels: frames where pitch (smoothed over ~0.23 s, which also
    # averages out a 4-8 Hz vibrato) and loudness are locally steady. Looked
    # for inside voiced stretches rather than whole stretches, because a held
    # vowel usually follows ordinary speech with no silence in between.
    semis = np.full(n, np.nan)
    ok = ~np.isnan(f) & v
    if ok.sum() > 8:
        semis[ok] = 12 * np.log2(f[ok] / np.median(f[ok]))
    filled = semis.copy()
    if ok.any():
        filled[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], semis[ok])
    sm = _smooth(filled, 7)
    loc_pitch = np.array([sm[max(0, i - 5):i + 6].std() for i in range(n)])
    loc_level = np.array([db[max(0, i - 5):i + 6].std() for i in range(n)])
    steady = ok & (loc_pitch < 0.5) & (loc_level < 3.0)
    held, vibrato = [], []
    for s, e in _runs(steady, int(0.4 * FPS)):
        held.append((s, e))
        raw = semis[s:e]
        raw = raw[~np.isnan(raw)]
        if len(raw) < 12:
            continue
        spec = np.abs(np.fft.rfft((raw - raw.mean()) * np.hanning(len(raw)))) ** 2
        fr = np.fft.rfftfreq(len(raw), 1 / FPS)
        band = spec[(fr >= 4) & (fr <= 8)].sum() / (spec[fr > 0.5].sum() + 1e-9)
        if band > 0.5 and raw.std() > 0.1:
            vibrato.append((s, e))
    return {"db": db, "speech": speech, "voiced": v, "f0": f, "held": held, "vibrato": vibrato}


# ---------------------------------------------------------------------------
# jaw angle <-> rotvec
# ---------------------------------------------------------------------------
def _jaw_axis(jaw, speech):
    """Unit axis along which this clip's jaw opens."""
    sel = jaw[speech] if speech.sum() > 10 else jaw
    big = sel[np.linalg.norm(sel, axis=1) > np.percentile(np.linalg.norm(sel, axis=1), 50)]
    if len(big) == 0:
        return np.array([1.0, 0.0, 0.0])
    a = big.mean(0)
    nrm = np.linalg.norm(a)
    return a / nrm if nrm > 1e-6 else np.array([1.0, 0.0, 0.0])


def _smooth(x, k):
    if k <= 1:
        return x.copy()
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _ramp_mask(mask, before, after, ramp):
    """0/1 mask widened by `before`/`after` frames, then eased over `ramp`."""
    n = len(mask)
    w = np.zeros(n)
    for s, e in _runs(mask):
        w[max(0, s - before):min(n, e + after)] = 1.0
    return np.clip(_smooth(w, ramp), 0, 1)


# ---------------------------------------------------------------------------
# the fix
# ---------------------------------------------------------------------------
def fix_mouth(poses, expr, audio, sr, frozen_rel=0.4, verbose=True):
    """Returns (poses, expr, info). Only the jaw (joint 22) and the expression
    coefficients change; the body is untouched."""
    poses = np.asarray(poses, np.float64).copy()
    expr = np.asarray(expr, np.float64).copy()
    n = len(poses)
    A = analyse_audio(audio, sr, n)
    speech, db = A["speech"], A["db"]

    jaw = poses[:, JAW]
    axis = _jaw_axis(jaw, speech)
    theta = jaw @ axis                                   # opening angle, radians
    resid = jaw - theta[:, None] * axis[None, :]         # sideways/twist part, kept

    sp = theta[speech] if speech.sum() > 10 else theta
    # "Closed" from speech, where the model does close for p/b/m — not from
    # the whole clip: if the model keeps the mouth open in pauses (it does),
    # a whole-clip percentile would call that open jaw "closed".
    closed = float(np.percentile(sp, 2))
    p25, p50, p75, p90 = (float(np.percentile(sp, q)) for q in (25, 50, 75, 90))
    span = max(p90 - closed, 1e-4)
    new = theta.copy()
    info = {"frozen_frames": 0, "held_spans": [], "vibrato_spans": [], "silence_frames": 0}

    # 1. Frozen speech: half-second windows in which the jaw moves less than
    #    `frozen_rel` x a typical speech window. Drive the jaw from loudness
    #    there: each syllable's energy peak opens it, the dip between closes it.
    win = 15
    stds = [theta[i:i + win].std() for i in range(0, max(1, n - win + 1), 5)
            if speech[i:i + win].mean() > 0.8]
    typical = float(np.median(stds)) if stds else 0.0
    frozen = np.zeros(n, bool)
    for i in range(0, max(1, n - win + 1), 5):
        if speech[i:i + win].mean() > 0.8 and theta[i:i + win].std() < frozen_rel * typical:
            frozen[i:i + win] = True
    in_held = np.zeros(n, bool)
    for s, e in A["held"]:
        in_held[s:e] = True
    frozen &= ~in_held                     # a still mouth is right on a held vowel
    if frozen.any():
        lo, hi = np.percentile(db[speech], 10), np.percentile(db[speech], 95)
        env = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.8
        env = np.roll(env, -2)                             # mouth leads sound ~65 ms
        driven = closed + 0.15 * span + env * 0.85 * span
        wf = np.clip(_smooth(frozen.astype(float), 5), 0, 1)
        new = (1 - wf) * new + wf * (0.2 * theta + 0.8 * driven)
        info["frozen_frames"] = int(frozen.sum())

    # 2. Held vowels: one steady opening for the whole hold, reached at the
    #    onset (fixes the late opening), with a short ease in and out.
    for s, e in A["held"]:
        target = max(p75, float(np.percentile(theta[s:e], 75)))
        w = np.zeros(n)
        w[s:e] = 1.0
        w = np.clip(_smooth(w, 5), 0, 1)
        # start easing in 3 frames before the sound
        pre = max(0, s - 3)
        w[pre:s] = np.maximum(w[pre:s], np.linspace(0.3, 0.9, s - pre)) if s > pre else w[pre:s]
        new = (1 - w) * new + w * target
        ee = expr[s:e].mean(0)
        expr = (1 - w[:, None]) * expr + w[:, None] * ee[None, :]
        info["held_spans"].append([round(s / FPS, 2), round(e / FPS, 2)])

    # 3. Vibrato guard: whatever the model did there, keep the jaw slow.
    for s, e in A["vibrato"]:
        new[s:e] = _smooth(new[s:e], 9)
        info["vibrato_spans"].append([round(s / FPS, 2), round(e / FPS, 2)])

    # 4. Silence: close the mouth and relax the face. Opening starts 3 frames
    #    (100 ms) before speech, closing 2 frames after it ends, eased over 5.
    w_speech = _ramp_mask(speech, before=3, after=2, ramp=5)
    # Jaw: whatever the model did in a pause is replaced, not blended — it
    # tends to hold the mouth wide open there, and blending would carry that
    # into the run-up. Closed in the pause, opening over the 3 frames before
    # the first sound, closing over the 3 frames after the last.
    sp_runs = _runs(speech)
    quiet = ~speech
    for s0, e0 in sp_runs:
        quiet[s0:e0] = False
    jaw_new = np.where(quiet, closed, new)
    for s0, e0 in sp_runs:
        for k in range(1, 4):
            if s0 - k >= 0 and quiet[s0 - k]:
                jaw_new[s0 - k] = closed + (new[s0] - closed) * (1 - k / 4)
            if e0 - 1 + k < n and quiet[e0 - 1 + k]:
                jaw_new[e0 - 1 + k] = closed + (new[e0 - 1] - closed) * (1 - k / 4)
    new = jaw_new
    neutral = np.zeros(expr.shape[1])
    expr = w_speech[:, None] * expr + (1 - w_speech[:, None]) * (0.25 * expr + 0.75 * neutral)
    info["silence_frames"] = int((w_speech < 0.5).sum())

    # light smoothing so none of the blends leave a step
    new = _smooth(new, 3)
    poses[:, JAW] = new[:, None] * axis[None, :] + resid * np.clip(w_speech, 0.2, 1)[:, None]

    info.update({"jaw_axis": [round(float(x), 3) for x in axis],
                 "jaw_deg": {"closed": round(np.degrees(closed), 1), "p50": round(np.degrees(p50), 1),
                             "p90": round(np.degrees(p90), 1)}})
    if verbose:
        print(f"    mouth_post: frozen {info['frozen_frames']} frames, held {len(A['held'])}, "
              f"vibrato {len(A['vibrato'])}, silent {info['silence_frames']} frames")
    return poses.astype(np.float32), expr.astype(np.float32), info


if __name__ == "__main__":
    # Self-test on synthetic audio + a deliberately bad jaw track.
    sr = 16000
    t = np.arange(int(6 * sr)) / sr
    f0 = 150 + 25 * np.sin(2 * np.pi * 0.6 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(np.sin(k * ph) / k for k in range(1, 8))
    env = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 0.7
    env[int(2.0 * sr):int(2.6 * sr)] = 0                        # a pause
    held_env = np.zeros_like(t)
    held_env[int(3.5 * sr):int(5.5 * sr)] = 1.0                 # a held vowel 3.5-5.5 s
    env = np.where(held_env > 0, 1.0, env)
    env[int(3.3 * sr):int(3.5 * sr)] = 0
    f_hold = np.where(held_env > 0, 160.0, f0)
    ph = 2 * np.pi * np.cumsum(f_hold) / sr
    audio = (0.2 * sum(np.sin(k * ph) / k for k in range(1, 8)) * env).astype(np.float32)

    n = int(6 * FPS)
    tt = np.arange(n) / FPS
    jaw_theta = 0.12 + 0.08 * np.sin(2 * np.pi * 3 * tt)       # flapping everywhere
    jaw_theta[30:60] = 0.15                                    # frozen 1.0-2.0 s
    jaw_theta[60:78] = 0.2                                     # open in the pause
    poses = np.zeros((n, 165), np.float32)
    poses[:, 66] = jaw_theta
    expr = np.random.default_rng(0).normal(0, 0.3, (n, 100)).astype(np.float32)

    p2, e2, info = fix_mouth(poses, expr, audio, sr)
    th2 = p2[:, 66]
    print("info", {k: v for k, v in info.items()})
    pause = slice(int(2.15 * FPS), int(2.5 * FPS))
    hold = slice(int(3.7 * FPS), int(5.3 * FPS))
    frozen = slice(33, 57)
    print(f"pause jaw  before {np.degrees(jaw_theta[pause]).mean():.1f}  after {np.degrees(th2[pause]).mean():.1f} deg")
    print(f"hold  std  before {np.degrees(jaw_theta[hold]).std():.2f}  after {np.degrees(th2[hold]).std():.2f} deg")
    print(f"frozen std before {np.degrees(jaw_theta[frozen]).std():.2f}  after {np.degrees(th2[frozen]).std():.2f} deg")
    assert np.degrees(th2[pause]).mean() < np.degrees(jaw_theta[pause]).mean() - 3
    assert np.degrees(th2[hold]).std() < 0.5 * np.degrees(jaw_theta[hold]).std()
    assert np.degrees(th2[frozen]).std() > 2 * np.degrees(jaw_theta[frozen]).std()
    assert np.abs(np.diff(th2)).max() < np.radians(6), "a blend left a step"
    print("self-test OK")
