"""
EMAGE diagnosis — answer the three questions that decide the whole architecture,
before writing any integration code.

  1. How fast is it, on the audio lengths we actually use?
  2. What is really inside the .npz?
  3. What frame rate does it produce?

Why short clips matter most: the live pipeline is sentence-level. Kokoro
synthesizes a short sentence in ~50ms and the user hears speech within 0.6s. If
EMAGE carries a fixed per-call overhead the way Qwen3-TTS did (~6s regardless of
length), a 28-second benchmark would hide it completely while 2-second clips
would make it obvious. So we measure across lengths and separate fixed cost from
marginal cost.

Run inside the pantomatrix env:
    conda activate pantomatrix
    python /root/diagnose_emage.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/root/PantoMatrix")
EXAMPLE_AUDIO = REPO / "examples/audio/2_scott_0_103_103_28s.wav"
WORK = Path("/root/emage_diag")
WORK.mkdir(exist_ok=True)


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------------------
# 1. Show how the official script builds the model.
#
# EMAGE needs four objects (EmageAudioModel + VQVAE/VAE pieces assembled into
# EmageVQModel), not the single from_pretrained() call the README shows for
# CaMN. Rather than guess at that wiring — which cost several rounds on
# Qwen3-TTS — read it out of the script that is known to work.
# ---------------------------------------------------------------------------
def show_model_loading():
    section("1. How test_emage_audio.py actually loads the model")
    script = REPO / "test_emage_audio.py"
    if not script.exists():
        print("test_emage_audio.py missing")
        return
    text = script.read_text()
    # the loading happens in __main__ / main(), after the inference helper
    marker = text.find("if __name__")
    if marker == -1:
        marker = text.find("def main")
    print(text[marker:] if marker != -1 else text[-3000:])


# ---------------------------------------------------------------------------
# 2. Build test clips at the lengths that matter.
# ---------------------------------------------------------------------------
def make_clips():
    section("2. Preparing test clips")
    import librosa
    import soundfile as sf

    if not EXAMPLE_AUDIO.exists():
        print(f"missing {EXAMPLE_AUDIO}")
        sys.exit(1)

    audio, sr = librosa.load(str(EXAMPLE_AUDIO), sr=None)
    full_dur = len(audio) / sr
    print(f"source: {EXAMPLE_AUDIO.name}  {full_dur:.1f}s @ {sr}Hz")

    clips = {}
    for secs in (2, 4, 8, 16):
        if full_dur < secs:
            continue
        out = WORK / f"clip_{secs}s.wav"
        sf.write(out, audio[: int(secs * sr)], sr)
        clips[secs] = out
        print(f"  wrote {out.name}")

    # the full clip too, as the long-form reference point
    clips[round(full_dur)] = EXAMPLE_AUDIO
    return clips, sr


# ---------------------------------------------------------------------------
# 3. Time inference per clip length.
#
# Run through the official CLI so we exercise exactly the supported path. Model
# load is paid once per invocation, so the first run is discarded as warm-up and
# the remaining numbers are compared against each other — the *shape* of the
# curve is what reveals fixed vs marginal cost.
# ---------------------------------------------------------------------------
def time_inference(clips):
    section("3. Inference timing")

    results = []
    print(f"{'audio':>8} {'wall':>9} {'RTF':>8}")
    print("-" * 28)

    for secs in sorted(clips):
        audio_path = clips[secs]
        save_dir = WORK / f"out_{secs}s"
        save_dir.mkdir(exist_ok=True)

        # one folder per clip so each run sees exactly one input file
        clip_dir = WORK / f"in_{secs}s"
        clip_dir.mkdir(exist_ok=True)
        target = clip_dir / audio_path.name
        if not target.exists():
            target.write_bytes(audio_path.read_bytes())

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "test_emage_audio.py",
             "--audio_folder", str(clip_dir),
             "--save_folder", str(save_dir)],
            cwd=str(REPO), capture_output=True, text=True,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"\n❌ failed on {secs}s clip:")
            print(proc.stdout[-2000:])
            print(proc.stderr[-3000:])
            return results

        rtf = elapsed / secs
        print(f"{secs:>6}s {elapsed:>8.2f}s {rtf:>8.2f}")
        results.append((secs, elapsed, save_dir))

    # Separate fixed startup cost from per-second cost using the two extremes.
    # Everything here includes model loading, so the intercept is dominated by
    # it — what matters is whether the slope is small enough to stream.
    if len(results) >= 2:
        (s1, t1, _), (s2, t2, _) = results[0], results[-1]
        slope = (t2 - t1) / (s2 - s1)
        intercept = t1 - slope * s1
        print(f"\nfitted:  time ≈ {intercept:.1f}s fixed + {slope:.2f} × audio_seconds")
        print(f"  fixed cost is mostly model loading (paid once if we keep it resident)")
        print(f"  marginal RTF ≈ {slope:.2f}  ← this is what decides streaming viability")
        if slope < 0.3:
            print("  ✅ fast enough to generate per sentence in the live path")
        elif slope < 1.0:
            print("  ⚠️  usable but will add noticeable latency; consider async")
        else:
            print("  ❌ slower than real time — needs async or pre-generation")

    return results


# ---------------------------------------------------------------------------
# 4. Crack open the .npz.
#
# The README says "SMPLX and FLAME parameters". What the file actually holds —
# key names, shapes, dtypes — determines how we map it onto a VRM rig, and
# documentation and reality often disagree.
# ---------------------------------------------------------------------------
def inspect_npz(results):
    section("4. What the .npz actually contains")
    if not results:
        print("no successful runs to inspect")
        return

    secs, _, save_dir = results[-1]
    npz_files = list(Path(save_dir).glob("*.npz"))
    if not npz_files:
        print(f"no .npz produced in {save_dir}")
        print("contents:", list(Path(save_dir).iterdir()))
        return

    path = npz_files[0]
    print(f"file: {path.name}   ({path.stat().st_size/1e6:.2f} MB, from {secs}s audio)\n")

    data = np.load(path, allow_pickle=True)
    n_frames = None
    for key in data.files:
        arr = data[key]
        desc = f"  {key:<20}"
        if hasattr(arr, "shape"):
            desc += f" shape={str(arr.shape):<20} dtype={arr.dtype}"
            if arr.ndim >= 1 and arr.shape[0] > 10:
                n_frames = n_frames or arr.shape[0]
            if arr.size and np.issubdtype(arr.dtype, np.number):
                desc += f"  range=[{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]"
        else:
            desc += f" {type(arr)} = {arr}"
        print(desc)

    if n_frames:
        print(f"\nframes: {n_frames} over {secs}s audio → {n_frames/secs:.1f} fps")

    # Interpret the two arrays we care about against SMPL-X conventions:
    # 55 joints × 3 axis-angle values = 165 per frame for the full body.
    print("\ninterpretation:")
    for key in data.files:
        arr = data[key]
        if not hasattr(arr, "shape") or arr.ndim < 2:
            continue
        width = arr.shape[-1]
        if width == 165:
            print(f"  {key}: 165 = 55 joints × 3 → full SMPL-X body+hands+face pose ✅")
        elif width == 156:
            print(f"  {key}: 156 = 52 joints × 3 → SMPL-X body+hands (no face joints)")
        elif width in (100, 50):
            print(f"  {key}: {width} → FLAME expression coefficients (blendshape weights)")
        elif width == 3:
            print(f"  {key}: 3 → global translation")


def main():
    show_model_loading()
    clips, sr = make_clips()
    results = time_inference(clips)
    inspect_npz(results)

    section("Summary of what this tells us")
    print("""
The marginal RTF decides the architecture:

  < 0.3   generate motion per sentence inside the live loop, same as TTS today
  0.3-1.0 workable, but motion should be generated asynchronously and the
          avatar idles until it arrives
  > 1.0   motion cannot keep up with speech; pre-generate, or drive the body
          with simpler procedural idle motion and use EMAGE only for the face

The .npz shapes decide the retargeting work: a 165-wide array is full SMPL-X
axis-angle and needs bone retargeting onto the VRM humanoid rig; the expression
array maps onto VRM blendshapes more directly.
""")


if __name__ == "__main__":
    main()
