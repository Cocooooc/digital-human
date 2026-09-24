"""
EMAGE diagnosis, v2 — bypasses the visualization dependency.

test_emage_audio.py imports fast_render at module level, which pulls in
pyrender → OpenGL. On a headless server that means EGL/OSMesa system packages
and a good chance of a dependency spiral, all to produce MP4 previews we will
never use: the goal is a motion-generation service, not a renderer.

So we stub pyrender out before importing, then reuse the repo's own model
loading code rather than reimplementing it — guessing at model wiring is what
cost several rounds on Qwen3-TTS.

What this measures, and why:
  Model load and inference are timed separately. In production the model stays
  resident, so load cost is paid once at startup and is irrelevant to per-turn
  latency; only inference time competes with the ~1s conversation loop.

  Clips of several lengths, because the live pipeline is sentence-level. A
  fixed per-call overhead — Qwen3-TTS had ~6s of it — is invisible in a 28s
  benchmark and fatal for 2s sentences.

Run:
    conda activate pantomatrix
    python /root/diagnose_emage2.py
"""
import inspect
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# --- stub the rendering stack before anything imports it -------------------
for mod in ["pyrender", "trimesh", "OpenGL", "OpenGL.GL"]:
    sys.modules.setdefault(mod, MagicMock())

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path("/root/PantoMatrix")
EXAMPLE_AUDIO = REPO / "examples/audio/2_scott_0_103_103_28s.wav"
WORK = Path("/root/emage_diag")
WORK.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))


def section(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


# ---------------------------------------------------------------------------
section("1. Importing the official script with rendering stubbed out")
# ---------------------------------------------------------------------------
try:
    import test_emage_audio as official
    print("✅ imported test_emage_audio")
except Exception as e:
    print(f"❌ import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n--- its main(), i.e. how the models are really constructed ---")
try:
    print(inspect.getsource(official.main))
except Exception as e:
    print(f"(couldn't read main: {e})")
    print("module-level names:", [n for n in dir(official) if not n.startswith("_")])


# ---------------------------------------------------------------------------
section("2. Loading models (timed separately — paid once in production)")
# ---------------------------------------------------------------------------
device = "cuda"
t0 = time.time()

try:
    from models.emage_audio import (
        EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel
    )

    # Mirrors the official main(): four checkpoints assembled into one wrapper.
    # Repo ids are read back from main()'s source above if these differ.
    model = EmageAudioModel.from_pretrained("H-Liu1997/emage_audio").to(device).eval()

    face_motion_vq = EmageVQVAEConv.from_pretrained(
        "H-Liu1997/emage_audio", subfolder="emage_vq/face").to(device).eval()
    upper_motion_vq = EmageVQVAEConv.from_pretrained(
        "H-Liu1997/emage_audio", subfolder="emage_vq/upper").to(device).eval()
    hands_motion_vq = EmageVQVAEConv.from_pretrained(
        "H-Liu1997/emage_audio", subfolder="emage_vq/hands").to(device).eval()
    lower_motion_vq = EmageVQVAEConv.from_pretrained(
        "H-Liu1997/emage_audio", subfolder="emage_vq/lower").to(device).eval()
    global_motion_ae = EmageVAEConv.from_pretrained(
        "H-Liu1997/emage_audio", subfolder="emage_vq/global").to(device).eval()

    motion_vq = EmageVQModel(
        face_model=face_motion_vq, upper_model=upper_motion_vq,
        hands_model=hands_motion_vq, lower_model=lower_motion_vq,
        global_model=global_motion_ae,
    ).to(device).eval()

    load_time = time.time() - t0
    print(f"✅ models loaded in {load_time:.1f}s")
    print(f"   VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

except Exception as e:
    print(f"❌ model construction failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nThe main() source printed above shows the correct wiring —")
    print("send it back and this script can be corrected in one step.")
    sys.exit(1)


# ---------------------------------------------------------------------------
section("3. Preparing clips at the lengths we actually use")
# ---------------------------------------------------------------------------
import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

audio_full, sr_native = librosa.load(str(EXAMPLE_AUDIO), sr=None)
print(f"source: {EXAMPLE_AUDIO.name}  {len(audio_full)/sr_native:.1f}s @ {sr_native}Hz")

SR = getattr(getattr(model, "cfg", None), "audio_sr", 16000)
print(f"model expects sr={SR}")

clip_lengths = [2, 3, 5, 10, 20]
clips = {}
for secs in clip_lengths:
    if len(audio_full) / sr_native < secs:
        continue
    a = librosa.load(str(EXAMPLE_AUDIO), sr=SR, duration=secs)[0]
    clips[secs] = a
    print(f"  {secs}s → {len(a)} samples")


# ---------------------------------------------------------------------------
section("4. Inference timing (model already resident)")
# ---------------------------------------------------------------------------
def run_once(audio_np):
    audio = torch.from_numpy(audio_np).float().to(device).unsqueeze(0)
    speaker_id = torch.zeros(1, 1).long().to(device)
    trans = torch.zeros(1, 1, 3).to(device)

    with torch.no_grad():
        latent_dict = model.inference(
            audio, speaker_id, motion_vq, masked_motion=None, mask=None)

        import torch.nn.functional as F
        cfg = model.cfg
        face_latent = latent_dict["rec_face"] if cfg.lf > 0 and cfg.cf == 0 else None
        upper_latent = latent_dict["rec_upper"] if cfg.lu > 0 and cfg.cu == 0 else None
        hands_latent = latent_dict["rec_hands"] if cfg.lh > 0 and cfg.ch == 0 else None
        lower_latent = latent_dict["rec_lower"] if cfg.ll > 0 and cfg.cl == 0 else None

        pick = lambda k: torch.max(F.log_softmax(latent_dict[k], dim=2), dim=2)[1]
        face_index = pick("cls_face") if cfg.cf > 0 else None
        upper_index = pick("cls_upper") if cfg.cu > 0 else None
        hands_index = pick("cls_hands") if cfg.ch > 0 else None
        lower_index = pick("cls_lower") if cfg.cl > 0 else None

        all_pred = motion_vq.decode(
            face_latent=face_latent, upper_latent=upper_latent,
            lower_latent=lower_latent, hands_latent=hands_latent,
            face_index=face_index, upper_index=upper_index,
            lower_index=lower_index, hands_index=hands_index,
            get_global_motion=True, ref_trans=trans[:, 0])
    return all_pred


# warm-up: first CUDA call compiles kernels and allocates, and would otherwise
# be charged entirely to the shortest clip
print("warming up...")
_ = run_once(clips[min(clips)])
torch.cuda.synchronize()

print(f"\n{'audio':>7} {'infer':>9} {'RTF':>8}   {'frames':>7} {'fps':>6}")
print("-" * 45)

measurements = []
last_pred = None
for secs in sorted(clips):
    torch.cuda.synchronize()
    t0 = time.time()
    pred = run_once(clips[secs])
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    n_frames = pred["motion_axis_angle"].shape[1]
    fps = n_frames / secs
    print(f"{secs:>5}s {elapsed:>8.3f}s {elapsed/secs:>8.3f}   {n_frames:>7} {fps:>6.1f}")
    measurements.append((secs, elapsed))
    last_pred = pred

# Fixed vs marginal cost: with the model resident, any large intercept is
# genuine per-call overhead and would be paid once per sentence.
if len(measurements) >= 2:
    xs = np.array([m[0] for m in measurements], dtype=float)
    ys = np.array([m[1] for m in measurements], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1)
    print(f"\nfit: infer_time ≈ {intercept:.3f}s + {slope:.3f} × audio_seconds")
    print(f"  per-call overhead : {intercept:.3f}s")
    print(f"  marginal RTF      : {slope:.3f}")

    typical = intercept + slope * 3  # a ~3s spoken sentence
    print(f"\n  a 3s sentence would take ~{typical:.2f}s of motion generation")
    if typical < 0.5:
        print("  ✅ fits inside the live loop alongside TTS")
    elif typical < 2.0:
        print("  ⚠️  noticeable; generate asynchronously and let the avatar idle first")
    else:
        print("  ❌ too slow for the live loop; face-only or pre-generation instead")


# ---------------------------------------------------------------------------
section("5. What the output actually contains")
# ---------------------------------------------------------------------------
for key, val in last_pred.items():
    if torch.is_tensor(val):
        arr = val.detach().cpu().numpy()
        line = f"  {key:<22} shape={str(arr.shape):<22} dtype={arr.dtype}"
        if arr.size:
            line += f"  range=[{np.nanmin(arr):.3f}, {np.nanmax(arr):.3f}]"
        print(line)
    else:
        print(f"  {key:<22} {type(val)}")

print("\ninterpretation against SMPL-X conventions:")
for key, val in last_pred.items():
    if not torch.is_tensor(val) or val.ndim < 2:
        continue
    w = val.shape[-1]
    if w == 165:
        print(f"  {key}: 165 = 55 joints × 3 → full SMPL-X axis-angle (body+hands+jaw) ✅")
    elif w == 156:
        print(f"  {key}: 156 = 52 joints × 3 → SMPL-X body+hands")
    elif w in (50, 100):
        print(f"  {key}: {w} → FLAME expression coefficients → convertible to blendshapes ✅")
    elif w == 3:
        print(f"  {key}: 3 → global translation")

section("Bottom line")
print(f"""
  model load      : {load_time:.1f}s   (once at startup, irrelevant per turn)
  VRAM            : {torch.cuda.memory_allocated()/1e9:.2f} GB
  motion output   : SMPL-X axis-angle → needs retargeting onto the VRM rig
  face output     : expression coefficients → maps to VRM blendshapes

  The marginal RTF above is what decides whether motion generation can live
  inside the 1-second conversation loop or has to run asynchronously.
""")
