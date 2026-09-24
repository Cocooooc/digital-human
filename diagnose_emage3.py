"""
EMAGE diagnosis, v3 — no dependency on the official test script at all.

v2 tried to import test_emage_audio and stub out its rendering imports, but
fast_render pulls pyrender → imageio → (likely moviepy, matplotlib...). Stubbing
them one at a time is whack-a-mole for code we never call: the goal is a motion
service, not a video renderer.

The reconnaissance already gave us the full inference body from that script, so
this reimplements just that path directly. The one unknown — the checkpoint
subfolder layout — is read off the weights already on disk rather than guessed.

Timing separates model load from inference: in production the model stays
resident, so only inference competes with the ~1s conversation loop. Clips span
several lengths because the pipeline is sentence-level, and a fixed per-call
overhead (Qwen3-TTS had ~6s of it) is invisible at 28s and fatal at 2s.

Run:
    conda activate pantomatrix
    python /root/diagnose_emage3.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/root/PantoMatrix")
WEIGHTS = Path("/root/emage_weights")
EXAMPLE_AUDIO = REPO / "examples/audio/2_scott_0_103_103_28s.wav"
sys.path.insert(0, str(REPO))

DEVICE = "cuda"


def section(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


# ---------------------------------------------------------------------------
section("1. What's actually in the downloaded weights")
# ---------------------------------------------------------------------------
if not WEIGHTS.exists():
    print(f"❌ {WEIGHTS} missing — run the snapshot_download step first")
    sys.exit(1)

subfolders = []
for p in sorted(WEIGHTS.rglob("*")):
    rel = p.relative_to(WEIGHTS)
    if p.is_dir():
        print(f"  [dir ] {rel}")
    elif p.suffix in (".bin", ".safetensors", ".pt", ".pth", ".json"):
        size = p.stat().st_size / 1e6
        print(f"  [file] {rel}  ({size:.1f} MB)")
        # a directory holding a config+weights pair is a loadable subfolder
        if p.name in ("config.json", "model.safetensors", "pytorch_model.bin"):
            parent = str(rel.parent)
            if parent != "." and parent not in subfolders:
                subfolders.append(parent)

print(f"\ndetected subfolders: {subfolders or '(none — flat layout)'}")


# ---------------------------------------------------------------------------
section("2. Loading models (timed separately; paid once at startup)")
# ---------------------------------------------------------------------------
from models.emage_audio import (  # noqa: E402
    EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel
)

t0 = time.time()
SRC = str(WEIGHTS)  # load from local disk, no network


def load(cls, subfolder=None):
    """Try the local snapshot first; fall back to the hub id if the local
    layout doesn't match what the class expects."""
    try:
        if subfolder:
            return cls.from_pretrained(SRC, subfolder=subfolder).to(DEVICE).eval()
        return cls.from_pretrained(SRC).to(DEVICE).eval()
    except Exception as local_err:
        print(f"  local load failed for {subfolder or '<root>'}: {local_err}")
        print(f"  retrying from hub…")
        if subfolder:
            return cls.from_pretrained(
                "H-Liu1997/emage_audio", subfolder=subfolder).to(DEVICE).eval()
        return cls.from_pretrained("H-Liu1997/emage_audio").to(DEVICE).eval()


try:
    model = load(EmageAudioModel)
    print("✅ EmageAudioModel")

    parts = {}
    for name in ("face", "upper", "hands", "lower"):
        parts[name] = load(EmageVQVAEConv, f"emage_vq/{name}")
        print(f"✅ VQVAE {name}")
    global_ae = load(EmageVAEConv, "emage_vq/global")
    print("✅ VAE global")

    motion_vq = EmageVQModel(
        face_model=parts["face"], upper_model=parts["upper"],
        hands_model=parts["hands"], lower_model=parts["lower"],
        global_model=global_ae,
    ).to(DEVICE).eval()

    load_time = time.time() - t0
    print(f"\n✅ all loaded in {load_time:.1f}s | VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB")

except Exception as e:
    print(f"\n❌ loading failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n--- for reference, the official script's main() ---")
    # read as text; importing it is what dragged in the rendering stack
    txt = (REPO / "test_emage_audio.py").read_text()
    i = txt.find("def main")
    print(txt[i:i + 4000] if i != -1 else txt[-4000:])
    sys.exit(1)


# ---------------------------------------------------------------------------
section("3. Clips at the lengths the live pipeline actually produces")
# ---------------------------------------------------------------------------
import librosa  # noqa: E402

SR = getattr(getattr(model, "cfg", None), "audio_sr", 16000)
print(f"model sample rate: {SR}")

full_dur = librosa.get_duration(path=str(EXAMPLE_AUDIO))
print(f"source: {EXAMPLE_AUDIO.name} ({full_dur:.1f}s)")

clips = {}
for secs in (2, 3, 5, 10, 20):
    if secs > full_dur:
        continue
    clips[secs] = librosa.load(str(EXAMPLE_AUDIO), sr=SR, duration=secs)[0]
    print(f"  {secs}s → {len(clips[secs])} samples")


# ---------------------------------------------------------------------------
section("4. Inference timing")
# ---------------------------------------------------------------------------
def run_once(audio_np):
    audio = torch.from_numpy(audio_np).float().to(DEVICE).unsqueeze(0)
    speaker_id = torch.zeros(1, 1).long().to(DEVICE)
    trans = torch.zeros(1, 1, 3).to(DEVICE)

    with torch.no_grad():
        latent = model.inference(audio, speaker_id, motion_vq,
                                 masked_motion=None, mask=None)
        cfg = model.cfg
        pick = lambda k: torch.max(F.log_softmax(latent[k], dim=2), dim=2)[1]

        return motion_vq.decode(
            face_latent=latent["rec_face"] if cfg.lf > 0 and cfg.cf == 0 else None,
            upper_latent=latent["rec_upper"] if cfg.lu > 0 and cfg.cu == 0 else None,
            hands_latent=latent["rec_hands"] if cfg.lh > 0 and cfg.ch == 0 else None,
            lower_latent=latent["rec_lower"] if cfg.ll > 0 and cfg.cl == 0 else None,
            face_index=pick("cls_face") if cfg.cf > 0 else None,
            upper_index=pick("cls_upper") if cfg.cu > 0 else None,
            hands_index=pick("cls_hands") if cfg.ch > 0 else None,
            lower_index=pick("cls_lower") if cfg.cl > 0 else None,
            get_global_motion=True, ref_trans=trans[:, 0])


# First CUDA call compiles kernels and allocates buffers; without a warm-up
# that one-off cost lands entirely on the shortest clip and distorts the fit.
print("warming up…")
try:
    _ = run_once(clips[min(clips)])
    torch.cuda.synchronize()
except Exception as e:
    print(f"❌ inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'audio':>7} {'infer':>10} {'RTF':>8} {'frames':>8} {'fps':>7}")
print("-" * 45)

measurements = []
last = None
for secs in sorted(clips):
    torch.cuda.synchronize()
    t0 = time.time()
    pred = run_once(clips[secs])
    torch.cuda.synchronize()
    dt = time.time() - t0

    n = pred["motion_axis_angle"].shape[1]
    print(f"{secs:>5}s {dt:>9.3f}s {dt/secs:>8.3f} {n:>8} {n/secs:>7.1f}")
    measurements.append((secs, dt))
    last = pred

if len(measurements) >= 2:
    xs = np.array([m[0] for m in measurements], float)
    ys = np.array([m[1] for m in measurements], float)
    slope, intercept = np.polyfit(xs, ys, 1)
    print(f"\nfit: infer ≈ {intercept:.3f}s + {slope:.3f} × audio_seconds")
    print(f"  per-call overhead : {intercept:.3f}s")
    print(f"  marginal RTF      : {slope:.3f}")

    t3 = intercept + slope * 3
    print(f"\n  a 3s sentence → ~{t3:.2f}s of motion generation")
    if t3 < 0.5:
        print("  ✅ fits inside the live loop next to TTS")
    elif t3 < 2.0:
        print("  ⚠️  noticeable — generate async, avatar idles until motion arrives")
    else:
        print("  ❌ too slow for the live loop — face-only or pre-generate")


# ---------------------------------------------------------------------------
section("5. Output structure")
# ---------------------------------------------------------------------------
for k, v in last.items():
    if torch.is_tensor(v):
        a = v.detach().cpu().numpy()
        line = f"  {k:<22} shape={str(a.shape):<24} dtype={a.dtype}"
        if a.size:
            line += f"  range=[{np.nanmin(a):.3f}, {np.nanmax(a):.3f}]"
        print(line)
    else:
        print(f"  {k:<22} {type(v)}")

print("\nagainst SMPL-X conventions:")
for k, v in last.items():
    if not torch.is_tensor(v) or v.ndim < 2:
        continue
    w = v.shape[-1]
    if w == 165:
        print(f"  {k}: 165 = 55 joints × 3 → full SMPL-X axis-angle ✅")
    elif w == 156:
        print(f"  {k}: 156 = 52 joints × 3 → SMPL-X body+hands")
    elif w in (50, 100):
        print(f"  {k}: {w} → FLAME expression coefficients → blendshapes ✅")
    elif w == 3:
        print(f"  {k}: 3 → global translation")

section("Bottom line")
print(f"""
  model load : {load_time:.1f}s  (startup only)
  VRAM       : {torch.cuda.memory_allocated()/1e9:.2f} GB
  body       : SMPL-X axis-angle → needs retargeting onto the VRM rig
  face       : expression coefficients → maps to VRM blendshapes

  The marginal RTF decides whether motion generation lives inside the
  1-second conversation loop or has to run asynchronously.
""")
