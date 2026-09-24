"""
EMAGE motion service — audio in, SMPL-X motion out.

Runs in the pantomatrix conda env (Python 3.9, torch 2.8), separate from the
conversation stack (Python 3.10, torch 2.6). The two environments genuinely
cannot be merged — different torch majors, and PantoMatrix pins numpy 1.23 —
so motion generation is its own process and the conversation server calls it
over localhost. At RTF 0.012 the round trip costs more than the inference, and
both are negligible against the ~1s conversation loop.

Endpoints:
  GET  /            health + what's loaded
  POST /motion      upload wav → SMPL-X motion + FLAME expression

Motion is returned as base64-encoded float32 rather than JSON numbers: a 3s
sentence is 90 frames × 165 floats, which as JSON text would be ~400KB of
decimal strings versus 59KB of raw bytes. The browser decodes it straight into
a Float32Array.

Run (the port must be one the cloud console actually exposes — the range
changes with every new instance, and a port outside it listens fine locally
while being unreachable from outside):
    conda activate pantomatrix
    python -m uvicorn emage_server:app --host 0.0.0.0 --port 15313
"""

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

REPO = Path("/root/PantoMatrix")
WEIGHTS = Path("/root/emage_weights")
sys.path.insert(0, str(REPO))

DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print("Loading EMAGE...")
t0 = time.time()

from models.emage_audio import (  # noqa: E402
    EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel
)

SRC = str(WEIGHTS)
model = EmageAudioModel.from_pretrained(SRC).to(DEVICE).eval()
_parts = {
    name: EmageVQVAEConv.from_pretrained(SRC, subfolder=f"emage_vq/{name}").to(DEVICE).eval()
    for name in ("face", "upper", "hands", "lower")
}
_global = EmageVAEConv.from_pretrained(SRC, subfolder="emage_vq/global").to(DEVICE).eval()
motion_vq = EmageVQModel(
    face_model=_parts["face"], upper_model=_parts["upper"],
    hands_model=_parts["hands"], lower_model=_parts["lower"],
    global_model=_global,
).to(DEVICE).eval()

SR = model.cfg.audio_sr
POSE_FPS = model.cfg.pose_fps
print(f"EMAGE loaded in {time.time()-t0:.1f}s | sr={SR} fps={POSE_FPS} "
      f"| VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB")

# Warm up: the first CUDA call compiles kernels, and paying that on a user's
# first sentence would show up as a visible stall.
with torch.no_grad():
    _dummy = np.zeros(SR, dtype=np.float32)
    _ = model.inference(
        torch.from_numpy(_dummy).float().to(DEVICE).unsqueeze(0),
        torch.zeros(1, 1).long().to(DEVICE), motion_vq,
        masked_motion=None, mask=None)
torch.cuda.synchronize()
print("warmed up")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], expose_headers=["*"],
)


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "EMAGE",
        "audio_sr": SR,
        "pose_fps": POSE_FPS,
        "output": {
            "motion": "SMPL-X axis-angle, 55 joints x 3 = 165 per frame",
            "expression": "FLAME coefficients, 100 per frame",
            "trans": "global translation, 3 per frame",
        },
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }


def generate_motion(audio_np: np.ndarray):
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


def to_b64(arr: np.ndarray) -> str:
    """float32 little-endian, so the browser can read it as a Float32Array
    with no per-number parsing."""
    return base64.b64encode(arr.astype("<f4").tobytes()).decode()


@app.post("/motion")
async def motion(file: UploadFile = File(...)):
    raw = await file.read()

    t0 = time.time()
    import soundfile as sf
    audio_np, file_sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio_np.ndim > 1:                       # stereo → mono
        audio_np = audio_np.mean(axis=1)
    if file_sr != SR:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=file_sr, target_sr=SR)
    duration = len(audio_np) / SR
    t_decode = time.time() - t0

    t0 = time.time()
    pred = generate_motion(audio_np)
    torch.cuda.synchronize()
    t_infer = time.time() - t0

    motion = pred["motion_axis_angle"][0].cpu().numpy()   # (frames, 165)
    expression = pred["expression"][0].cpu().numpy()      # (frames, 100)
    translation = pred["trans"][0].cpu().numpy()          # (frames, 3)
    n_frames = motion.shape[0]

    print(f"[MOTION] audio {duration:.2f}s → {n_frames} frames "
          f"| decode {t_decode:.3f}s infer {t_infer:.3f}s "
          f"| RTF {t_infer/max(duration, 1e-6):.3f}")

    return {
        "fps": POSE_FPS,
        "frames": n_frames,
        "audio_duration": round(duration, 3),
        "motion": to_b64(motion),            # 165 per frame, SMPL-X axis-angle
        "expression": to_b64(expression),    # 100 per frame, FLAME
        "trans": to_b64(translation),        # 3 per frame
        "timing": {"decode": round(t_decode, 3), "infer": round(t_infer, 3)},
    }
