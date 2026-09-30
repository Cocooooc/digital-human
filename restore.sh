#!/bin/bash
# Digital Human — full recovery on a fresh server.
#
# Supersedes setup.sh / setup_english.sh / setup_pantomatrix.sh, which were
# written at different points and no longer agree with what actually works.
# Every fix this project paid for is folded in here:
#
#   torch >= 2.6 in the EMAGE env. transformers 4.57 refuses torch.load on
#     account of CVE-2025-32434, and EMAGE's weights are .bin, not safetensors.
#     The old script pinned 2.4.1, which now fails at model load.
#   torch/torchvision/torchaudio upgraded together. Bumping torch alone leaves
#     torchvision compiled against the old ABI: "operator torchvision::nms
#     does not exist".
#   transformers, omegaconf, fastapi, uvicorn, python-multipart — discovered
#     missing one ImportError at a time.
#   SMPL-X from emage_evaltools. The URL hard-coded in PantoMatrix's own
#     motion_rep_transfer.py points at an HF Space that was since replaced;
#     it 404s.
#   No Qwen3-TTS. The Chinese TTS stack was replaced by Kokoro; downloading it
#     costs several GB for nothing.
#
# Usage, on a fresh box:
#     apt update && apt install -y git
#     cd /root && git clone https://github.com/Cocooooc/digital-human.git
#     cd digital-human && bash restore.sh
#
# Downloads are ~9GB, so run it under tmux — an ssh drop mid-download otherwise
# means starting that phase again:
#     tmux new -s setup     (reattach later with: tmux attach -t setup)
#
# Re-running is safe: each phase records completion and is skipped next time.
# Force one with:  bash restore.sh --redo 3
set -u

REPO_DIR=/root/digital-human
PANTO=/root/PantoMatrix
WEIGHTS=/root/emage_weights
LLM=/root/qwen3-4b
SMPLX_DIR=$PANTO/emage_evaltools/smplx_models/smplx
STAMPS=/root/.restore_done
ENV_NAME=pantomatrix

mkdir -p "$STAMPS"
REDO=""
[ "${1:-}" = "--redo" ] && REDO="${2:-}"

banner() { printf '\n\033[1;36m═══ %s ═══\033[0m\n' "$1"; }
ok()     { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn()   { printf '  \033[33m!\033[0m %s\n' "$1"; }

phase() {           # phase <n> <name>; returns 1 if it should be skipped
  local n="$1" name="$2"
  if [ -f "$STAMPS/$n" ] && [ "$REDO" != "$n" ]; then
    printf '\n\033[2m─── %s. %s — already done, skipping (bash restore.sh --redo %s to force)\033[0m\n' "$n" "$name" "$n"
    return 1
  fi
  banner "$n. $name"
  PHASE_T0=$(date +%s)
  return 0
}
done_phase() { touch "$STAMPS/$1"; ok "phase $1 done in $(( $(date +%s) - PHASE_T0 ))s"; }

T0=$(date +%s)

# ---------------------------------------------------------------------------
if phase 1 "System packages"; then
  apt-get update -qq
  # espeak-ng: Kokoro's phonemiser. sox/ffmpeg: audio conversion.
  # The GL libraries are for headless rendering; installing them now avoids a
  # second apt round later.
  apt-get install -y -qq git wget tmux sox libsox-dev ffmpeg espeak-ng \
      libgl1 libgl1-mesa-dri libegl1 libglib2.0-0 \
      libosmesa6 libosmesa6-dev freeglut3-dev 2>&1 | tail -2
  ok "$(git --version)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU: /'
  done_phase 1
fi

# ---------------------------------------------------------------------------
if phase 2 "Conversation stack (base env)"; then
  python -c "import torch; print('  base torch', torch.__version__, '| cuda', torch.cuda.is_available())"
  pip install -q -U transformers accelerate modelscope soundfile \
      faster-whisper kokoro fastapi uvicorn python-multipart requests
  # A pip install has silently replaced torch with a CPU build on this project
  # before; if that happened, everything downstream would be slow and it would
  # not be obvious why.
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA lost — a pip install pulled a CPU torch'; print('  still CUDA OK:', torch.__version__)"
  done_phase 2
fi

# ---------------------------------------------------------------------------
if phase 3 "LLM weights (~8GB)"; then
  if [ -d "$LLM" ] && [ -n "$(ls -A "$LLM" 2>/dev/null)" ]; then
    ok "$LLM already present"
  else
    modelscope download --model Qwen/Qwen3-4B --local_dir "$LLM"
  fi
  du -sh "$LLM" | sed 's/^/  /'
  done_phase 3
fi

# ---------------------------------------------------------------------------
if phase 4 "PantoMatrix repo"; then
  if [ -d "$PANTO/.git" ]; then
    ok "already cloned"
  else
    git clone --depth 1 https://github.com/PantoMatrix/PantoMatrix.git "$PANTO"
  fi
  ls "$PANTO" | head -8 | sed 's/^/  /'
  done_phase 4
fi

# ---------------------------------------------------------------------------
if phase 5 "EMAGE environment (isolated py39)"; then
  # Isolated because PantoMatrix wants python 3.9 and numpy 1.23.5, while the
  # conversation stack runs 3.10 with a much newer numpy. Merging them would
  # put a working system at risk to save a few hundred MB.
  source /root/miniconda3/etc/profile.d/conda.sh
  conda env list | grep -q "^${ENV_NAME} " || conda create -n ${ENV_NAME} python=3.9 -y
  conda activate ${ENV_NAME}
  python --version | sed 's/^/  /'

  # All three together, and >= 2.6. See the header.
  pip install -q -U torch torchvision torchaudio
  python -c "import torch, torchvision; print('  env torch', torch.__version__, '| tv', torchvision.__version__, '| cuda', torch.cuda.is_available())"
  python -c "import torch; from torchvision.ops import nms; assert torch.cuda.is_available()" \
    && ok "torch/torchvision ABI match" || warn "torchvision mismatch — reinstall all three together"

  pip install -q "numpy==1.23.5" librosa soundfile smplx ConfigArgParse igraph \
      huggingface_hub tqdm opencv-python scipy transformers einops omegaconf \
      fastapi uvicorn python-multipart
  python -c "import numpy; print('  numpy', numpy.__version__)"
  conda deactivate
  done_phase 5
fi

# ---------------------------------------------------------------------------
if phase 6 "EMAGE weights (~613MB)"; then
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}
  if [ -f "$WEIGHTS/pytorch_model.bin" ]; then
    ok "already downloaded"
  else
    # hf-mirror is roughly 10x faster from this region.
    export HF_ENDPOINT=https://hf-mirror.com
    pip install -q -U "huggingface_hub[cli]"
    huggingface-cli download H-Liu1997/emage_audio --local-dir "$WEIGHTS" \
      || warn "download failed — retry, or unset HF_ENDPOINT to use huggingface.co"
  fi
  du -sh "$WEIGHTS" 2>/dev/null | sed 's/^/  /'
  conda deactivate
  done_phase 6
fi

# ---------------------------------------------------------------------------
if phase 7 "SMPL-X body model (167MB)"; then
  mkdir -p "$SMPLX_DIR"
  URL="https://huggingface.co/H-Liu1997/emage_evaltools/resolve/main/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
  if [ -s "$SMPLX_DIR/SMPLX_NEUTRAL_2020.npz" ]; then
    ok "already downloaded"
  else
    wget -q --show-progress -O "$SMPLX_DIR/SMPLX_NEUTRAL_2020.npz" "$URL" \
      || wget -q --show-progress -O "$SMPLX_DIR/SMPLX_NEUTRAL_2020.npz" "${URL/huggingface.co/hf-mirror.com}"
  fi
  # smplx.create() looks for the undated name; the renderer reads the dated one.
  ln -sf SMPLX_NEUTRAL_2020.npz "$SMPLX_DIR/SMPLX_NEUTRAL.npz"
  ls -lh "$SMPLX_DIR" | tail -2 | sed 's/^/  /'
  done_phase 7
fi

# ---------------------------------------------------------------------------
if phase 8 "Headless rendering"; then
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}
  pip install -q "numpy==1.23.5" pyrender trimesh "pyglet<2" \
      imageio imageio-ffmpeg matplotlib PyOpenGL PyOpenGL-accelerate
  python -c "import numpy; assert numpy.__version__.startswith('1.23'), 'numpy moved — EMAGE will break'; print('  numpy still', numpy.__version__)"

  # A machine with no display needs an offscreen GL backend. EGL uses the GPU;
  # OSMesa falls back to the CPU. Draw one cube through each and see which
  # answers, rather than discovering it 300 frames into a render.
  BACKEND=""
  for b in egl osmesa; do
    if PYOPENGL_PLATFORM=$b python - <<'PY' 2>/dev/null
import numpy as np, trimesh, pyrender
s = pyrender.Scene(ambient_light=[.5]*3)
s.add(pyrender.Mesh.from_trimesh(trimesh.creation.box()))
p = np.eye(4); p[2, 3] = 3
s.add(pyrender.PerspectiveCamera(yfov=np.pi/3), pose=p)
s.add(pyrender.DirectionalLight(intensity=3), pose=p)
r = pyrender.OffscreenRenderer(64, 64); c, _ = r.render(s); r.delete()
raise SystemExit(0 if c.mean() > 0 else 1)
PY
    then ok "offscreen backend: $b"; BACKEND=$b; break
    else warn "$b unavailable"; fi
  done
  if [ -n "$BACKEND" ]; then
    echo "export PYOPENGL_PLATFORM=$BACKEND" > /root/.render_env
    ok "wrote /root/.render_env  (source it before rendering)"
  else
    warn "no offscreen backend — rendering will fail; send the errors above"
  fi
  conda deactivate
  done_phase 8
fi

# ---------------------------------------------------------------------------
banner "Verification"
source /root/miniconda3/etc/profile.d/conda.sh
check() { [ -e "$2" ] && ok "$1" || warn "MISSING: $1  ($2)"; }
check "repo"              "$REPO_DIR/server_english.py"
check "PantoMatrix"       "$PANTO/test_emage_audio.py"
check "EMAGE weights"     "$WEIGHTS/pytorch_model.bin"
check "SMPL-X"            "$SMPLX_DIR/SMPLX_NEUTRAL_2020.npz"
check "LLM"               "$LLM"
check "render backend"    "/root/.render_env"
conda env list | grep -q "^${ENV_NAME} " && ok "conda env ${ENV_NAME}" || warn "MISSING: conda env"

cat <<TXT

  Total: $(( ($(date +%s) - T0) / 60 ))m$(( ($(date +%s) - T0) % 60 ))s

  Start the services — each in its own window, on a port the console
  actually exposes (the range changes with every instance, and a port
  outside it listens locally while being unreachable from outside):

    # conversation
    cd $REPO_DIR && python -m uvicorn server_english:app --host 0.0.0.0 --port <PORT>

    # motion
    conda activate ${ENV_NAME}
    cd $REPO_DIR && python -m uvicorn emage_server:app --host 0.0.0.0 --port <PORT>

    # render something
    conda activate ${ENV_NAME} && source /root/.render_env
    cd $REPO_DIR && python face_export.py <motion>.npz --audio <clip>.wav --side-by-side

TXT
