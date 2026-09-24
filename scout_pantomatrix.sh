#!/bin/bash
# PantoMatrix — RECONNAISSANCE ONLY. Installs nothing, changes nothing.
#
# Why this exists: PantoMatrix wants Python 3.9, but the working conversation
# stack (faster-whisper + Qwen3-4B + Kokoro) runs on 3.10.16. Installing into
# that environment risks breaking a system that currently works — last time a
# careless pip install silently replaced torch with a CPU build.
#
# So: look first, decide how to isolate, install second.
#
# Run this, then send the output back before installing anything.

set -u
echo "=============================================="
echo "1. Current environment (the one we must NOT break)"
echo "=============================================="
python --version
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())" 2>/dev/null || echo "torch not importable"
echo "conda envs:"
conda env list 2>/dev/null || echo "  (conda not found)"
echo ""
echo "python3.9 available? $(which python3.9 || echo 'no')"
echo "free disk:"
df -h /root | tail -1
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo ""
echo "=============================================="
echo "2. Fetching PantoMatrix source (no install)"
echo "=============================================="
cd /root
if [ -d PantoMatrix ]; then
  echo "already cloned, pulling latest"
  cd PantoMatrix && git pull
else
  git clone https://github.com/PantoMatrix/PantoMatrix.git
  cd PantoMatrix
fi

echo ""
echo "--- repo layout (top level) ---"
ls -la

echo ""
echo "--- what setup.sh actually does ---"
if [ -f setup.sh ]; then cat setup.sh; else echo "NO setup.sh FOUND"; fi

echo ""
echo "--- requirements ---"
for f in requirements.txt requirement.txt environment.yml pyproject.toml; do
  if [ -f "$f" ]; then echo "### $f"; cat "$f"; fi
done

echo ""
echo "--- inference entry points ---"
ls -1 test_*.py 2>/dev/null || echo "no test_*.py at top level"

echo ""
echo "--- model definitions ---"
ls -la models/ model/ 2>/dev/null | head -40

echo ""
echo "--- how EMAGE is actually invoked (real class name + args) ---"
if [ -f test_emage_audio.py ]; then
  head -60 test_emage_audio.py
else
  echo "test_emage_audio.py not found; searching for emage entry points:"
  grep -rl "emage" --include="*.py" . 2>/dev/null | head -10
fi

echo ""
echo "--- example audio provided? ---"
ls -la examples/audio/ 2>/dev/null || echo "no examples/audio"

echo ""
echo "=============================================="
echo "Done. Nothing was installed. Send this output back."
echo "=============================================="
