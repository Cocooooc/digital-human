#!/bin/bash
# PantoMatrix (EMAGE) — isolated install.
#
# Design decisions, and why:
#
# 1. Separate conda env (py39). The working conversation stack runs on 3.10
#    with torch 2.6+cu124. PantoMatrix wants 3.9 and pins numpy==1.23.5 and
#    xformers==0.0.19. Installing into the existing env would risk the system
#    that currently works — a careless pip install already silently replaced
#    torch with a CPU build once on this project.
#
# 2. Minimal dependency set, not requirements.txt wholesale. Roughly half of
#    that file is training-only (wandb, lpips, diffusers, controlnet-aux,
#    xformers). Inference doesn't need them, and xformers 0.0.19 was built for
#    the torch 2.0 era — pip would happily downgrade torch to satisfy it.
#    Install what inference needs; add more only if an import actually fails.
#
# 3. hf-mirror for weights. Hugging Face direct is ~1MB/s from this region.

set -e

ENV_NAME=pantomatrix
REPO=/root/PantoMatrix

echo "=============================================="
echo "0. Guard: record the existing env so we can prove we didn't break it"
echo "=============================================="
echo -n "base torch before: "
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo ""
echo "=============================================="
echo "1. Create isolated Python 3.9 environment"
echo "=============================================="
source /root/miniconda3/etc/profile.d/conda.sh
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "env '${ENV_NAME}' already exists, reusing"
else
  conda create -n ${ENV_NAME} python=3.9 -y
fi
conda activate ${ENV_NAME}
python --version

echo ""
echo "=============================================="
echo "2. torch first, pinned to this machine's CUDA"
echo "=============================================="
# Installing torch before anything else means later packages resolve against
# a CUDA build that already exists, instead of pulling a CPU wheel in.
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in new env'; print('env torch:', torch.__version__, '| CUDA OK')"

echo ""
echo "=============================================="
echo "3. Inference dependencies only"
echo "=============================================="
pip install \
  "numpy==1.23.5" \
  librosa \
  soundfile \
  smplx \
  ConfigArgParse \
  igraph \
  huggingface_hub \
  tqdm \
  opencv-python \
  scipy

echo ""
echo "=============================================="
echo "4. Repo + its local helper package"
echo "=============================================="
cd ${REPO}
# emage_utils / emage_evaltools ship as local packages in the repo
if [ -f setup.py ] || [ -f pyproject.toml ]; then
  pip install -e . || echo "editable install failed — will rely on PYTHONPATH instead"
fi

echo ""
echo "=============================================="
echo "5. Model weights (hf-mirror: ~10x faster from here)"
echo "=============================================="
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"

# The test script pulls checkpoints by name at runtime; pre-fetching them here
# makes the first inference fast and surfaces download problems now rather
# than in the middle of a timing measurement.
huggingface-cli download H-Liu1997/emage_audio --local-dir /root/emage_weights || \
  echo "WARNING: weight pre-fetch failed; test_emage_audio.py may still fetch them itself"

echo ""
echo "=============================================="
echo "6. Verify the base env is still intact"
echo "=============================================="
conda deactivate
echo -n "base torch after: "
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo ""
echo "=============================================="
echo "Done."
echo ""
echo "  Activate:  conda activate ${ENV_NAME}"
echo "  Next:      python /root/diagnose_emage.py"
echo ""
echo "If base torch printed the same thing before and after, the existing"
echo "conversation stack is untouched."
echo "=============================================="
