#!/bin/bash
# Digital Human — English stack setup
# Installs faster-whisper (ASR) and Kokoro (TTS), replacing the Chinese-optimized
# FunASR + Qwen3-TTS stack. The LLM (Qwen3-4B) is unchanged.
set -e

echo "Installing espeak-ng (Kokoro's phonemizer backend)..."
apt update && apt install -y espeak-ng ffmpeg

echo ""
echo "Installing faster-whisper (ASR)..."
pip install faster-whisper

echo ""
echo "Installing Kokoro (TTS)..."
pip install kokoro soundfile

echo ""
echo "Verifying CUDA still works after installs..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA broke — a pip install likely pulled a CPU-only torch'; print('CUDA OK:', torch.__version__)"

echo ""
echo "Setup complete. Next: run test_english_stack.py to verify APIs and measure speed."
