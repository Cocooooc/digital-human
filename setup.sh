#!/bin/bash
# Digital Human — one-shot environment recovery script
# Run this after cloning the repo onto a fresh server.
set -e

echo "Installing dependencies..."
pip install funasr
pip install -U transformers accelerate modelscope qwen-tts soundfile
pip install fastapi uvicorn python-multipart
apt update && apt install -y sox libsox-dev ffmpeg

echo ""
echo "Verifying CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — check the base image/driver'; print('CUDA OK:', torch.__version__)"

echo ""
echo "Downloading models via ModelScope..."
modelscope download --model Qwen/Qwen3-4B --local_dir /root/qwen3-4b
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir /root/qwen3-tts

echo ""
echo "Environment ready."
echo ""
echo "Remaining manual steps:"
echo "  1. Upload the voice-cloning reference audio (test10s.wav):"
echo "     scp -P <port> test10s.wav root@<server-ip>:/root/"
echo "  2. Start the server:"
echo "     python -m uvicorn server_stream:app --host 0.0.0.0 --port 15333"
