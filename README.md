# Digital Human — Real-Time Voice Conversation System

A locally-deployed conversational AI pipeline that listens, thinks, and speaks: **ASR → LLM → TTS**, fully connected over HTTP, running entirely on a self-hosted GPU server (no external API calls at runtime).

## Overview

```
   Client (Browser)                       GPU Server
   ─────────────────                      ──────────

   Chrome Web App
    ├ 🎤 Microphone recording               uvicorn (FastAPI)
    ├ Upload audio     ── HTTP ──→           ├ FunASR        (listen)
    └ 🔊 Playback       ←──────────          ├ Qwen3-4B      (think)
                                             ├ Qwen3-TTS     (speak)
                                             └ conversation history
```

The client only handles recording and playback; all inference runs on the server.

## Features

- **End-to-end voice pipeline**: speech in, speech out, single HTTP round trip
- **Multi-turn memory**: conversation history is fed back into the LLM context each turn
- **Streaming output**: sentence-level pipelining — the first audio segment plays while the LLM is still generating the rest of the reply, cutting time-to-first-audio from ~12s to ~2s
- **Voice cloning**: TTS output is cloned from a short reference audio sample
- **Fully self-hosted**: no dependency on external APIs after model download; data never leaves the server

## Tech Stack

| Component | Model | Notes |
|---|---|---|
| ASR | FunASR (paraformer-zh + fsmn-vad + ct-punc) | RTF ≈ 0.015 |
| LLM | Qwen3-4B | Thinking mode disabled for low latency |
| TTS | Qwen3-TTS-12Hz-0.6B-Base | Voice cloning via reference audio |
| Server | FastAPI + uvicorn | Non-streaming (`/chat`) and streaming (`/chat_stream`) endpoints |
| Client | Vanilla JS web page | Browser microphone capture, queued audio playback |

## Performance

Measured end-to-end latency (warm, single GPU — NVIDIA A800 40GB):

| Stage | Latency | Share |
|---|---|---|
| ASR | 0.21s | 2% |
| LLM | 0.70s | 6% |
| TTS | ~11s | 85% (bottleneck) |
| **Total (non-streaming)** | **~12s** | |
| **First audio (streaming pipeline)** | **~2s** | |

TTS is the dominant cost. The model's real-time factor (RTF ≈ 1.5) was found to be an inherent property of its autoregressive generation — four separate optimization attempts (precomputed voice-clone prompts, x-vector-only mode, non-streaming mode, flash-attention) produced no measurable improvement. Sentence-level streaming was implemented instead, so the user hears the first sentence while later sentences are still being synthesized.

## Getting Started

### Prerequisites
- Linux server with an NVIDIA GPU (tested on A800-40GB, CUDA 12.4)
- Python 3.10+

### Setup

```bash
git clone https://github.com/<your-username>/digital-human.git
cd digital-human
bash setup.sh
```

`setup.sh` installs dependencies and downloads the LLM/TTS models via ModelScope.

Then upload a reference audio clip for voice cloning (not tracked in git):

```bash
scp -P <port> test10s.wav root@<server-ip>:/root/digital-human/
```

Start the server:

```bash
python -m uvicorn server_stream:app --host 0.0.0.0 --port 15333
```

Open the client from `localhost` (browsers only grant microphone access on localhost or HTTPS):

```bash
python -m http.server 8000
# then open http://localhost:8000/digital_human_stream.html
```

## API

| Endpoint | Description |
|---|---|
| `GET /` | Health check, returns current conversation turn count |
| `POST /chat` | Non-streaming — upload audio, receive a complete WAV response |
| `POST /chat_stream` | Streaming — NDJSON, one event per line |
| `POST /reset` | Clear conversation memory |

Streaming event format:

```json
{"type": "asr",   "text": "..."}
{"type": "audio", "text": "...", "wav": "<base64 wav>"}
{"type": "done",  "text": "...", "first_audio": 1.83, "total": 4.12, "segments": 2}
```

## Design Notes

**Conversation memory**: the LLM itself is stateless. "Memory" is implemented by prepending recent turns to every request:

```python
from collections import deque
history = deque(maxlen=12)  # last 6 turns

messages = ([{"role": "system", "content": SYSTEM}]
            + list(history)
            + [{"role": "user", "content": user_text}])
```

**Sentence-level streaming pipeline**: as the LLM streams tokens, text is buffered until a sentence-ending punctuation mark is hit, then immediately sent to TTS and pushed to the client — while the LLM continues generating the next sentence. This overlaps the two slowest stages instead of running them sequentially.

**Why Qwen3-4B over larger models**: for real-time conversation, latency matters more than raw capability. Models above ~14B were too slow for interactive use; 4B was chosen as the best latency/quality tradeoff within the Qwen3 family.

**Why local deployment**: no per-request API cost, data stays on the server, and the base TTS model supports fine-tuning (not possible with hosted APIs).

## Project Status

- [x] ASR / LLM / TTS individually validated
- [x] Three modules pipelined
- [x] Non-streaming HTTP service
- [x] Multi-turn conversation memory
- [x] Web client (microphone input, auto playback)
- [x] Streaming output (sentence-level pipeline)
- [ ] Streaming input (WebSocket + streaming ASR)
- [ ] Fine-tuning

## License

MIT
