# Digital Human — Real-Time Voice Conversation System

A locally-deployed conversational AI pipeline that listens, thinks, and speaks: **ASR → LLM → TTS**, connected over HTTP, running entirely on a self-hosted GPU server (no external API calls at runtime).

Two stacks are included — an English one (`server_english.py`) and a Chinese one (`server_stream.py`) — because the right model choice turned out to depend almost entirely on the target language.

## Overview

```
   Client (Browser)                       GPU Server
   ─────────────────                      ──────────

   Chrome Web App
    ├ 🎤 Microphone recording               uvicorn (FastAPI)
    ├ Upload audio     ── HTTP ──→           ├ ASR      (listen)
    └ 🔊 Playback       ←──────────          ├ LLM      (think)
                                             ├ TTS      (speak)
                                             └ conversation history
```

The client only handles recording and playback; all inference runs on the server.

## Tech Stack

| Component | English stack (`server_english.py`) | Chinese stack (`server_stream.py`) |
|---|---|---|
| ASR | faster-whisper `small` | FunASR (paraformer-zh + fsmn-vad + ct-punc) |
| LLM | Qwen3-4B | Qwen3-4B |
| TTS | Kokoro 82M | Qwen3-TTS-12Hz-0.6B-Base (voice cloning) |
| Server | FastAPI + uvicorn | FastAPI + uvicorn |
| Client | Vanilla JS web page | same |

## Performance

Measured on a single NVIDIA A800-40GB, warm.

### English stack

Component throughput:

| Stage | RTF | Notes |
|---|---|---|
| ASR (faster-whisper small) | **0.041** | ~24× faster than real time |
| TTS (Kokoro 82M) | **0.022** | ~45× faster than real time; 2.3s of speech synthesized in 50ms |
| **Bottleneck** | — | **the LLM**, not TTS |

End-to-end, measured live over three consecutive exchanges:

| Turn | ASR | Time to first audio | Full reply |
|---|---|---|---|
| "Hello, I'm home." | 0.54s | 0.57s | 0.76s |
| "Can you hear me?" | 0.37s | 0.40s | 0.59s |
| "How are you doing?" | 0.37s | 0.38s | 0.55s |

**Total round trip ≈ 1 second**, with speech beginning under 0.6s after the request — close to the 200–500ms pause length of natural human conversation. The Chinese stack on the same hardware took roughly 60 seconds for the equivalent English exchange.

### Chinese stack

| Stage | Latency | Share |
|---|---|---|
| ASR | 0.21s | 2% |
| LLM | 0.70s | 6% |
| TTS | ~11–19s | ~85% (bottleneck) |

### Why the English stack is ~100× faster at synthesis

Qwen3-TTS measured **RTF ≈ 3.9 plus a ~6 second fixed overhead per call** in this environment (versus RTF ≈ 1.5 recorded in earlier runs on a different instance — the gap was never fully explained, but library versions had changed). Kokoro measured **RTF ≈ 0.022 with no meaningful fixed cost**, roughly a 177× improvement in synthesis throughput. End-to-end latency for an English exchange went from ~60s to a few seconds.

The lesson isn't that Kokoro beats Qwen3-TTS in general — it's that running a Chinese-optimized voice-cloning model on English text was an architectural mismatch, and no amount of parameter tuning fixed it. Four separate optimization attempts on the Chinese TTS path (precomputed voice-clone prompts, x-vector-only mode, non-streaming mode, bfloat16) produced no meaningful change; switching to a model built for the target language changed everything.

## Features

- **End-to-end voice pipeline**: speech in, speech out, one HTTP round trip
- **Multi-turn memory**: recent conversation turns are re-sent as LLM context each request
- **Streaming output**: sentence-level pipelining, so playback starts before the full reply is generated
- **Fully self-hosted**: no external API dependency at runtime; data never leaves the server

## Getting Started

### Prerequisites
- Linux server with an NVIDIA GPU (tested on A800-40GB, CUDA 12.4)
- Python 3.10+

### English stack

```bash
git clone https://github.com/Cocooooc/digital-human.git
cd digital-human
bash setup.sh          # installs deps, downloads Qwen3-4B
bash setup_english.sh  # installs faster-whisper + Kokoro + espeak-ng
python test_english_stack.py   # verify APIs and measure RTF
python -m uvicorn server_english:app --host 0.0.0.0 --port <your-port>
```

### Chinese stack

```bash
bash setup.sh   # also downloads Qwen3-TTS

# Reference audio for voice cloning (not tracked in git)
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav -O /root/test10s.wav
ffmpeg -y -i /root/test10s.wav -ar 16000 -ac 1 -c:a pcm_s16le /root/test10s_16k.wav
mv /root/test10s_16k.wav /root/test10s.wav

python -m uvicorn server_stream:app --host 0.0.0.0 --port <your-port>
```

`REF_TEXT` in `server_stream.py` must match the reference clip word-for-word. To regenerate it after swapping the clip:

```bash
python -c "
from funasr import AutoModel
model = AutoModel(model='paraformer-zh', vad_model='fsmn-vad', punc_model='ct-punc', device='cuda:0')
print(model.generate(input='/root/test10s.wav')[0]['text'])
"
```

### Client

```bash
python -m http.server 8000
# open http://localhost:8000/digital_human_stream.html
```

Browsers only grant microphone access on `localhost` or HTTPS — opening the file directly via `file://` will not work. Set the server URL in the page's input box to match your server's address and port.

## API

| Endpoint | Description |
|---|---|
| `GET /` | Health check; returns turn count and which models are loaded |
| `POST /chat` | Non-streaming — upload audio, receive a complete WAV |
| `POST /chat_stream` | Streaming — NDJSON, one event per line |
| `POST /reset` | Clear conversation memory |

Streaming event format:

```json
{"type": "asr",     "text": "..."}
{"type": "emotion", "emotion": "happy"}
{"type": "audio",   "text": "...", "wav": "<base64 wav>", "phonemes": "..."}
{"type": "done",    "text": "...", "emotion": "happy", "first_audio": 0.4, "total": 0.6, "segments": 2}
```

## Avatar Driving (English stack)

The server emits two extra signals so a rendered avatar can lip-sync and emote:

**Emotion.** The LLM is instructed to prefix each reply with a tag — `[happy] That's great to hear!` — chosen from `neutral`, `happy`, `sad`, `surprised`, `thinking`, `excited`. The server strips the tag before synthesis (so it is never spoken), stores the untagged text in history, and sends the emotion as its own event *before* the first audio chunk, letting the avatar react before it starts speaking. Unrecognized or missing tags fall back to `neutral`.

**Phonemes.** Kokoro's pipeline yields `(graphemes, phonemes, audio)`; the phoneme string is passed through to the client, giving accurate mouth shapes rather than the cruder approach of deriving mouth opening from audio amplitude.

On `/chat` these arrive as the `X-Emotion` and `X-Phonemes` response headers; on `/chat_stream` as the `emotion` event and the `phonemes` field of each `audio` event.

## Design Notes

**Conversation memory.** The LLM is stateless. "Memory" means prepending recent turns to every request:

```python
from collections import deque
history = deque(maxlen=12)  # last 6 turns

messages = ([{"role": "system", "content": SYSTEM}]
            + list(history)
            + [{"role": "user", "content": user_text}])
```

Longer history means more input tokens and slower generation. This is a single global history — concurrent users would need per-session keying.

**Sentence-level streaming.** As the LLM streams tokens, text is buffered until a sentence boundary, then sent straight to TTS and pushed to the client while generation continues. In the Chinese stack this existed to hide slow synthesis; in the English stack synthesis is essentially free, so it exists to start speaking before the LLM finishes.

**Non-ASCII in HTTP headers.** `/chat` returns the transcript and reply as response headers, but HTTP headers are latin-1 only — putting Chinese text in one raises `UnicodeEncodeError` and the request fails with a 500 that the browser misreports as a CORS error. The fix is percent-encoding server-side (`urllib.parse.quote`) and `decodeURIComponent()` client-side.

**Measure RTF, not wall-clock time.** TTS output length varies run to run due to sampling randomness, so raw elapsed time is misleading — a "faster" run is often just a shorter utterance. Real-time factor (elapsed ÷ audio duration) is the only comparable metric.

**Loading Qwen3-TTS.** `Qwen3TTSModel.from_pretrained()` takes no `device` argument and loads on CPU. The underlying module is at `.model` and has a normal `.to()`, but the wrapper's `.device` attribute is a plain writable field that other methods read when building tensors — both must be set, or generation crashes with a device mismatch.

## Project Status

- [x] ASR / LLM / TTS individually validated
- [x] Three modules pipelined
- [x] HTTP service (streaming + non-streaming)
- [x] Multi-turn conversation memory
- [x] Web client (microphone input, auto playback)
- [x] English stack with ~100× faster synthesis
- [ ] Streaming input (WebSocket + streaming ASR)
- [ ] Per-session history for concurrent users
- [ ] Fine-tuning

## License

MIT
