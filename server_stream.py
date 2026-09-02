"""
Digital Human — voice conversation server.

Pipeline: ASR (FunASR) -> LLM (Qwen3-4B) -> TTS (Qwen3-TTS voice clone)

Two HTTP endpoints:
  POST /chat         non-streaming: upload audio, get back one complete WAV file
  POST /chat_stream   streaming: NDJSON, one JSON event per line, sentence-level
                      pipeline (TTS starts on sentence 1 while the LLM is still
                      generating sentence 2, so the first audio arrives fast)
  POST /reset          clear the conversation history
  GET  /               health check

Run with:
  python -m uvicorn server_stream:app --host 0.0.0.0 --port 15333

NOTE: this is a reconstruction from the project's design notes (the working
copy that lived only on the previous server instance was lost when that
instance was torn down). Re-verify against the actual funasr / qwen_tts
library signatures on the server — some call parameters may need small
adjustments (see the README "Design Notes" section and the project doc for
the exact API signatures that were confirmed to work).
"""

import base64
import io
import json
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
from threading import Thread

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_PATH = "/root/qwen3-4b"
TTS_PATH = "/root/qwen3-tts"
REF_AUDIO = "/root/test10s.wav"
# Must match, word for word, what is actually said in REF_AUDIO.
REF_TEXT = "一个游戏开发小组当中遇到的一段经历。那我在这个小组当中呢，负责的是。"

ASR_SAMPLE_RATE = 16000
HISTORY_TURNS = 6  # keep the last N user/assistant turn pairs
MAX_NEW_TOKENS = 60

# Sentence-level streaming pipeline: cut the LLM's streamed output into
# chunks at these punctuation marks, and only once the buffered chunk has at
# least MIN_CHARS characters (too short = choppy audio, too long = slower
# first-audio latency).
CUT_MARKS = "。！？；，、\n"
MIN_CHARS = 5

SYSTEM_PROMPT = (
    "You are a friendly voice assistant having a live spoken conversation. "
    "Keep replies short, natural, and conversational, the way a person would "
    "actually talk out loud — no lists, no markdown, no written-style "
    "structure. Respond in the same language the user is speaking, which "
    "will normally be Chinese."
)

# ---------------------------------------------------------------------------
# Model loading (once, at process startup — all three models stay resident)
# ---------------------------------------------------------------------------

print("Loading ASR (FunASR)...")
from funasr import AutoModel

asr_model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    device="cuda:0",
)

print("Loading LLM (Qwen3-4B)...")
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

print("Loading TTS (Qwen3-TTS)...")
import qwen_tts  # confirm the actual class name on the server; adjust if needed

tts_model = qwen_tts.Qwen3TTS.from_pretrained(TTS_PATH, device="cuda:0")
voice_clone_prompt = tts_model.create_voice_clone_prompt(REF_AUDIO, ref_text=REF_TEXT)

print("All models loaded.")

# ---------------------------------------------------------------------------
# Conversation memory
#
# The LLM itself is stateless — "memory" means re-sending recent turns as
# part of the input every time. This is a single global history: fine for a
# single-user demo, but multiple concurrent users would need per-session
# history (e.g. keyed by a client-supplied session id) instead.
# ---------------------------------------------------------------------------

history: deque = deque(maxlen=HISTORY_TURNS * 2)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Without this, the browser can't read custom response headers
    # (e.g. X-User-Text) even though the request itself succeeds.
    expose_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "turns": len(history) // 2}


@app.post("/reset")
def reset():
    history.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def convert_to_wav(raw_bytes: bytes) -> str:
    """Browsers record webm/opus, not wav. FunASR needs 16kHz mono PCM wav,
    so normalize with ffmpeg regardless of the input format."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f_in:
        f_in.write(raw_bytes)
        in_path = f_in.name

    out_path = in_path.replace(".webm", ".wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", in_path,
            "-ar", str(ASR_SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
            out_path,
        ],
        check=True,
        capture_output=True,
    )
    return out_path


def run_asr(wav_path: str) -> str:
    t0 = time.time()
    res = asr_model.generate(input=wav_path)
    text = res[0]["text"]
    print(f"[ASR] {time.time()-t0:.2f}s  -> {text!r}")
    return text


def build_messages(user_text: str):
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(history)
        + [{"role": "user", "content": user_text}]
    )


def synthesize(text: str) -> tuple[np.ndarray, int]:
    audio_chunks, sr = tts_model.generate_voice_clone(
        text=text,
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=False,
    )
    audio = np.concatenate(audio_chunks) if isinstance(audio_chunks, list) else audio_chunks
    return audio, sr


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# POST /chat — non-streaming
# ---------------------------------------------------------------------------


@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    raw = await file.read()
    wav_path = convert_to_wav(raw)

    user_text = run_asr(wav_path)

    t0 = time.time()
    messages = build_messages(user_text)
    inputs = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,  # must be set explicitly, or latency spikes
        return_tensors="pt",
    ).to("cuda:0")

    output_ids = llm_model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
    reply = llm_tokenizer.decode(
        output_ids[0][inputs.shape[-1]:], skip_special_tokens=True
    )
    print(f"[LLM] {time.time()-t0:.2f}s  -> {reply!r}")

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})

    t0 = time.time()
    audio, sr = synthesize(reply)
    print(f"[TTS] {time.time()-t0:.2f}s")

    wav_bytes = audio_to_wav_bytes(audio, sr)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-User-Text": user_text,
            "X-Reply-Text": reply,
        },
    )


# ---------------------------------------------------------------------------
# POST /chat_stream — streaming, sentence-level pipeline
#
# The LLM's output is streamed token-by-token via a background thread +
# TextIteratorStreamer. As text accumulates, each time a sentence boundary is
# hit the buffered sentence is immediately sent to TTS and the resulting
# audio is yielded to the client — while the LLM keeps generating the next
# sentence in parallel. This is what takes time-to-first-audio from ~12s
# (wait for the whole reply, then synthesize all of it) down to ~2s.
# ---------------------------------------------------------------------------


@app.post("/chat_stream")
async def chat_stream(file: UploadFile = File(...)):
    raw = await file.read()
    wav_path = convert_to_wav(raw)
    user_text = run_asr(wav_path)

    def event_stream():
        yield json.dumps({"type": "asr", "text": user_text}, ensure_ascii=False) + "\n"

        messages = build_messages(user_text)
        inputs = llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to("cuda:0")

        streamer = TextIteratorStreamer(
            llm_tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_thread = Thread(
            target=llm_model.generate,
            kwargs=dict(
                input_ids=inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                streamer=streamer,
            ),
        )

        t_start = time.time()
        gen_thread.start()

        full_reply = ""
        buf = ""
        first_audio_time = None
        n_segments = 0

        for piece in streamer:
            full_reply += piece
            buf += piece

            if len(buf.strip()) >= MIN_CHARS and buf.rstrip()[-1:] in CUT_MARKS:
                audio, sr = synthesize(buf)
                if first_audio_time is None:
                    first_audio_time = time.time() - t_start
                n_segments += 1
                wav_b64 = base64.b64encode(audio_to_wav_bytes(audio, sr)).decode()
                yield json.dumps(
                    {"type": "audio", "text": buf, "wav": wav_b64}, ensure_ascii=False
                ) + "\n"
                buf = ""

        # flush whatever's left in the buffer (reply didn't end on punctuation)
        if buf.strip():
            audio, sr = synthesize(buf)
            if first_audio_time is None:
                first_audio_time = time.time() - t_start
            n_segments += 1
            wav_b64 = base64.b64encode(audio_to_wav_bytes(audio, sr)).decode()
            yield json.dumps(
                {"type": "audio", "text": buf, "wav": wav_b64}, ensure_ascii=False
            ) + "\n"

        gen_thread.join()

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": full_reply})

        yield json.dumps(
            {
                "type": "done",
                "text": full_reply,
                "first_audio": round(first_audio_time or 0, 2),
                "total": round(time.time() - t_start, 2),
                "segments": n_segments,
            },
            ensure_ascii=False,
        ) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
