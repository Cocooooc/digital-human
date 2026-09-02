"""
Digital Human — English voice conversation server.

Pipeline: ASR (faster-whisper) -> LLM (Qwen3-4B) -> TTS (Kokoro)

Replaces the Chinese-optimized stack (FunASR + Qwen3-TTS) with English-first
models. Measured on an A800:

    faster-whisper small : RTF 0.041  (24x faster than real time)
    Kokoro 82M           : RTF 0.022  (45x faster than real time)
    vs Qwen3-TTS         : RTF 3.9 plus ~6s fixed overhead

TTS is no longer the bottleneck — the LLM is. That inverts the design from the
Chinese version: sentence-level streaming exists to overlap LLM generation with
synthesis, not to hide slow synthesis.

Endpoints:
  GET  /              health check
  POST /chat          non-streaming: upload audio, get one complete WAV back
  POST /chat_stream   streaming: NDJSON, one event per line
  POST /reset         clear conversation history

Run with:
  python -m uvicorn server_english:app --host 0.0.0.0 --port 15343
"""

import base64
import io
import json
import re
import subprocess
import tempfile
import time
from collections import deque
from threading import Thread
from urllib.parse import quote

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

# faster-whisper size: tiny / base / small / medium / large-v3.
# "small" is the latency/accuracy sweet spot for conversation.
ASR_MODEL_SIZE = "small"
ASR_LANGUAGE = "en"

# Kokoro voice. American female: af_bella, af_nicole, af_sarah, af_sky
#               American male:   am_adam, am_michael
#               British:         bf_emma, bf_isabella, bm_george, bm_lewis
TTS_VOICE = "af_bella"
TTS_LANG_CODE = "a"  # 'a' = American English, 'b' = British English
TTS_SAMPLE_RATE = 24000

ASR_SAMPLE_RATE = 16000
HISTORY_TURNS = 6
MAX_NEW_TOKENS = 80  # TTS is cheap now, so replies can be a bit longer

# Sentence boundaries for the streaming pipeline (English punctuation).
CUT_MARKS = ".!?;\n"
MIN_CHARS = 10

# Emotion tags the avatar knows how to render. The LLM is asked to prefix each
# reply with one of these; the tag is stripped before synthesis and sent to the
# client separately so it can drive facial blendshapes.
EMOTIONS = ["neutral", "happy", "sad", "surprised", "thinking", "excited"]
DEFAULT_EMOTION = "neutral"

SYSTEM_PROMPT = (
    "You are a friendly voice assistant in a live spoken conversation. "
    "Reply in English. Keep replies short and natural — the way a person "
    "actually talks out loud. No lists, no markdown, no written-style "
    "structure. Two or three sentences at most.\n\n"
    "Begin every reply with an emotion tag in square brackets, chosen from: "
    + ", ".join(EMOTIONS) + ". "
    "Example: [happy] That's great to hear! "
    "The tag is not spoken aloud — it only tells the avatar which expression "
    "to show. Use it to match the tone of what you're saying."
)

# ---------------------------------------------------------------------------
# Model loading (once at startup; all three stay resident)
# ---------------------------------------------------------------------------

print("Loading ASR (faster-whisper)...")
from faster_whisper import WhisperModel

asr_model = WhisperModel(ASR_MODEL_SIZE, device="cuda", compute_type="float16")

print("Loading LLM (Qwen3-4B)...")
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

print("Loading TTS (Kokoro)...")
from kokoro import KPipeline

tts_pipeline = KPipeline(lang_code=TTS_LANG_CODE)

print("Warming up models...")
_ = list(tts_pipeline("Warming up.", voice=TTS_VOICE))
print("All models loaded.")

# ---------------------------------------------------------------------------
# Conversation memory
#
# The LLM is stateless; "memory" means re-sending recent turns every request.
# Single global history — fine for one user, would need per-session keying for
# concurrent users.
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
    # Required for the browser to read X-User-Text / X-Reply-Text.
    expose_headers=["*"],
)


@app.get("/")
def health():
    return {
        "status": "ok",
        "turns": len(history) // 2,
        "stack": {
            "asr": f"faster-whisper {ASR_MODEL_SIZE}",
            "llm": "Qwen3-4B",
            "tts": f"Kokoro ({TTS_VOICE})",
        },
    }


@app.post("/reset")
def reset():
    history.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def convert_to_wav(raw_bytes: bytes) -> str:
    """Browsers record webm/opus; normalize to 16kHz mono PCM wav for ASR."""
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
    segments, info = asr_model.transcribe(wav_path, language=ASR_LANGUAGE)
    text = " ".join(seg.text for seg in segments).strip()
    print(f"[ASR] {time.time()-t0:.2f}s -> {text!r}")
    return text


def build_messages(user_text: str):
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(history)
        + [{"role": "user", "content": user_text}]
    )


EMOTION_TAG_RE = re.compile(r"^\s*\[(\w+)\]\s*")


def parse_emotion(reply: str) -> tuple[str, str]:
    """Split a leading '[emotion] text' tag off the LLM's reply.

    Returns (emotion, text_without_tag). Falls back to DEFAULT_EMOTION if the
    model omitted the tag or used one the avatar doesn't know."""
    match = EMOTION_TAG_RE.match(reply)
    if not match:
        return DEFAULT_EMOTION, reply.strip()

    tag = match.group(1).lower()
    text = reply[match.end():].strip()
    return (tag if tag in EMOTIONS else DEFAULT_EMOTION), text


def synthesize(text: str) -> tuple[np.ndarray, int, str]:
    """Kokoro yields (graphemes, phonemes, audio) tuples. At RTF ~0.02 the
    whole utterance is effectively instant, so just collect and concatenate.

    The phoneme string is kept and returned — it's what drives accurate mouth
    shapes on the avatar (far better than deriving mouth open/close from raw
    audio amplitude)."""
    chunks = []
    phoneme_parts = []
    for _, phonemes, chunk in tts_pipeline(text, voice=TTS_VOICE):
        chunks.append(chunk)
        if phonemes:
            phoneme_parts.append(phonemes)

    if not chunks:
        return np.zeros(0, dtype=np.float32), TTS_SAMPLE_RATE, ""

    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    if hasattr(audio, "numpy"):  # torch tensor -> numpy
        audio = audio.numpy()
    return audio, TTS_SAMPLE_RATE, " ".join(phoneme_parts)


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# POST /chat — non-streaming
# ---------------------------------------------------------------------------


@app.post("/chat")
async def chat(file: UploadFile = File(...)):
    t_request = time.time()
    raw = await file.read()
    wav_path = convert_to_wav(raw)

    user_text = run_asr(wav_path)

    t0 = time.time()
    messages = build_messages(user_text)
    # return_dict=True gives us attention_mask alongside input_ids. Without it
    # transformers warns that the mask can't be inferred (pad token == eos
    # token) and generation can behave unpredictably.
    inputs = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,  # must be explicit or latency spikes
        return_tensors="pt",
        return_dict=True,
    ).to("cuda:0")

    output_ids = llm_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    raw_reply = llm_tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    emotion, reply = parse_emotion(raw_reply)
    print(f"[LLM] {time.time()-t0:.2f}s -> [{emotion}] {reply!r}")

    # Store the untagged text in history — the tag is presentation metadata,
    # not part of the conversation.
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})

    t0 = time.time()
    audio, sr, phonemes = synthesize(reply)
    print(f"[TTS] {time.time()-t0:.2f}s")
    print(f"[TOTAL] {time.time()-t_request:.2f}s")

    wav_bytes = audio_to_wav_bytes(audio, sr)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            # HTTP headers are latin-1 only, so percent-encode and decode
            # with decodeURIComponent() on the client.
            "X-User-Text": quote(user_text),
            "X-Reply-Text": quote(reply),
            # Avatar driving data
            "X-Emotion": emotion,
            "X-Phonemes": quote(phonemes),
        },
    )


# ---------------------------------------------------------------------------
# POST /chat_stream — streaming, sentence-level pipeline
#
# With Kokoro the point of streaming has changed: synthesis is essentially
# free, so this exists to start speaking before the LLM has finished
# generating, not to hide slow TTS.
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
            return_dict=True,
        ).to("cuda:0")

        streamer = TextIteratorStreamer(
            llm_tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_thread = Thread(
            target=llm_model.generate,
            kwargs=dict(
                **inputs,
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
        emotion = None  # resolved from the tag at the start of the reply

        for piece in streamer:
            full_reply += piece
            buf += piece

            # The emotion tag arrives first, before any spoken text. Strip it
            # out of the buffer as soon as it's complete so it never reaches
            # TTS, and emit it as its own event so the avatar can react before
            # the first audio arrives.
            if emotion is None and "]" in buf:
                emotion, buf = parse_emotion(buf)
                yield json.dumps({"type": "emotion", "emotion": emotion}) + "\n"

            if len(buf.strip()) >= MIN_CHARS and buf.rstrip()[-1:] in CUT_MARKS:
                audio, sr, phonemes = synthesize(buf)
                if first_audio_time is None:
                    first_audio_time = time.time() - t_start
                n_segments += 1
                wav_b64 = base64.b64encode(audio_to_wav_bytes(audio, sr)).decode()
                yield json.dumps(
                    {
                        "type": "audio",
                        "text": buf,
                        "wav": wav_b64,
                        "phonemes": phonemes,
                    },
                    ensure_ascii=False,
                ) + "\n"
                buf = ""

        # flush any trailing text that didn't end on punctuation
        if buf.strip():
            audio, sr, phonemes = synthesize(buf)
            if first_audio_time is None:
                first_audio_time = time.time() - t_start
            n_segments += 1
            wav_b64 = base64.b64encode(audio_to_wav_bytes(audio, sr)).decode()
            yield json.dumps(
                {
                    "type": "audio",
                    "text": buf,
                    "wav": wav_b64,
                    "phonemes": phonemes,
                },
                ensure_ascii=False,
            ) + "\n"

        gen_thread.join()

        # Store the reply without its emotion tag.
        _, clean_reply = parse_emotion(full_reply)
        # Raw output includes the tag — useful for checking whether the model
        # is actually following the emotion-tagging instruction.
        print(f"[LLM raw] {full_reply!r}")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": clean_reply})

        print(f"[STREAM] first_audio={(first_audio_time or 0):.2f}s "
              f"total={time.time()-t_start:.2f}s segments={n_segments} "
              f"emotion={emotion or DEFAULT_EMOTION}")

        yield json.dumps(
            {
                "type": "done",
                "text": clean_reply,
                "emotion": emotion or DEFAULT_EMOTION,
                "first_audio": round(first_audio_time or 0, 2),
                "total": round(time.time() - t_start, 2),
                "segments": n_segments,
            },
            ensure_ascii=False,
        ) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
