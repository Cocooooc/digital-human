"""
Minimal verification of the English stack before wiring it into the server.

Checks, in order:
  1. faster-whisper loads and transcribes
  2. Kokoro loads, and whether its pipeline really is a streaming generator
  3. Real speed (RTF) of both, measured after warm-up

Run:
    python test_english_stack.py
"""
import time
import numpy as np
import soundfile as sf

TEST_AUDIO = "/root/test10s.wav"  # the Chinese sample; fine for a smoke test
SHORT_TEXT = "Hello! Nice to meet you."
LONG_TEXT = (
    "Hello! Nice to meet you. The weather is quite lovely today. "
    "I have been working on some interesting projects lately, "
    "and I would love to hear what you have been up to as well."
)


def test_asr():
    print("=" * 60)
    print("1. faster-whisper (ASR)")
    print("=" * 60)

    from faster_whisper import WhisperModel

    t0 = time.time()
    # "small" is the sweet spot for real-time English; bump to "medium" if
    # accuracy matters more than latency.
    model = WhisperModel("small", device="cuda", compute_type="float16")
    print(f"Load time: {time.time()-t0:.2f}s")

    # warm-up
    _ = list(model.transcribe(TEST_AUDIO, language="en")[0])

    t0 = time.time()
    segments, info = model.transcribe(TEST_AUDIO, language="en")
    text = " ".join(seg.text for seg in segments)
    elapsed = time.time() - t0

    print(f"Transcribe time: {elapsed:.3f}s (audio {info.duration:.2f}s, "
          f"RTF {elapsed/info.duration:.4f})")
    print(f"Text: {text!r}")
    return model


def test_tts():
    print()
    print("=" * 60)
    print("2. Kokoro (TTS)")
    print("=" * 60)

    from kokoro import KPipeline

    t0 = time.time()
    pipeline = KPipeline(lang_code="a")  # 'a' = American English
    print(f"Load time: {time.time()-t0:.2f}s")

    # Is it really a generator that yields progressively, or does it block
    # until everything is done? This is the whole reason we picked Kokoro,
    # so verify it rather than assume.
    print("\nChecking whether generation is truly streaming...")
    t_start = time.time()
    chunk_times = []
    chunks = []
    for i, (_, _, chunk) in enumerate(pipeline(LONG_TEXT, voice="af_bella")):
        arrival = time.time() - t_start
        chunk_times.append(arrival)
        chunks.append(chunk)
        print(f"  chunk {i+1} arrived at {arrival:.3f}s "
              f"({len(chunk)/24000:.2f}s of audio)")

    total = time.time() - t_start
    audio = np.concatenate(chunks)
    duration = len(audio) / 24000

    print(f"\nTotal: {total:.3f}s for {duration:.2f}s of audio, "
          f"RTF {total/duration:.3f}")
    if chunk_times:
        print(f"*** TIME TO FIRST AUDIO: {chunk_times[0]:.3f}s ***")
    if len(chunk_times) > 1:
        print("Streaming confirmed: chunks arrived progressively, so audio can "
              "start playing before generation finishes.")
    else:
        print("Only one chunk — for this text length it generated in one go.")

    sf.write("/root/test_kokoro_output.wav", audio, 24000)
    print("Saved to /root/test_kokoro_output.wav")

    # Warm speed on a short utterance (typical of a chat reply)
    print("\nWarm speed on a short reply (3 runs):")
    _ = list(pipeline(SHORT_TEXT, voice="af_bella"))  # warm-up
    for i in range(3):
        t0 = time.time()
        chunks = [c for _, _, c in pipeline(SHORT_TEXT, voice="af_bella")]
        elapsed = time.time() - t0
        audio = np.concatenate(chunks)
        duration = len(audio) / 24000
        print(f"  run {i+1}: {elapsed:.3f}s for {duration:.2f}s audio, "
              f"RTF {elapsed/duration:.3f}")

    return pipeline


def list_voices():
    print()
    print("=" * 60)
    print("3. Available voices")
    print("=" * 60)
    print("Common American English voices:")
    print("  Female: af_bella, af_nicole, af_sarah, af_sky")
    print("  Male:   am_adam, am_michael")
    print("British English: bf_emma, bf_isabella, bm_george, bm_lewis")
    print("(Pick one for the server's TTS_VOICE setting.)")


if __name__ == "__main__":
    test_asr()
    test_tts()
    list_voices()

    print()
    print("=" * 60)
    print("If both RTFs above are well under 1.0, the English stack is a big")
    print("win over the current one (Qwen3-TTS was RTF ~3.9 plus ~6s overhead).")
    print("=" * 60)
