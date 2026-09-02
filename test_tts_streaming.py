"""
Test the true streaming capability of Qwen3-TTS: time-to-first-audio latency.

Context:
RTF (real-time factor) ~1.5, recorded in earlier benchmarks, measures how long
the *entire* utterance takes to synthesize relative to its length. That is a
different question from "how soon does the first audio chunk become available
to the user" — the latter is what actually matters for perceived responsiveness.

This script:
1. Inspects the qwen_tts library to see whether it exposes a true streaming /
   chunked-callback interface (as opposed to a synchronous call that blocks
   until the whole utterance is generated).
2. Measures the best latency achievable with whatever interface is available.

Usage:
    python test_tts_streaming.py
"""
import time
import inspect
import numpy as np
import soundfile as sf

MODEL_PATH = "/root/qwen3-tts"
REF_AUDIO = "/root/test10s.wav"
# Source: FunASR official sample audio
# (https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav),
# transcribed with our own ASR model.
REF_TEXT = "欢迎大家来体验达摩院推出的语音识别模型。"
TEST_TEXT = "你好，很高兴认识你。今天天气不错，我们聊聊你最近在忙什么呢？"


def inspect_api():
    """Step 1: print the real signature/docstring of generate_voice_clone,
    to check whether it supports streaming callbacks or is a generator."""
    import qwen_tts  # noqa
    print("=" * 60)
    print("Available names in qwen_tts:")
    print([x for x in dir(qwen_tts) if not x.startswith("_")])
    print("=" * 60)

    candidates = ["Qwen3TTSModel", "Qwen3TTS", "QwenTTS", "TTSModel", "Model"]
    tts_cls = None
    for name in candidates:
        if hasattr(qwen_tts, name):
            tts_cls = getattr(qwen_tts, name)
            print(f"Found class: {name}")
            break

    if tts_cls is None:
        print("Class not found under the expected names. Run manually:")
        print("   python -c 'import qwen_tts; help(qwen_tts)'")
        return None

    print(f"\nMethods/attrs on {tts_cls.__name__}:")
    print([x for x in dir(tts_cls) if not x.startswith("_")])

    for method_name in ["from_pretrained", "create_voice_clone_prompt", "generate_voice_clone"]:
        if hasattr(tts_cls, method_name):
            method = getattr(tts_cls, method_name)
            try:
                sig = inspect.signature(method)
                print(f"\n{method_name} signature: {sig}")
            except (ValueError, TypeError):
                print(f"\n{method_name}: (signature unavailable, likely a C/pybind method)")
            if method.__doc__:
                print(f"docstring:\n{method.__doc__}")
        else:
            print(f"\n⚠️ {method_name} NOT FOUND on {tts_cls.__name__}")

    return tts_cls


def measure_first_audio_latency(tts, prompt):
    """Core measurement. If the call is synchronous and blocking, we can only
    measure total wall-clock time and chunk count — a real first-chunk
    latency requires a streaming/callback API, which step 1 will reveal."""
    print("\n" + "=" * 60)
    print("Measuring...")

    t_start = time.time()
    audio_chunks, sr = tts.generate_voice_clone(
        text=TEST_TEXT,
        voice_clone_prompt=prompt,
        non_streaming_mode=False,  # explicitly request the streaming code path
    )
    t_end = time.time()

    total_time = t_end - t_start
    n_chunks = len(audio_chunks) if isinstance(audio_chunks, list) else 1

    if isinstance(audio_chunks, list):
        total_samples = sum(len(c) for c in audio_chunks)
    else:
        total_samples = len(audio_chunks)
    audio_duration = total_samples / sr

    print(f"Total time: {total_time:.3f}s")
    print(f"Audio duration: {audio_duration:.3f}s")
    print(f"RTF: {total_time / audio_duration:.3f}")
    print(f"Returned chunk count: {n_chunks}")

    if n_chunks > 1:
        print("Note: the call is synchronous — even with multiple chunks, ")
        print("all of them are generated before the function returns.")
        print("A true 'speak while generating' effect requires a callback/")
        print("generator API, or falling back to the sentence-level pipeline.")

    if isinstance(audio_chunks, list):
        full_audio = np.concatenate(audio_chunks)
    else:
        full_audio = audio_chunks
    sf.write("/root/test_streaming_output.wav", full_audio, sr)
    print(f"\nSaved to /root/test_streaming_output.wav")

    return total_time, audio_duration


if __name__ == "__main__":
    tts_cls = inspect_api()

    if tts_cls is None:
        print("\nPlease share the help() output above so the script can be adjusted.")
        exit(1)

    try:
        print("\nLoading model...")
        t0 = time.time()
        # from_pretrained() loads onto CPU by default (no `device` kwarg).
        # The real nn.Module is at `.model`; `.device` is a plain writable
        # attribute that other methods read to decide where to build
        # tensors, so both need to be set or you get a device-mismatch
        # crash inside generate_voice_clone.
        tts = tts_cls.from_pretrained(MODEL_PATH)
        tts.model = tts.model.to("cuda:0")
        tts.device = "cuda:0"
        print(f"Load time: {time.time()-t0:.2f}s, device: {tts.device}")
    except Exception as e:
        print(f"\n❌ from_pretrained failed: {e}")
        print("The signature printed above should show the real expected args.")
        exit(1)

    try:
        print("\nPrecomputing voice clone prompt...")
        prompt = tts.create_voice_clone_prompt(REF_AUDIO, ref_text=REF_TEXT)
    except Exception as e:
        print(f"\n❌ create_voice_clone_prompt failed: {e}")
        exit(1)

    try:
        print("\nWarming up...")
        _ = tts.generate_voice_clone(
            text="你好",
            voice_clone_prompt=prompt,
            non_streaming_mode=False,
        )
    except Exception as e:
        print(f"\n❌ generate_voice_clone failed: {e}")
        exit(1)

    results = []
    for i in range(3):
        print(f"\n--- Run {i+1} ---")
        total_time, audio_duration = measure_first_audio_latency(tts, prompt)
        results.append(total_time)

    print("\n" + "=" * 60)
    print(f"Average over 3 runs: {sum(results)/len(results):.3f}s")
