"""
Latency benchmark for the digital-human pipeline.

The teacher asked for per-stage timings. The server already logged them, but a
log line answers "how long did that turn take", and latency depends on what was
said and how much the model chose to say back — so one turn tells you almost
nothing. This drives a fixed set of utterances through the real HTTP endpoints
and reports the distribution, with P95 alongside the median: a system with a
0.9s median and a 4s P95 feels broken once every twenty turns, and the median
alone hides that completely.

It measures from outside the server, over real HTTP, because that is where the
user sits. The server's own /stats is read as well, so per-stage numbers and
end-to-end numbers can be checked against each other — a gap between them is
upload, JSON encoding, or something else nobody was accounting for.

Run (base env, where the conversation server lives):
    conda activate base
    python /root/benchmark_latency.py --turns 15

    # include motion generation if emage_server is up
    python /root/benchmark_latency.py --turns 15 --emage http://127.0.0.1:15313
"""
import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--server", default="http://127.0.0.1:15343")
ap.add_argument("--emage", default="", help="motion service URL; skipped if empty")
ap.add_argument("--turns", type=int, default=15)
ap.add_argument("--voice", default="am_michael", help="voice for the synthetic user")
ap.add_argument("--out", default="/root/latency_report")
ap.add_argument("--stream", action="store_true",
                help="use /chat_stream and measure time-to-first-audio")
args = ap.parse_args()

# Utterances of deliberately different lengths. Latency is not one number: a
# three-word question and a two-sentence one exercise ASR and the LLM's context
# very differently, and a benchmark of one length would report a fiction.
PROMPTS = [
    "Hi.",
    "How are you?",
    "What's the weather like?",
    "Can you hear me clearly?",
    "Tell me something interesting.",
    "What did we just talk about?",
    "I'm working on a digital human project for my professor.",
    "Could you explain what you just said in simpler terms?",
    "What's your favorite thing about working with people?",
    "I've been studying computer science at Cornell for three years now.",
    "Do you think speech recognition has gotten good enough for real conversation?",
    "If you had to describe yourself in one sentence, what would you say?",
]


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# ---------------------------------------------------------------------------
section("1. Checking the services")
# ---------------------------------------------------------------------------
try:
    health = requests.get(args.server + "/", timeout=10).json()
    print(f"  conversation: {health.get('stack', health)}")
except Exception as e:
    print(f"  ❌ can't reach {args.server}: {e}")
    print("     start it with: python -m uvicorn server_english:app --host 0.0.0.0 --port 15343")
    sys.exit(1)

use_emage = bool(args.emage)
if use_emage:
    try:
        print(f"  motion: {requests.get(args.emage + '/', timeout=10).json().get('model')}")
    except Exception as e:
        print(f"  ⚠️  motion service unreachable ({e}) — continuing without it")
        use_emage = False

requests.post(args.server + "/stats/reset", timeout=10)
requests.post(args.server + "/reset", timeout=10)
print("  stats and history cleared")


# ---------------------------------------------------------------------------
section("2. Synthesising the user's side")
# ---------------------------------------------------------------------------
# The benchmark needs real audio to feed ASR. Using TTS to produce it is
# slightly circular, but only for the input: ASR still has to do full work on a
# waveform, and every run gets the identical audio, which is what makes runs
# comparable at all. Recording by hand would make each run different.
from kokoro import KPipeline  # noqa: E402

pipe = KPipeline(lang_code="a")
clips = []
for text in PROMPTS[:args.turns] * (args.turns // len(PROMPTS) + 1):
    if len(clips) >= args.turns:
        break
    chunks = [c for _, _, c in pipe(text, voice=args.voice)]
    audio = np.concatenate([np.asarray(c) for c in chunks])
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    clips.append((text, buf.getvalue(), len(audio) / 24000))
    print(f"  {len(clips):>2}. {len(audio)/24000:4.1f}s  {text[:52]}")


# ---------------------------------------------------------------------------
section("3. Running turns")
# ---------------------------------------------------------------------------
rows = []
endpoint = "/chat_stream" if args.stream else "/chat"

for i, (text, wav, dur) in enumerate(clips, 1):
    t0 = time.time()
    r = requests.post(args.server + endpoint,
                      files={"file": ("turn.wav", wav, "audio/wav")}, timeout=300)
    row = {"i": i, "prompt": text, "input_seconds": round(dur, 2)}

    if args.stream:
        first_audio = None
        reply_wavs = []
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("type") == "audio" and first_audio is None:
                first_audio = time.time() - t0
            if ev.get("type") == "done":
                row["server_total"] = ev.get("total")
        row["first_audio"] = round(first_audio or 0, 3)
        reply_audio = None
    else:
        reply_audio = r.content
        row["reply_bytes"] = len(reply_audio)

    row["end_to_end"] = round(time.time() - t0, 3)

    if use_emage and reply_audio:
        t1 = time.time()
        m = requests.post(args.emage + "/motion",
                          files={"file": ("reply.wav", reply_audio, "audio/wav")},
                          timeout=300).json()
        row["motion"] = round(time.time() - t1, 3)
        row["motion_infer"] = m["timing"]["infer"]
        row["motion_rtf"] = round(m["timing"]["infer"] / max(m["audio_duration"], 1e-6), 4)
        row["reply_seconds"] = m["audio_duration"]

    rows.append(row)
    extra = f" motion {row.get('motion', '-')}" if use_emage else ""
    print(f"  {i:>2}/{len(clips)}  in {dur:4.1f}s → e2e {row['end_to_end']:5.2f}s{extra}")


# ---------------------------------------------------------------------------
section("4. Per-stage, from the server")
# ---------------------------------------------------------------------------
server_stats = requests.get(args.server + "/stats", timeout=10).json()

LABELS = {
    "asr": "ASR (faster-whisper)", "llm": "LLM (Qwen3-4B)",
    "tts": "TTS (Kokoro)", "total": "server total",
    "tts_rtf": "TTS RTF", "speech_seconds": "reply length (s)",
    "llm_tokens": "LLM tokens out",
    "stream_first_audio": "time to first audio", "stream_total": "stream total",
}
print(f"  {'stage':<24}{'n':>4}{'p50':>9}{'p95':>9}{'max':>9}{'mean':>9}")
print("  " + "-" * 64)
for k, v in server_stats.items():
    if k.startswith("_") or not v:
        continue
    print(f"  {LABELS.get(k, k):<24}{v['count']:>4}{v['p50']:>9.3f}"
          f"{v['p95']:>9.3f}{v['max']:>9.3f}{v['mean']:>9.3f}")

note = server_stats.get("_note")
if note:
    print(f"\n  stages add to {note['stages_p50_sum']}s, server total {note['total_p50']}s"
          f"  → {note['unaccounted']}s elsewhere (audio decode, wav encode)")


# ---------------------------------------------------------------------------
section("5. End to end, from the client")
# ---------------------------------------------------------------------------
def stats_of(key):
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    idx95 = max(0, min(len(vals_sorted) - 1, int(round(0.95 * len(vals_sorted) + 0.5)) - 1))
    return {"count": len(vals), "p50": round(statistics.median(vals), 3),
            "p95": round(vals_sorted[idx95], 3), "max": round(max(vals), 3),
            "mean": round(statistics.mean(vals), 3)}


client = {k: stats_of(k) for k in
          ("end_to_end", "first_audio", "motion", "motion_infer", "motion_rtf")}
print(f"  {'measure':<24}{'n':>4}{'p50':>9}{'p95':>9}{'max':>9}")
print("  " + "-" * 55)
for k, v in client.items():
    if v:
        print(f"  {k:<24}{v['count']:>4}{v['p50']:>9.3f}{v['p95']:>9.3f}{v['max']:>9.3f}")

e2e = client["end_to_end"]
motion = client.get("motion_infer")
full = e2e["p50"] + (motion["p50"] if motion else 0)


# ---------------------------------------------------------------------------
section("6. Report")
# ---------------------------------------------------------------------------
out = Path(args.out)
payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "endpoint": endpoint, "turns": len(rows),
           "server_stages": server_stats, "client": client, "rows": rows}
out.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

md = [f"# Digital Human — Latency Report",
      f"", f"Generated {payload['generated']} · {len(rows)} turns · `{endpoint}`", f"",
      f"## Per-stage (server, P50 / P95 in seconds)", f"",
      f"| stage | n | P50 | P95 | max | mean |", f"|---|---|---|---|---|---|"]
for k, v in server_stats.items():
    if k.startswith("_") or not v:
        continue
    md.append(f"| {LABELS.get(k, k)} | {v['count']} | {v['p50']} | {v['p95']} "
              f"| {v['max']} | {v['mean']} |")
md += ["", "## End to end (client, over HTTP)", "",
       "| measure | n | P50 | P95 | max |", "|---|---|---|---|---|"]
for k, v in client.items():
    if v:
        md.append(f"| {k} | {v['count']} | {v['p50']} | {v['p95']} | {v['max']} |")
md += ["", "## Per turn", "",
       "| # | input (s) | end-to-end (s) | " + ("motion (s) | " if use_emage else "") + "utterance |",
       "|---|---|---|" + ("---|" if use_emage else "") + "---|"]
for r in rows:
    md.append(f"| {r['i']} | {r['input_seconds']} | {r['end_to_end']} | "
              + (f"{r.get('motion_infer','-')} | " if use_emage else "")
              + f"{r['prompt']} |")
out.with_suffix(".md").write_text("\n".join(md))

print(f"  {out.with_suffix('.json')}")
print(f"  {out.with_suffix('.md')}")
print(f"""
  Median turn: {e2e['p50']}s conversation""" + (f" + {motion['p50']}s motion = {full:.2f}s" if motion else "") + f"""
  P95:         {e2e['p95']}s  ← the number that decides whether it feels reliable

  Human conversation leaves 200-500ms between turns. Under ~1s reads as
  responsive; the P95 is what people remember, because one slow turn in
  twenty is noticed far more than nineteen fast ones.
""")
