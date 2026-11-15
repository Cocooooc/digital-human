"""
Chinese clips for the robot — 30 s to 1 min each, as the teacher asked.

text -> Kokoro (Mandarin voice) -> EMAGE -> mouth_post (lip-sync fixes)
     -> motion_post (idle lead-in/out, seam-free) -> gaze -> npz
Each clip is also saved raw (EMAGE as generated) so the effect of the
post-processing can be checked side by side.

Two stages (Kokoro in base env, EMAGE in pantomatrix):
    python make_chinese_clips.py tts    --out /root/zh_clips
    python make_chinese_clips.py motion --out /root/zh_clips
make_chinese_clips.sh runs everything including render + robot export.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# ~420-520 characters each -> roughly 45-60 s (Kokoro Mandarin runs ~9-10 chars/s;
# the first version at ~200 chars came out at only ~20 s).
# Varied on purpose: plain narration, questions, pauses ("……"), a drawn-out
# "嗯——", numbers and lists — the cases where lip-sync went wrong in English.
TEXTS = [
    ("zh_01_intro",
     "大家好，我是一个数字人。很高兴认识你！我的声音、手势，还有说话时的眼神，都是由人工智能根据语音实时生成的。"
     "你对我说一句话，我会先把声音转成文字，再用大语言模型想好怎么回答，然后合成语音，"
     "同时生成和这段语音配合的身体动作和面部表情。整个过程，从你说完到我开口，大约只需要一点七秒。"
     "……你可能会好奇，我的动作是怎么来的？其实，模型学习了很多真人说话时的动作数据，"
     "知道人在强调的时候会抬手，在思考的时候会停顿，在提问的时候语气会上扬。"
     "不过，模型也有不完美的地方。比如，它以前不会转动眼睛，说话时总是盯着一个地方；"
     "停下来的时候，嘴巴也合不上。现在，我们用信号处理的方法，把这些问题一个一个修好了。"
     "你看，我现在说完一句话，嘴巴会自然地闭上；拖长声音的时候，嘴型也会保持稳定。"
     "如果你有什么问题，随时可以问我。我会尽量用简短、清楚的方式回答你。谢谢！"),
    ("zh_02_weather",
     "今天天气不错，早上有一点凉，到了中午就暖和起来了。嗯——我觉得这样的天气，最适合出去走一走。"
     "你可以去公园散散步，看看树叶慢慢变黄，或者找一家安静的咖啡店，点一杯热拿铁，坐下来看看书。"
     "……不过，天气预报说，晚上可能会下雨，气温也会降到五度左右，出门的时候最好带一把伞，再多穿一件外套。"
     "对了，如果你周末有空，也可以约上几个朋友，去郊外爬爬山，呼吸一下新鲜空气。"
     "山上的风景这个季节特别好，红色、黄色、橙色的树叶，一层一层的，像一幅画一样。"
     "爬到山顶的时候，你会觉得，一周的疲劳一下子就没有了。下山以后，还可以去附近的小镇，"
     "吃一顿热乎乎的晚饭。你觉得怎么样？有没有想去的地方？如果你还没想好，我也可以帮你推荐几个。"),
    ("zh_03_explain",
     "很多人会问，机器人是怎么听懂我们说话的？其实可以分成三步。第一步，语音识别，"
     "把声音变成文字。第二步，理解和回答，由大语言模型根据上下文，决定该说什么。"
     "第三步，语音合成，把文字重新变成声音。在这三步之外，我们还加了第四步：动作生成。"
     "模型会根据声音的节奏和语气，生成手势、头部动作和口型，让回答看起来更自然，"
     "而不只是一个会说话的声音。……那么，这四步里，哪一步最慢呢？我们做了测试，"
     "答案是第二步，也就是大语言模型。它大概占了整个时间的百分之六十，而且回答越长，花的时间越多。"
     "相比之下，动作生成非常快，一分钟的语音，只需要一秒多就能生成全部动作。"
     "另外，我们还发现，模型生成的动作有时候不够自然，比如停顿的时候嘴巴张着，拖音的时候嘴巴乱动。"
     "我们没有重新训练模型，而是用工程的方法，分析声音的大小、音高和节奏，再去调整口型。"
     "这样做的好处是，不管是中文还是英文，都可以直接使用。"),
    ("zh_04_story",
     "我给你讲一个小故事吧。从前，有一个小女孩，她特别喜欢看星星。每天晚上，她都会爬到屋顶上，"
     "一颗一颗地数。有一天，她问妈妈：天上到底有多少颗星星呀？妈妈笑着说：……数不清的。"
     "小女孩想了想，说：那我就一直数下去，总有一天能数完！第二天晚上，她又爬上了屋顶。"
     "可是，天上飘来了很多云，一颗星星也看不见。她有点难过，坐在那里，等啊等。"
     "等了很久很久，云终于散开了，星星一颗一颗地冒了出来，比以前还要亮。"
     "那一刻，她突然明白了：星星一直都在，只是有时候被云挡住了而已。"
     "后来，她长大了，读了很多很多书，真的成了一名天文学家。"
     "她用望远镜看到了比小时候多得多的星星，可是她知道，还是数不完。"
     "有人问她：你小时候的愿望，实现了吗？她笑着说：嗯——没有数完，但是我一直在数。"
     "你看，很多了不起的事情，都是从一个简单的问题开始的。"),
]

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["tts", "motion"])
ap.add_argument("--engine", default="edge", choices=["edge", "kokoro"],
                help="edge: Microsoft neural voices via edge-tts (natural Mandarin prosody, "
                     "needs internet); kokoro: offline, but its Mandarin sounds flat")
ap.add_argument("--edge-voice", default="zh-CN-XiaoxiaoNeural",
                help="e.g. zh-CN-XiaoxiaoNeural (f), zh-CN-XiaoyiNeural (f), zh-CN-YunxiNeural (m), "
                     "zh-CN-YunjianNeural (m)")
ap.add_argument("--out", default="/root/zh_clips")
ap.add_argument("--voice", default="zf_xiaoxiao",
                help="Kokoro Mandarin voices: zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi "
                     "zm_yunjian zm_yunxi zm_yunxia zm_yunyang")
ap.add_argument("--scale", type=float, default=0.85, help="gesture size (1 = as generated)")
ap.add_argument("--speed", type=float, default=0.9, help="speaking rate (1 = the voice's default)")
ap.add_argument("--repo", default="/root/PantoMatrix")
ap.add_argument("--weights", default="/root/emage_weights")
args = ap.parse_args()
out = Path(args.out)
(out / "wav").mkdir(parents=True, exist_ok=True)
TTS_SR = 24000

if args.stage == "tts" and args.engine == "edge":
    # Kokoro's Mandarin came out flat and mechanical (tones and phrasing),
    # which also matters for the motion: EMAGE takes its rhythm from the
    # prosody, so flat speech gives flat gestures and lip movement.
    import asyncio
    import subprocess
    import edge_tts
    rate = f"{round((args.speed - 1) * 100):+d}%"

    async def say(text, mp3):
        await edge_tts.Communicate(text, args.edge_voice, rate=rate).save(str(mp3))

    for cid, text in TEXTS:
        mp3 = out / "wav" / f"{cid}.mp3"
        wav = out / "wav" / f"{cid}.wav"
        asyncio.run(say(text, mp3))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1",
                        "-ar", str(TTS_SR), str(wav)], check=True)
        import soundfile as sf
        print(f"  {cid:<14} {sf.info(str(wav)).duration:5.1f}s  {len(text)} chars  {args.edge_voice} {rate}")
    raise SystemExit(0)

if args.stage == "tts":
    import soundfile as sf
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="z")
    for cid, text in TEXTS:
        chunks = [np.asarray(a, np.float32) for _, _, a in pipe(text, voice=args.voice, speed=args.speed)]
        gap = np.zeros(int(0.15 * TTS_SR), np.float32)
        a = np.concatenate([x for c in chunks for x in (c, gap)][:-1])
        sf.write(out / "wav" / f"{cid}.wav", a, TTS_SR)
        print(f"  {cid:<14} {len(a) / TTS_SR:5.1f}s  {len(text)} chars")
    raise SystemExit(0)

# --- motion -------------------------------------------------------------------
import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

import motion_post as mp  # noqa: E402
from mouth_post import fix_mouth, analyse_audio  # noqa: E402


def jaw_stats(poses, A):
    """Jaw-angle measures that need no mesh: how open the jaw is in pauses,
    and the share of speech in which it barely moves."""
    th = np.degrees(np.linalg.norm(poses[:, 66:69], axis=1))
    sp = A["speech"]
    quiet = ~sp
    w = 15
    stds = [(th[i:i + w].std(), i) for i in range(0, len(th) - w + 1, 5) if sp[i:i + w].mean() > 0.8]
    return th, quiet, stds


def compare(raw, fixed, A):
    th_r, quiet, s_r = jaw_stats(raw, A)
    th_f, _, s_f = jaw_stats(fixed, A)
    typical = np.median([x for x, _ in s_r]) if s_r else 0.0
    frozen = lambda s: 100.0 * np.mean([x < 0.4 * typical for x, _ in s]) if s else 0.0
    q = quiet.copy()
    q[:3] = False                     # first frames: model and fix both start closed-ish
    return {"pause_jaw_deg": [round(float(th_r[q].mean()), 1) if q.any() else None,
                              round(float(th_f[q].mean()), 1) if q.any() else None],
            "speech_jaw_deg": [round(float(th_r[A["speech"]].mean()), 1), round(float(th_f[A["speech"]].mean()), 1)],
            "frozen_speech_pct": [round(frozen(s_r), 1), round(frozen(s_f), 1)]}
from evaluate_shortcomings import load_emage, save_emage_npz, make_gaze  # noqa: E402

FPS = mp.FPS
print("loading EMAGE…")
gen, sr = load_emage(Path(args.repo), Path(args.weights))
example = Path(args.repo) / "examples" / "audio" / "2_scott_0_103_103_28s.wav"
rest = mp.rest_pose_from(gen(librosa.load(str(example), sr=sr)[0])[0])

summary = {}
for i, (cid, text) in enumerate(TEXTS):
    wav = out / "wav" / f"{cid}.wav"
    if not wav.exists():
        raise SystemExit(f"{wav} missing — run the tts stage first (base env)")
    a16 = librosa.load(str(wav), sr=sr)[0]
    poses, expr, _ = gen(a16)
    print(f"  {cid}: {len(a16) / sr:.1f}s -> {len(poses)} frames")

    # raw, for comparison: EMAGE exactly as generated
    save_emage_npz(out / f"{cid}_raw.npz", poses, expr, np.zeros((len(poses), 3)))

    p2, e2, minfo = fix_mouth(poses, expr, a16, sr)
    cmp = compare(poses, p2, analyse_audio(a16, sr, len(poses)))
    print(f"    jaw in pauses {cmp['pause_jaw_deg'][0]} -> {cmp['pause_jaw_deg'][1]} deg   "
          f"frozen speech {cmp['frozen_speech_pct'][0]}% -> {cmp['frozen_speech_pct'][1]}%")
    s = mp.Session(rest, scale=args.scale, seed=i)
    lead = s.idle(int(1.0 * FPS))
    talk = s.speak(p2, e2)
    tail = s.idle(int(1.5 * FPS))
    P = np.concatenate([lead[0], talk[0], tail[0]])
    E = np.concatenate([lead[1], talk[1], tail[1]])
    gaze, _, _ = make_gaze(P, seed=i, look_ratio=0.8, include_root=True)
    P[:, 69:72] = gaze
    P[:, 72:75] = gaze
    save_emage_npz(out / f"{cid}.npz", P, E, np.zeros((len(P), 3)))

    # soundtrack aligned to the final frames (silence for lead/tail)
    a24 = librosa.load(str(wav), sr=TTS_SR)[0]
    track = np.zeros(int(round(len(P) / FPS * TTS_SR)), np.float32)
    st = int(round(len(lead[0]) / FPS * TTS_SR))
    seg = a24[:max(0, len(track) - st)]
    track[st:st + len(seg)] = seg
    sf.write(out / f"{cid}_track.wav", track, TTS_SR)
    # raw has no lead/tail: its own audio
    sf.write(out / f"{cid}_raw_track.wav", a24, TTS_SR)

    summary[cid] = {"text": text, "seconds": round(len(P) / FPS, 1), "frames": len(P),
                    "mouth_post": {k: v for k, v in minfo.items() if k != "jaw_axis"},
                    "before_after": cmp}

(out / "clips.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n  == Chinese clips: EMAGE raw -> after mouth_post ==")
print("  clip            secs   jaw in pauses (deg)   frozen speech %")
for cid, v in summary.items():
    b = v["before_after"]
    print(f"  {cid:<14} {v['seconds']:5.1f}   {b['pause_jaw_deg'][0]:>6} -> {b['pause_jaw_deg'][1]:<6}"
          f"      {b['frozen_speech_pct'][0]:>5} -> {b['frozen_speech_pct'][1]}")
print(f"\n  wrote {out}/<clip>.npz (processed), <clip>_raw.npz, *_track.wav, clips.json")
