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

# ~180-260 characters each -> roughly 40-60 s at Kokoro's Mandarin pace.
# Varied on purpose: plain narration, questions, pauses ("……"), a drawn-out
# "嗯——", numbers and lists — the cases where lip-sync went wrong in English.
TEXTS = [
    ("zh_01_intro",
     "大家好，我是一个数字人。我的声音、手势，还有说话时的眼神，都是由人工智能根据语音实时生成的。"
     "你对我说一句话，我会先把声音转成文字，再用大语言模型想好怎么回答，然后合成语音，"
     "同时生成和这段语音配合的身体动作和面部表情。整个过程大约一点七秒。"
     "如果你有什么问题，随时可以问我，我会尽量用简短、清楚的方式回答你。"),
    ("zh_02_weather",
     "今天天气不错，早上有一点凉，到了中午就暖和起来了。嗯——我觉得这样的天气，最适合出去走一走。"
     "你可以去公园散散步，或者找一家安静的咖啡店，坐下来看看书。……不过，晚上可能会下雨，"
     "出门的时候最好带一把伞。对了，如果你周末有空，也可以约上朋友去郊外爬山，呼吸一下新鲜空气，"
     "放松一下心情。"),
    ("zh_03_explain",
     "很多人会问，机器人是怎么听懂我们说话的？其实可以分成三步。第一步，语音识别，"
     "把声音变成文字。第二步，理解和回答，由大语言模型根据上下文，决定该说什么。"
     "第三步，语音合成，把文字重新变成声音。在这三步之外，我们还加了第四步：动作生成。"
     "模型会根据声音的节奏和语气，生成手势、头部动作和口型，让回答看起来更自然，"
     "而不只是一个会说话的声音。"),
    ("zh_04_story",
     "我给你讲一个小故事吧。从前，有一个小女孩，她特别喜欢看星星。每天晚上，她都会爬到屋顶上，"
     "一颗一颗地数。有一天，她问妈妈：天上到底有多少颗星星呀？妈妈笑着说：……数不清的。"
     "小女孩想了想，说：那我就一直数下去，总有一天能数完！后来，她长大了，真的成了一名天文学家。"
     "你看，很多了不起的事情，都是从一个简单的问题开始的。"),
]

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["tts", "motion"])
ap.add_argument("--out", default="/root/zh_clips")
ap.add_argument("--voice", default="zf_xiaoxiao",
                help="Kokoro Mandarin voices: zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi "
                     "zm_yunjian zm_yunxi zm_yunxia zm_yunyang")
ap.add_argument("--scale", type=float, default=0.85, help="gesture size (1 = as generated)")
ap.add_argument("--repo", default="/root/PantoMatrix")
ap.add_argument("--weights", default="/root/emage_weights")
args = ap.parse_args()
out = Path(args.out)
(out / "wav").mkdir(parents=True, exist_ok=True)
TTS_SR = 24000

if args.stage == "tts":
    import soundfile as sf
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="z")
    for cid, text in TEXTS:
        chunks = [np.asarray(a, np.float32) for _, _, a in pipe(text, voice=args.voice)]
        gap = np.zeros(int(0.15 * TTS_SR), np.float32)
        a = np.concatenate([x for c in chunks for x in (c, gap)][:-1])
        sf.write(out / "wav" / f"{cid}.wav", a, TTS_SR)
        print(f"  {cid:<14} {len(a) / TTS_SR:5.1f}s  {len(text)} chars")
    raise SystemExit(0)

# --- motion -------------------------------------------------------------------
import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

import motion_post as mp  # noqa: E402
from mouth_post import fix_mouth  # noqa: E402
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
                    "mouth_post": {k: v for k, v in minfo.items() if k != "jaw_axis"}}

(out / "clips.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\n  wrote {out}/<clip>.npz (processed), <clip>_raw.npz, *_track.wav, clips.json")
