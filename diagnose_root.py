"""
Root-cause diagnosis for the five chosen problems, and the data-derived pieces
the fixes will need. One run on the GPU box; everything measured on held-out
BEAT2 clips (the model never trained on them).

  1. blinks          data blink rate, duration, intervals, where they fall
                     (pause vs speech); the blink's FLAME expression shape
  2. mouth too wide  lip gap data vs model; is it speaker style? (same audio,
                     speaker embedding swept); quantiles for calibration
  3. brow events     raises / frowns per minute, data vs model; do raises
                     follow pitch accents in the data?; their FLAME shapes;
                     the model's brow "jitter" energy
  4. gestures        speed / high-frequency share / variety, data vs model,
                     and the model with "sticky" decoding at several margins
  5. eyes            is the data's eye motion real? (share of moving frames,
                     saccades, counter-rotation against the head)

Run (pantomatrix env):  python diagnose_root.py --out /root/diag
"""
import argparse
import json
from pathlib import Path

import numpy as np

import beat2_tools as bt

FPS = bt.FPS
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/root/diag")
ap.add_argument("--speakers", type=int, default=8)
ap.add_argument("--clips-per-speaker", type=int, default=2)
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--repo", default="/root/PantoMatrix")
ap.add_argument("--weights", default="/root/emage_weights")
ap.add_argument("--model-folder", default="/root/PantoMatrix/emage_evaltools/smplx_models")
ap.add_argument("--data-dir", default="/root/beat2_subset")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

import librosa  # noqa: E402

print("== data ==")
clips = bt.fetch_beat2(args.data_dir, args.speakers, args.clips_per_speaker)
for c in clips:
    print(f"  {c[0]}{'  (held-out)' if c[3] else ''}")

print("\n== model ==")
em = bt.Emage(args.repo, args.weights)
SR = em.sr
print(f"  embedding tables in the model (sizes): {em.embedding_sizes}")
n_spk = em.n_speakers
print(f"  speaker embedding: {n_spk} speakers" if n_spk > 1 else "  speaker embedding not identified — speaker tests skipped")

R = {"clips": {}}
acc = {k: [] for k in ("blink_tpl", "raise_tpl", "frown_tpl")}
pool = {"data": {"blink_ibi": [], "blink_dur": [], "blink_in_pause": [], "pause_frac": [],
                 "raise_near_accent": [], "accent_density": []}}
margins = [0.0, 0.5, 1.0, 2.0, 3.0]

for stem, mp, wp, held in clips:
    d = np.load(mp, allow_pickle=True)
    n = int(min(len(d["poses"]), args.seconds * FPS))
    audio = librosa.load(wp, sr=SR)[0][:int(n / FPS * SR)]
    n = min(n, int(len(audio) / SR * FPS))
    P_d = np.asarray(d["poses"][:n], np.float32)
    E_d = np.asarray(d["expressions"][:n], np.float32)
    db, sp, f0 = bt.audio_feats(audio, SR, n)
    accents = bt.pitch_accents(f0)
    spk_num = bt.speaker_number(stem)
    rec = {"speaker": stem.split("_")[1], "speaker_number": spk_num, "seconds": round(n / FPS, 1)}

    srcs = {"data": (P_d, E_d)}
    srcs["model"] = tuple(x[:n] for x in em.gen(audio, speaker=0))
    if 0 <= spk_num - 1 < n_spk:
        srcs["model_own_speaker"] = tuple(x[:n] for x in em.gen(audio, speaker=spk_num - 1))
    n = min(len(v[0]) for v in srcs.values())

    for s, (P, E) in srcs.items():
        P, E = P[:n], E[:n]
        J, _ = bt.face_mesh(args.model_folder, P, E)
        sig = bt.face_signals(J)
        bl = bt.blinks(sig["aperture"])
        ra, fr, thr = bt.brow_events(sig["brow"], sig["inner"], exclude=bl)
        mins = n / FPS / 60
        g = sig["gap"]
        brow_hf = np.abs(np.fft.rfft((sig["brow"] - sig["brow"].mean()) * np.hanning(n))) ** 2
        frq = np.fft.rfftfreq(n, 1 / FPS)
        eyes = P[:, 69:75].reshape(n, 2, 3)
        eye_deg = np.degrees(np.linalg.norm(eyes, axis=2)).mean(1)
        eye_vel = np.abs(np.diff(eye_deg)) * FPS
        head_yaw = P[:, 46] + P[:, 37]                       # head + neck yaw (y component)
        eye_yaw = P[:, 70]
        vor = float(np.corrcoef(np.diff(head_yaw), np.diff(eye_yaw))[0, 1]) \
            if np.std(np.diff(eye_yaw)) > 1e-6 else float("nan")
        rec[s] = {
            "blinks_per_min": len(bl) / mins,
            "raises_per_min": len(ra) / mins, "frowns_per_min": len(fr) / mins,
            "brow_jitter_hf": float(brow_hf[frq > 3].sum() / (brow_hf[frq > 0.2].sum() + 1e-12)),
            "gap_p50": float(np.percentile(g[sp], 50)) if sp.sum() > 30 else float("nan"),
            "gap_p95": float(np.percentile(g[sp], 95)) if sp.sum() > 30 else float("nan"),
            "gap_std": float(g[sp].std()) if sp.sum() > 30 else float("nan"),
            "jaw_p95_deg": float(np.percentile(np.degrees(np.linalg.norm(P[:, 66:69], axis=1))[sp], 95)),
            "eye_moving_frac": float((eye_deg > 0.5).mean()),
            "eye_std_deg": float(eye_deg.std()),
            "saccades_per_min": len(bt.runs(eye_vel > 60, 1, 6)) / mins,
            "eye_head_vel_corr": vor,
            **{f"body_{k}": v for k, v in bt.body_stats(P).items()},
        }
        if s == "data":
            for k, ev in (("blink_tpl", bl), ("raise_tpl", ra), ("frown_tpl", fr)):
                t, cnt = bt.template(E, ev)
                if cnt:
                    acc[k].append((t, cnt))
            starts = [b[0] for b in bl]
            pool["data"]["blink_ibi"] += list(np.diff(starts) / FPS)
            pool["data"]["blink_dur"] += [(e - s_) / FPS for s_, e, _ in bl]
            pool["data"]["blink_in_pause"] += [bool(not sp[min(b[2], n - 1)]) for b in bl]
            pool["data"]["pause_frac"].append(float((~sp).mean()))
            near = [any(abs(r[2] - a) <= 8 for a in accents) for r in ra]
            pool["data"]["raise_near_accent"] += near
            pool["data"]["accent_density"].append(len(accents) * 17 / max(n, 1))   # chance a ±8 window hits one

    # sticky decoding sweep (body only; face is continuous and unaffected)
    rec["sticky"] = {}
    for m in margins:
        P, _ = em.gen(audio, speaker=0, margin=m)
        st = bt.body_stats(P[:n])
        st["switch_rate"] = {k: round(v, 3) for k, v in em.last_switch_rate.items()}
        rec["sticky"][str(m)] = st
    R["clips"][stem] = rec
    print(f"  {stem}: blinks/min data {rec['data']['blinks_per_min']:.1f} model {rec['model']['blinks_per_min']:.1f} | "
          f"raises {rec['data']['raises_per_min']:.1f}/{rec['model']['raises_per_min']:.1f} "
          f"frowns {rec['data']['frowns_per_min']:.1f}/{rec['model']['frowns_per_min']:.1f} | "
          f"gap p95 {rec['data']['gap_p95']:.1f}/{rec['model']['gap_p95']:.1f}mm | "
          f"speed {rec['data']['body_speed_dps']:.0f}/{rec['model']['body_speed_dps']:.0f}")

# ---- speaker sweep on two clips: does the mouth size follow the speaker embedding?
sweep = {}
if n_spk > 1:
    for stem, mp, wp, _ in clips[:2]:
        audio = librosa.load(wp, sr=SR)[0][:int(20 * SR)]
        n = int(20 * FPS)
        _, sp, _ = bt.audio_feats(audio, SR, n)
        row = {}
        for k in range(n_spk):
            P, E = em.gen(audio, speaker=k)
            J, _ = bt.face_mesh(args.model_folder, P[:n], E[:n])
            g = bt.face_signals(J)["gap"]
            m_ = min(len(g), len(sp))
            row[k] = round(float(np.percentile(g[:m_][sp[:m_]], 95)), 2)
        sweep[stem] = row
R["speaker_sweep_gap_p95"] = sweep

# ---- templates (data-derived FLAME directions)
def wavg(lst):
    if not lst:
        return np.zeros(100, np.float32), 0
    w = np.array([c for _, c in lst], float)
    return (np.sum([t * c for t, c in lst], 0) / w.sum()).astype(np.float32), int(w.sum())


tpl = {k: wavg(v) for k, v in acc.items()}
D = pool["data"]
np.savez(out / "face_templates.npz",
         blink=tpl["blink_tpl"][0], raise_=tpl["raise_tpl"][0], frown=tpl["frown_tpl"][0],
         blink_ibi=np.array(D["blink_ibi"]), blink_dur=np.array(D["blink_dur"]),
         blink_in_pause=np.array(D["blink_in_pause"]), pause_frac=np.array(D["pause_frac"]))
R["templates_from_events"] = {k: v[1] for k, v in tpl.items()}

# ---- render the templates: do they look like what they claim to be?
z = np.zeros((1, 165), np.float32)
rows = []
for name, key in (("blink", "blink_tpl"), ("raise", "raise_tpl"), ("frown", "frown_tpl")):
    t = tpl[key][0]
    E = np.stack([np.zeros(100), t, 2 * t]).astype(np.float32)
    _, V = bt.face_mesh(args.model_folder, np.repeat(z, 3, 0), E, keep_verts=True)
    rows.append((f"{name} ({tpl[key][1]} events)", list(V)))
bt.render_faces(out / "templates.png", rows, labels=["neutral", "+template", "+2x template"])

# ---- aggregate
def mean_of(src, key):
    vals = [R["clips"][c][src][key] for c in R["clips"] if src in R["clips"][c]]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


keys = list(next(iter(R["clips"].values()))["data"].keys())
summary = {k: {s: round(mean_of(s, k), 3) for s in ("data", "model", "model_own_speaker")} for k in keys}
sticky = {m: {k: round(float(np.mean([R["clips"][c]["sticky"][m][k] for c in R["clips"]])), 3)
              for k in ("speed_dps", "hf_ratio", "variety", "amplitude_deg")} for m in map(str, margins)}
ibi = np.array(D["blink_ibi"])
facts = {
    "blink": {"rate_per_min_data": summary["blinks_per_min"]["data"],
              "rate_per_min_model": summary["blinks_per_min"]["model"],
              "duration_s_median": round(float(np.median(D["blink_dur"])), 3) if D["blink_dur"] else None,
              "interval_s_median": round(float(np.median(ibi)), 2) if len(ibi) else None,
              "interval_s_p10_p90": [round(float(np.percentile(ibi, 10)), 2), round(float(np.percentile(ibi, 90)), 2)]
              if len(ibi) else None,
              "share_in_pauses": round(float(np.mean(D["blink_in_pause"])), 2) if D["blink_in_pause"] else None,
              "pause_share_of_time": round(float(np.mean(D["pause_frac"])), 2)},
    "brow": {"raise_near_pitch_accent": round(float(np.mean(D["raise_near_accent"])), 2)
             if D["raise_near_accent"] else None,
             "chance_level": round(float(np.mean(D["accent_density"])), 2) if D["accent_density"] else None},
}
R.update({"summary": summary, "sticky_decoding": sticky, "facts": facts})
(out / "diagnose_report.json").write_text(json.dumps(R, indent=2, default=float))

L = ["# Root-cause diagnosis", "", f"{len(R['clips'])} held-out BEAT2 clips. Model speaker embedding sizes: {em.embedding_sizes}", "",
     "| measure | data | model (speaker 0) | model (own speaker) |", "|---|---|---|---|"]
for k, v in summary.items():
    L.append(f"| {k} | {v['data']} | {v['model']} | {v['model_own_speaker']} |")
L += ["", "## blinks (data)", json.dumps(facts["blink"]), "", "## brow raises vs pitch accents (data)",
      json.dumps(facts["brow"]), "", "## sticky decoding (body), mean over clips",
      "| margin | speed deg/s | high-freq share | variety | amplitude deg |", "|---|---|---|---|---|"]
for m, v in sticky.items():
    L.append(f"| {m} | {v['speed_dps']} | {v['hf_ratio']} | {v['variety']} | {v['amplitude_deg']} |")
L.append(f"| data | {summary['body_speed_dps']['data']} | {summary['body_hf_ratio']['data']} | "
         f"{summary['body_variety']['data']} | {summary['body_amplitude_deg']['data']} |")
L += ["", "## speaker sweep: lip gap p95 (mm) by speaker embedding", json.dumps(sweep), "",
      f"## templates from data events: {R['templates_from_events']}  (see templates.png)"]
(out / "diagnose_report.md").write_text("\n".join(L))
print("\n" + "\n".join(L))
