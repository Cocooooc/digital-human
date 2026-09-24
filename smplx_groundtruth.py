"""
SMPL-X ground truth: render EMAGE's output as a 2D skeleton, and check the
rest skeleton the VRM retarget is built on.

Why this exists
---------------
Right now a bad-looking avatar has two possible causes and no way to tell them
apart: EMAGE generated poor motion, or our SMPL-X→VRM retarget mangled good
motion. This renders the motion on SMPL-X's *own* skeleton, which is the
reference the data was authored against — so any difference from the avatar is
ours.

Why not the repo's renderer
---------------------------
fast_render.py goes through pyrender → OpenGL, which on a headless box means
EGL/OSMesa and a dependency spiral. npz2pose.render2d is lighter but imports
pytorch3d purely to project 3D points onto a screen, which is a division. Both
are avoidable: smplx and cv2 are already installed, so the only genuinely
missing piece is the body model file — and PantoMatrix itself points at a
mirror of it (see MODEL_URL below).

Run:
    conda activate pantomatrix
    python /root/smplx_groundtruth.py                    # rest skeleton only
    python /root/smplx_groundtruth.py motion.npy         # + render a video
"""
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path("/root/smplx_models")
SMPLX_DIR = MODEL_DIR / "smplx"
# The URL is taken from PantoMatrix's own emage_utils/motion_rep_transfer.py,
# which downloads this file automatically when it needs the body model.
# PantoMatrix's own motion_rep_transfer.py points at the author's old HF Space
# for this file, but that Space was later replaced with a code mirror and the
# path 404s. The file lives in the emage_evaltools repo instead — which is also
# the folder name the repo's renderer expects, so this is where it was meant to
# come from all along.
MODEL_URL = ("https://huggingface.co/H-Liu1997/emage_evaltools/resolve/main/"
             "smplx_models/smplx/SMPLX_NEUTRAL_2020.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def section(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


# ---------------------------------------------------------------------------
section("1. Body model")
# ---------------------------------------------------------------------------
SMPLX_DIR.mkdir(parents=True, exist_ok=True)
dated = SMPLX_DIR / "SMPLX_NEUTRAL_2020.npz"
# smplx.create() looks for SMPLX_NEUTRAL.npz; PantoMatrix's renderer reads the
# _2020 name directly. Keep both so either path works.
plain = SMPLX_DIR / "SMPLX_NEUTRAL.npz"

if not dated.exists():
    print(f"downloading {MODEL_URL}")
    print("  (167MB, a minute or two)")
    try:
        urllib.request.urlretrieve(MODEL_URL, dated)
    except Exception as e:
        print(f"❌ download failed: {e}")
        print("\nTry the mirror instead:")
        print(f"  export HF_ENDPOINT=https://hf-mirror.com")
        print(f"  wget -O {dated} \\\n    {MODEL_URL.replace('huggingface.co', 'hf-mirror.com')}")
        sys.exit(1)
print(f"✅ {dated}  ({dated.stat().st_size/1e6:.0f} MB)")

if not plain.exists():
    os.link(dated, plain) if hasattr(os, "link") else None
print(f"✅ {plain}")

import smplx  # noqa: E402

model = smplx.create(
    str(MODEL_DIR), model_type="smplx", gender="neutral", ext="npz",
    num_betas=300, num_expression_coeffs=100,
    use_pca=False, use_face_contour=False, batch_size=1,
).to(DEVICE).eval()
print(f"✅ SMPL-X loaded on {DEVICE}")


# ---------------------------------------------------------------------------
section("2. The rest skeleton — what the VRM retarget is measured against")
# ---------------------------------------------------------------------------
# The retarget correction is "rotate this VRM bone from where it rests onto
# where SMPL-X rests it". The VRM half is read in the browser; this is the
# other half, and until now it was a hand-written table of guesses like
# "arms point along ±X". Here it comes from the model itself.
SMPLX_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
    18, 19, 15, 15, 15,
    20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
]
NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye", "right_eye",
]

with torch.no_grad():
    zero = lambda n: torch.zeros(1, n, device=DEVICE)
    rest = model(betas=zero(300), transl=zero(3), expression=zero(100),
                 global_orient=zero(3), body_pose=zero(63), jaw_pose=zero(3),
                 leye_pose=zero(3), reye_pose=zero(3),
                 left_hand_pose=zero(45), right_hand_pose=zero(45))
J = rest.joints[0, :55].cpu().numpy()

# child list, for "which way does this bone point"
children = {}
for i, p in enumerate(SMPLX_PARENTS):
    if p >= 0:
        children.setdefault(p, []).append(i)

REF_CHILD = {0: 3}          # pelvis follows the spine, not the mean of spine+legs
dirs = {}
print(f"{'idx':>3} {'joint':<16} {'rest direction (x, y, z)':>28}")
for i in range(55):
    kids = [REF_CHILD[i]] if i in REF_CHILD else children.get(i, [])
    if not kids:
        continue
    v = J[kids].mean(axis=0) - J[i]
    n = np.linalg.norm(v)
    if n < 1e-9:
        continue
    v = v / n
    dirs[i] = v
    if i <= 21:
        name = NAMES[i] if i < len(NAMES) else str(i)
        print(f"{i:>3} {name:<16} {v[0]:>9.3f} {v[1]:>8.3f} {v[2]:>8.3f}")

print("\n--- paste this into test_retarget.html, replacing SMPLX_REST_DIR ---")
print("const SMPLX_REST_DIR = {")
for i in sorted(dirs):
    x, y, z = dirs[i]
    print(f"  {i}: [{x:.4f}, {y:.4f}, {z:.4f}],")
print("};")

# The point of measuring: does the guessed table hold up?
GUESSED = {16: [1, 0, 0], 17: [-1, 0, 0], 18: [1, 0, 0], 19: [-1, 0, 0],
           3: [0, 1, 0], 7: [0, 0, 1], 8: [0, 0, 1]}
print("\n--- how good were the guesses ---")
for i, g in GUESSED.items():
    if i not in dirs:
        continue
    ang = np.degrees(np.arccos(np.clip(np.dot(dirs[i], g), -1, 1)))
    flag = "✅" if ang < 5 else ("⚠️" if ang < 20 else "❌ was wrong")
    print(f"  {NAMES[i]:<16} guessed {g} → off by {ang:5.1f}°  {flag}")


# ---------------------------------------------------------------------------
section("3. Render EMAGE motion as a 2D skeleton")
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("no motion file given — skipping.")
    print("\nTo render, save EMAGE's output first:")
    print("  np.save('motion.npy', pred['motion_axis_angle'][0].cpu().numpy())")
    print("then re-run:  python /root/smplx_groundtruth.py motion.npy")
    sys.exit(0)

import cv2  # noqa: E402

motion = np.load(sys.argv[1]).astype(np.float32)   # (frames, 165)
n_frames = motion.shape[0]
print(f"{sys.argv[1]}: {n_frames} frames × {motion.shape[1]}")
assert motion.shape[1] == 165, "expected 165 = 55 joints × 3 axis-angle"

# Forward kinematics, chunked so a long clip doesn't blow up VRAM.
all_joints = []
CHUNK = 120
for s in range(0, n_frames, CHUNK):
    m = torch.from_numpy(motion[s:s + CHUNK]).to(DEVICE)
    b = m.shape[0]
    z = lambda n: torch.zeros(b, n, device=DEVICE)
    sub = smplx.create(str(MODEL_DIR), model_type="smplx", gender="neutral",
                       ext="npz", num_betas=300, num_expression_coeffs=100,
                       use_pca=False, use_face_contour=False,
                       batch_size=b).to(DEVICE).eval()
    with torch.no_grad():
        out = sub(betas=z(300), transl=z(3), expression=z(100),
                  global_orient=z(3),              # keep the body centred
                  body_pose=m[:, 3:66],
                  jaw_pose=m[:, 66:69],
                  leye_pose=m[:, 69:72], reye_pose=m[:, 72:75],
                  left_hand_pose=m[:, 75:120], right_hand_pose=m[:, 120:165])
    all_joints.append(out.joints[:, :55].cpu().numpy())
    print(f"  frames {s}-{s+b}")
joints = np.concatenate(all_joints, 0)    # (frames, 55, 3)

# Perspective projection — the one thing pytorch3d was being imported for.
W = H = 640
FOCAL = 600.0
CAM_Z = 2.6                      # camera on +Z looking back at the body
Y_MID = float(joints[:, :, 1].mean())

def project(p3d):
    depth = np.maximum(CAM_Z - p3d[:, 2], 1e-3)
    u = W / 2 + FOCAL * p3d[:, 0] / depth
    v = H / 2 - FOCAL * (p3d[:, 1] - Y_MID) / depth
    return np.stack([u, v], 1).astype(np.int32)

out_path = str(Path(sys.argv[1]).with_suffix("")) + "_skeleton.mp4"
vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

for f in range(n_frames):
    canvas = np.full((H, W, 3), 24, np.uint8)
    p = project(joints[f])
    for i, par in enumerate(SMPLX_PARENTS):
        if par < 0:
            continue
        # hands thinner and dimmer, so the body reads at a glance
        hand = i >= 25
        cv2.line(canvas, tuple(p[par]), tuple(p[i]),
                 (110, 110, 110) if hand else (90, 200, 255), 1 if hand else 3)
    for i in range(22):
        cv2.circle(canvas, tuple(p[i]), 4, (255, 255, 255), -1)
    cv2.putText(canvas, f"frame {f}/{n_frames}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
    vw.write(canvas)

vw.release()
print(f"\n✅ {out_path}")
print("\nPull it to your Mac:")
print(f"  scp -P <ssh port> root@<host>:{out_path} ~/Desktop/")
print("""
This is the reference. Whatever it shows is what EMAGE actually generated —
so if it looks natural here and wrong on the avatar, the fault is the
retarget, and if it looks wrong here too, no amount of retarget tuning will
fix it.
""")
