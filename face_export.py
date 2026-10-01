"""
Face: measure it, export it, and render it close enough to see.

The feedback was that facial expression isn't visible in the body render. That
has two possible causes and they lead in opposite directions:

  A. framing — in a 720px full-body shot the head is ~60 pixels tall, and no
     expression survives that. Fix the camera.
  B. the data — EMAGE's face output is genuinely weak, in which case no camera
     helps and a face-specialist model (Audio2Face) is the answer.

Guessing between them wastes days, so section 1 measures the data before any
picture is rendered. If the jaw and the expression coefficients move with
speech, it was the camera; if they barely move, it was the model.

Section 2 exports the face as its own file, because the required delivery
format is body-as-SMPL-X and face-as-blendshape, separately.

  A note on "blendshape": EMAGE emits 100 FLAME expression coefficients. FLAME's
  expression space IS a linear blendshape basis, so these are blendshape
  weights — but in FLAME's own basis, not ARKit's 52 named shapes
  (jawOpen, mouthSmileLeft, ...). If the target rig or the robot expects ARKit
  names, that is a further conversion, and worth settling before building on it.

Run:
    conda activate pantomatrix
    export PYOPENGL_PLATFORM=egl
    python /root/face_export.py /root/render_test/out/clip5s_output.npz \
           --audio /root/render_test/audio/clip5s.wav --side-by-side
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--audio")
ap.add_argument("--out-dir", default="")
ap.add_argument("--model-folder", default="/root/PantoMatrix/emage_evaltools/smplx_models")
ap.add_argument("--size", type=int, default=720)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--chunk", type=int, default=60)
ap.add_argument("--side-by-side", action="store_true",
                help="body and face in one frame, so the two can be judged together")
ap.add_argument("--no-render", action="store_true", help="measure and export only")
ap.add_argument("--face-cam", default="lock", choices=["lock", "follow", "fixed"],
                help="lock: ride the head so the face stays front-on (default); "
                     "follow: track its position only; fixed: a static camera")
args = ap.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
npz_path = Path(args.npz)
out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent
out_dir.mkdir(parents=True, exist_ok=True)
stem = npz_path.stem


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


data = np.load(npz_path, allow_pickle=True)
poses = np.asarray(data["poses"], dtype=np.float32)
n = poses.shape[0]
jaw = poses[:, 66:69]                     # SMPL-X joint 22
expr = (np.asarray(data["expressions"], dtype=np.float32)
        if "expressions" in data.files else np.zeros((n, 100), np.float32))
duration = n / args.fps


# ---------------------------------------------------------------------------
section("1. Is the face actually moving?")
# ---------------------------------------------------------------------------
# Standard deviation over time, per channel. A coefficient that never changes
# contributes a fixed offset to the face, not an expression — so the count of
# channels that actually vary is the honest measure of how much animation
# exists, regardless of how large the raw numbers look.
jaw_mag = np.linalg.norm(jaw, axis=1)
jaw_deg = np.degrees(jaw_mag)
expr_std = expr.std(axis=0)

print(f"  {n} frames, {duration:.2f}s at {args.fps}fps\n")

print("  jaw (the most visible facial motion there is)")
print(f"    opening      min {jaw_deg.min():6.2f}°   mean {jaw_deg.mean():6.2f}°   "
      f"max {jaw_deg.max():6.2f}°")
print(f"    variation    std {jaw_deg.std():6.2f}°   range {jaw_deg.max()-jaw_deg.min():6.2f}°")

# Speech moves the jaw several times a second; a jaw that drifts slowly is not
# talking. Count direction changes as a crude but unfakeable rate measure.
d = np.diff(jaw_deg)
turns = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))
print(f"    direction changes: {turns} over {duration:.1f}s "
      f"= {turns/max(duration,1e-6):.1f}/s")

print("\n  expression coefficients (FLAME, 100 channels)")
for thresh, label in ((0.001, "barely"), (0.01, "visibly"), (0.1, "strongly")):
    print(f"    {label:<9} active (std > {thresh}): "
          f"{int((expr_std > thresh).sum()):>3} / 100")
print(f"    largest std  {expr_std.max():.4f} on channel {int(expr_std.argmax())}")
print(f"    value range  [{expr.min():.3f}, {expr.max():.3f}]")

top = np.argsort(expr_std)[::-1][:8]
print("    most active channels: " +
      ", ".join(f"#{i}({expr_std[i]:.3f})" for i in top))

# The verdict. Jaw and expression are judged separately, because this model
# does not necessarily put mouth opening in the jaw joint: FLAME's expression
# basis contains mouth shapes too, so a speaking face can show a quiet jaw
# alongside a very loud expression vector.
#
# An earlier version of this check treated a quiet jaw as proof of a dead face
# and announced "close to static" over data with 97 of 100 channels varying
# strongly. Wrong, and confidently wrong — the kind of answer that sends a
# project off to integrate a second model it does not need.
# Rate, not amplitude. Measured on real EMAGE output the jaw opens at most
# 2.5 degrees while reversing direction 7 times a second — unmistakably
# syllable rate, and unmistakably small. An amplitude gate of 1 degree called
# that face dead. What distinguishes speech from drift is how often the jaw
# turns around, so that is what gets tested; amplitude is reported and left
# to the eye.
jaw_rate = turns / max(duration, 1e-6)
lively_jaw = jaw_rate > 3.0
n_strong = int((expr_std > 0.1).sum())
lively_expr = int((expr_std > 0.01).sum()) >= 5
print()
if lively_expr and lively_jaw:
    print(f"  -> Both channels animate: the jaw reverses {jaw_rate:.1f} times a second\n"
          f"     (syllable rate) and {n_strong}/100 expression channels vary strongly.\n"
          "     Anything invisible before was rendering — framing, exposure, or a\n"
          "     camera that did not follow the head — not missing data.")
elif lively_expr:
    print(f"  -> The expression vector carries it: {n_strong}/100 channels vary strongly\n"
          "     while the jaw joint stays quiet. Normal for this model — FLAME's\n"
          "     basis includes mouth shapes, so speech can live in the coefficients\n"
          "     rather than the jaw rotation. The data is rich; judge it on the\n"
          "     close-up render, not on the jaw number.")
elif lively_jaw:
    print("  -> Jaw moves, expression channels barely do. Lip-sync without emotion:\n"
          "     usable for mouth shapes, thin for expression. This is the case where\n"
          "     a face-specialist model earns its keep.")
else:
    print("  -> Both channels are close to flat. No camera fixes this; the face has\n"
          "     to come from elsewhere (Audio2Face) or a different EMAGE config.\n"
          "     Re-check on a longer clip first — 5s can under-represent.")


# ---------------------------------------------------------------------------
section("2. Export the face as its own stream")
# ---------------------------------------------------------------------------
face_npz = out_dir / f"{stem}_face_blendshape.npz"
np.savez(face_npz, fps=args.fps, frames=n,
         jaw_pose=jaw, expression=expr,
         eye_left=poses[:, 69:72], eye_right=poses[:, 72:75],
         basis="FLAME expression (100 linear blendshape coefficients)",
         note="jaw_pose is axis-angle radians; not ARKit-named blendshapes")
print(f"  {face_npz}")

face_csv = out_dir / f"{stem}_face_blendshape.csv"
active = [int(i) for i in np.argsort(expr_std)[::-1][:20]]
with open(face_csv, "w") as f:
    f.write("frame,time_s,jaw_x,jaw_y,jaw_z,jaw_open_deg,"
            + ",".join(f"expr_{i}" for i in active) + "\n")
    for k in range(n):
        f.write(f"{k},{k/args.fps:.4f},{jaw[k,0]:.5f},{jaw[k,1]:.5f},{jaw[k,2]:.5f},"
                f"{jaw_deg[k]:.3f},"
                + ",".join(f"{expr[k,i]:.5f}" for i in active) + "\n")
print(f"  {face_csv}   (jaw + the 20 most active channels, readable in Excel)")

body_npz = out_dir / f"{stem}_body_smplx.npz"
np.savez(body_npz, fps=args.fps, frames=n,
         poses=poses, global_orient=poses[:, 0:3], body_pose=poses[:, 3:66],
         left_hand_pose=poses[:, 75:120], right_hand_pose=poses[:, 120:165],
         format="SMPL-X axis-angle, 55 joints x 3 = 165 per frame")
print(f"  {body_npz}")

if args.no_render:
    raise SystemExit(0)


# ---------------------------------------------------------------------------
section("3. Forward kinematics")
# ---------------------------------------------------------------------------
import smplx  # noqa: E402

faces = np.load(Path(args.model_folder) / "smplx" / "SMPLX_NEUTRAL_2020.npz",
                allow_pickle=True)["f"].astype(np.int32)
betas = np.asarray(data["betas"], dtype=np.float32) if "betas" in data.files else None
if betas is not None and betas.ndim == 2:
    betas = betas[0]

verts, heads = [], []
t0 = time.time()
for s in range(0, n, args.chunk):
    p = torch.from_numpy(poses[s:s + args.chunk]).to(DEVICE)
    b = p.shape[0]
    z = lambda k: torch.zeros(b, k, device=DEVICE)
    model = smplx.create(args.model_folder, model_type="smplx", gender="neutral",
                         ext="npz", num_betas=300, num_expression_coeffs=100,
                         use_pca=False, use_face_contour=False, flat_hand_mean=False,
                         batch_size=b).to(DEVICE).eval()
    with torch.no_grad():
        o = model(
            betas=(torch.from_numpy(betas).to(DEVICE).unsqueeze(0).repeat(b, 1)
                   if betas is not None else z(300)),
            transl=z(3), expression=torch.from_numpy(expr[s:s + b]).to(DEVICE),
            global_orient=z(3), body_pose=p[:, 3:66], jaw_pose=p[:, 66:69],
            leye_pose=p[:, 69:72], reye_pose=p[:, 72:75],
            left_hand_pose=p[:, 75:120], right_hand_pose=p[:, 120:165])
    verts.append(o.vertices.cpu().numpy())
    heads.append(o.joints[:, 15].cpu().numpy())     # SMPL-X joint 15 = head
    del model
    torch.cuda.empty_cache()
    print(f"  frames {s}-{s+b}")

vertices = np.concatenate(verts, 0)
head_track = np.concatenate(heads, 0)              # (n, 3), per frame
head_pos = head_track.mean(axis=0)
print(f"  {vertices.shape} in {time.time()-t0:.1f}s")

# Where the head is pointing, per frame. A face close-up on a static camera
# does not survive this data: EMAGE pitches the head far enough down that it
# leaves a fixed frame entirely. Chaining the rotations along the spine gives
# the head's world orientation, and a camera placed in that frame keeps the
# face front-on no matter where it turns — which is what makes the expression
# judgeable rather than confounded with head pose.
from scipy.spatial.transform import Rotation as _R  # noqa: E402

HEAD_CHAIN = [0, 3, 6, 9, 12, 15]   # pelvis, spine1-3, neck, head
head_rot = np.tile(np.eye(3), (n, 1, 1))
for j in HEAD_CHAIN:
    rv = np.zeros((n, 3), np.float32) if j == 0 else poses[:, j*3:j*3+3]
    head_rot = np.matmul(head_rot, _R.from_rotvec(rv).as_matrix())

# Light temporal smoothing. Un-smoothed, the camera inherits every frame of
# head jitter and the result is unwatchable even though each frame is correct.
def _smooth(a, k=5):
    pad = np.concatenate([a[:1]] * (k // 2) + [a] + [a[-1:]] * (k // 2), 0)
    return np.stack([pad[i:i + k].mean(0) for i in range(len(a))])

head_track_s = _smooth(head_track)
head_rot_s = _R.from_matrix(_smooth(head_rot).reshape(-1, 3, 3)).as_matrix()
print(f"  head pitch over the clip: "
      f"{np.degrees(np.arctan2(-head_rot[:,2,1], head_rot[:,2,2])).ptp():.0f}° of range")


# ---------------------------------------------------------------------------
section("4. Render")
# ---------------------------------------------------------------------------
import trimesh  # noqa: E402
import pyrender  # noqa: E402
import imageio  # noqa: E402

S = args.size
YFOV = np.pi / 4


def _look_rot(eye, target, up=(0.0, 1.0, 0.0)):
    """Rotation for a camera at `eye` aimed at `target`. pyrender cameras and
    lights look down their local -Z."""
    f = np.asarray(eye, float) - np.asarray(target, float)
    f /= max(np.linalg.norm(f), 1e-9)
    r = np.cross(np.asarray(up, float), f)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f], axis=1)


def camera_at(centre, half_height):
    """Place the camera so `half_height` metres fill half the frame."""
    pose = np.eye(4)
    pose[:3, 3] = [centre[0], centre[1], centre[2] + half_height / np.tan(YFOV / 2)]
    return pose


lo, hi = vertices.reshape(-1, 3).min(0), vertices.reshape(-1, 3).max(0)
body_centre = (lo + hi) / 2
body_cam_pose = camera_at(body_centre, float(max(hi - lo)) / 2 * 1.25)

# A head is about 22cm tall. Framing 16cm of it fills the frame with face,
# which is the whole point — the body shot had the head at ~8% of frame height.
face_centre = head_pos + np.array([0.0, 0.04, 0.0])
FACE_DIST = 0.16 / np.tan(YFOV / 2)      # 16cm of head fills the frame
face_cam_pose = camera_at(face_centre, 0.16)

# Exposure matters more here than anywhere else in this project. An earlier
# pass used a bright base colour with strong lights and clipped the face to
# flat white — and expression lives entirely in shallow creases around the
# mouth and eyes, which are the first thing blown highlights erase. Dimmer
# ambient plus a mid-grey material keeps those gradients inside range.
scene = pyrender.Scene(bg_color=[0.09, 0.09, 0.11, 1.0], ambient_light=[0.12] * 3)
body_cam = scene.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0),
                     pose=body_cam_pose)
face_cam = scene.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0),
                     pose=face_cam_pose)

# Light from slightly above and to the side of the face: flat frontal light
# flattens exactly the shallow creases that make an expression readable.
# Key from above and to one side: frontal light flattens exactly the creases
# that carry the expression. Fill opposite it, well below the key, so the
# shadow side stays readable without washing the modelling back out.
key = np.eye(4)
key[:3, 3] = face_centre + np.array([0.45, 0.55, 0.75])
key[:3, :3] = _look_rot(key[:3, 3], face_centre)
scene.add(pyrender.DirectionalLight(intensity=2.2), pose=key)
fill = np.eye(4)
fill[:3, 3] = face_centre + np.array([-0.6, 0.1, 0.6])
fill[:3, :3] = _look_rot(fill[:3, 3], face_centre)
scene.add(pyrender.DirectionalLight(intensity=0.7), pose=fill)

material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.52, 0.54, 0.60, 1.0], metallicFactor=0.0, roughnessFactor=0.7)

W = S * 2 if args.side_by_side else S
renderer = pyrender.OffscreenRenderer(S, S)
silent = out_dir / f"{stem}_face_silent.mp4"
writer = imageio.get_writer(str(silent), fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1)

t0 = time.time()
node = None
for i in range(n):
    tm = trimesh.Trimesh(vertices[i], faces, process=False)
    mesh = pyrender.Mesh.from_trimesh(tm, material=material, smooth=True)
    if node is not None:
        scene.remove_node(node)
    node = scene.add(mesh)

    if args.face_cam != "fixed":
        # Ride the head: position always, orientation too when locked. The
        # offset is expressed in the head's own frame, so "in front of the
        # face" stays in front of the face however the head turns.
        Rh = head_rot_s[i] if args.face_cam == "lock" else np.eye(3)
        eye = head_track_s[i] + Rh @ np.array([0.0, 0.04, FACE_DIST])
        pose = np.eye(4)
        pose[:3, 3] = eye
        pose[:3, :3] = _look_rot(eye, head_track_s[i] + Rh @ np.array([0.0, 0.04, 0.0]))
        face_cam.matrix = pose

    scene.main_camera_node = face_cam
    face_img, _ = renderer.render(scene)
    if args.side_by_side:
        scene.main_camera_node = body_cam
        body_img, _ = renderer.render(scene)
        frame = np.hstack([body_img, face_img])
    else:
        frame = face_img
    writer.append_data(frame)
    if i % 30 == 0 or i == n - 1:
        el = time.time() - t0
        print(f"  frame {i+1}/{n}  {el:.1f}s ({(i+1)/max(el,1e-6):.1f} fps)")

writer.close()
renderer.delete()
render_time = time.time() - t0

out_mp4 = out_dir / f"{stem}_face.mp4"
if args.audio and Path(args.audio).exists():
    ok = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent),
                         "-i", args.audio, "-c:v", "copy", "-c:a", "aac",
                         "-shortest", str(out_mp4)]).returncode == 0
    if ok:
        silent.unlink(missing_ok=True)
else:
    silent.rename(out_mp4)
    ok = True

print(f"\n  {'✅' if ok else '⚠️'} {out_mp4 if ok else silent}")
print(f"  {n} frames in {render_time:.1f}s"
      + ("  (body + face side by side)" if args.side_by_side else "  (face only)"))
print(f"""
Watch the mouth against the audio. If it opens and closes on the syllables,
lip-sync works and the earlier video simply framed it too small. If the mouth
moves but the rest of the face stays still, that is the measurement in
section 1 showing up on screen, and it is the case for a dedicated face model.
""")
