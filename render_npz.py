"""
Render an EMAGE .npz to mp4 — one process, one model load, one GL context.

Why not the repo's renderer
---------------------------
test_emage_audio.py's visualization spawns one process per frame. For a 5s clip
that is 150 processes, each of which loads SMPL-X, builds its own EGL context
and allocates its own CUDA memory; they took 226s and then sat at 97% idle CPU
without exiting — the work was done, the pool would not close.

None of that setup needs repeating per frame. Loading the body model once and
reusing a single renderer turns the per-frame cost into what it should be: one
forward pass and one draw.

Run (EGL was verified working by setup_render.sh):
    conda activate pantomatrix
    export PYOPENGL_PLATFORM=egl
    python /root/render_npz.py /root/render_test/out/clip5s_output.npz \
                              --audio /root/render_test/audio/clip5s.wav
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Must be set before pyrender imports OpenGL, so do it here rather than relying
# on the caller remembering to export it.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("npz", help="EMAGE output .npz")
ap.add_argument("--audio", help="wav to mux onto the video")
ap.add_argument("--out", help="output mp4 (default: alongside the npz)")
ap.add_argument("--model-folder", default="/root/PantoMatrix/emage_evaltools/smplx_models")
ap.add_argument("--size", type=int, default=720)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--chunk", type=int, default=60, help="frames per forward pass")
ap.add_argument("--max-frames", type=int, default=0, help="stop early, for a quick look")
ap.add_argument("--use-global", action="store_true",
                help="keep the clip's global orientation instead of facing the camera")
args = ap.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
npz_path = Path(args.npz)
out_path = Path(args.out) if args.out else npz_path.with_suffix("").with_name(
    npz_path.stem + "_render.mp4")


def step(t):
    print(f"\n\033[1m== {t} ==\033[0m")


# ---------------------------------------------------------------------------
step("1. What's in the npz")
# ---------------------------------------------------------------------------
data = np.load(npz_path, allow_pickle=True)
for k in data.files:
    v = data[k]
    print(f"  {k:<22} {getattr(v, 'shape', type(v))}  {getattr(v, 'dtype', '')}")

poses = np.asarray(data["poses"], dtype=np.float32)          # (n, 165)
n_all = poses.shape[0]
n = min(n_all, args.max_frames) if args.max_frames else n_all
poses = poses[:n]
assert poses.shape[1] == 165, f"expected 165 axis-angle values, got {poses.shape[1]}"

betas = np.asarray(data["betas"], dtype=np.float32) if "betas" in data.files else None
if betas is not None and betas.ndim == 2:
    betas = betas[0]                                          # shape is fixed over time
expr = np.asarray(data["expressions"], dtype=np.float32)[:n] if "expressions" in data.files else None
print(f"\n  rendering {n} of {n_all} frames at {args.fps}fps "
      f"→ {n/args.fps:.1f}s of video")


# ---------------------------------------------------------------------------
step("2. Body model (loaded once)")
# ---------------------------------------------------------------------------
import smplx  # noqa: E402

t0 = time.time()


def make_model(batch):
    return smplx.create(
        args.model_folder, model_type="smplx", gender="neutral", ext="npz",
        num_betas=300, num_expression_coeffs=100,
        use_pca=False, use_face_contour=False, flat_hand_mean=False,
        batch_size=batch,
    ).to(DEVICE).eval()


faces = np.load(Path(args.model_folder) / "smplx" / "SMPLX_NEUTRAL_2020.npz",
                allow_pickle=True)["f"].astype(np.int32)
print(f"  faces {faces.shape} | device {DEVICE} | {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
step("3. Forward kinematics → vertices")
# ---------------------------------------------------------------------------
t0 = time.time()
verts = []
for s in range(0, n, args.chunk):
    p = torch.from_numpy(poses[s:s + args.chunk]).to(DEVICE)
    b = p.shape[0]
    model = make_model(b)
    z = lambda k: torch.zeros(b, k, device=DEVICE)

    kw = dict(
        betas=(torch.from_numpy(betas).to(DEVICE).unsqueeze(0).repeat(b, 1)
               if betas is not None else z(300)),
        transl=z(3),                     # keep the body centred in frame
        expression=(torch.from_numpy(expr[s:s + b]).to(DEVICE) if expr is not None else z(100)),
        # Zero by default: EMAGE's global orientation drifts over a clip, and a
        # body that slowly turns away from the camera looks like a bug in the
        # motion when it isn't one.
        global_orient=p[:, 0:3] if args.use_global else z(3),
        body_pose=p[:, 3:66],
        jaw_pose=p[:, 66:69],
        leye_pose=p[:, 69:72],
        reye_pose=p[:, 72:75],
        left_hand_pose=p[:, 75:120],
        right_hand_pose=p[:, 120:165],
    )
    with torch.no_grad():
        verts.append(model(**kw).vertices.cpu().numpy())
    print(f"  frames {s}-{s+b}")
    del model
    torch.cuda.empty_cache()

vertices = np.concatenate(verts, 0)      # (n, 10475, 3)
print(f"  {vertices.shape} in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
step("4. Render")
# ---------------------------------------------------------------------------
import trimesh  # noqa: E402
import pyrender  # noqa: E402
import imageio  # noqa: E402

W = H = args.size

# Frame the whole clip, not the first pose: a camera fitted to frame 0 clips
# the arms the moment they move.
lo = vertices.reshape(-1, 3).min(0)
hi = vertices.reshape(-1, 3).max(0)
centre = (lo + hi) / 2
extent = float(max(hi - lo))
YFOV = np.pi / 4
dist = (extent / 2) / np.tan(YFOV / 2) * 1.25

cam_pose = np.eye(4)
cam_pose[:3, 3] = [centre[0], centre[1], centre[2] + dist]

scene = pyrender.Scene(bg_color=[0.09, 0.09, 0.11, 1.0], ambient_light=[0.35] * 3)
scene.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0), pose=cam_pose)
# Key light on the camera, fill from the side, so the body reads as a shape
# rather than a silhouette.
scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=cam_pose)
side = np.eye(4)
side[:3, 3] = [centre[0] - dist, centre[1] + dist / 2, centre[2] + dist / 2]
scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.5), pose=side)

material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.65, 0.70, 0.78, 1.0], metallicFactor=0.0, roughnessFactor=0.75)

renderer = pyrender.OffscreenRenderer(W, H)
silent = out_path.with_name(out_path.stem + "_silent.mp4")
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
    color, _ = renderer.render(scene)
    writer.append_data(color)
    if i % 30 == 0 or i == n - 1:
        el = time.time() - t0
        print(f"  frame {i+1}/{n}  {el:.1f}s  ({(i+1)/max(el,1e-6):.1f} fps)")

writer.close()
renderer.delete()
render_time = time.time() - t0
print(f"  {n} frames in {render_time:.1f}s → {silent}")


# ---------------------------------------------------------------------------
step("5. Audio")
# ---------------------------------------------------------------------------
if args.audio and Path(args.audio).exists():
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(silent), "-i", args.audio,
           "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)]
    if subprocess.run(cmd).returncode == 0:
        silent.unlink(missing_ok=True)
        print(f"  ✅ {out_path}")
    else:
        print(f"  ffmpeg failed; the silent video is still at {silent}")
else:
    silent.rename(out_path)
    print(f"  no audio given → {out_path}")

print(f"""
  {n} frames, {n/args.fps:.1f}s of video, rendered in {render_time:.1f}s
  ({render_time/(n/args.fps):.1f}x realtime)

Pull it to your Mac:
  scp -P <ssh port> root@<host>:{out_path} ~/Desktop/
""")
