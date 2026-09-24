#!/bin/bash
# Turn on PantoMatrix's own renderer: audio -> EMAGE -> SMPL-X mesh -> mp4.
#
# This is the dependency chain we deliberately skipped during the first setup,
# when the goal was an inference service. The goal has changed, so it has to be
# faced — but carefully, because two things here can break the working EMAGE
# install:
#
#   numpy  — PantoMatrix pins 1.23.5; matplotlib/imageio will happily upgrade it
#   torch  — nothing here should touch it, so it gets checked before and after
#
# Offscreen OpenGL on a machine with no display needs a backend. EGL uses the
# GPU and is fast; OSMesa renders on the CPU and is slow but works anywhere.
# Probe EGL first, fall back, and report which one the render command must use
# rather than guessing later.
#
# Run:
#   conda activate pantomatrix
#   bash /root/setup_render.sh
set -u

say() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

say "0. State before touching anything"
python - <<'PY'
import numpy, torch
print(f"  numpy {numpy.__version__}   torch {torch.__version__}  cuda={torch.cuda.is_available()}")
PY

say "1. System OpenGL libraries"
# libgl/libegl for the GPU path, libosmesa for the CPU fallback. freeglut and
# libglib are pulled in by pyrender's and opencv's import chains.
apt-get update -qq
apt-get install -y -qq \
  libgl1 libgl1-mesa-dri libegl1 libglib2.0-0 \
  libosmesa6 libosmesa6-dev freeglut3-dev \
  ffmpeg 2>&1 | tail -3
echo "  installed"

say "2. Python render packages"
# numpy is pinned on the same line so pip resolves it as a constraint rather
# than upgrading it to satisfy matplotlib.
pip install -q \
  "numpy==1.23.5" \
  pyrender trimesh "pyglet<2" \
  imageio imageio-ffmpeg matplotlib \
  PyOpenGL PyOpenGL-accelerate 2>&1 | tail -5
echo "  installed"

say "3. Did that break EMAGE?"
python - <<'PY'
import numpy, torch
ok = numpy.__version__.startswith("1.23")
print(f"  numpy {numpy.__version__}  {'✅' if ok else '❌ CHANGED — EMAGE may break'}")
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
try:
    import smplx, cv2
    print("  smplx ✅  cv2 ✅")
except Exception as e:
    print(f"  ❌ {e}")
PY

say "4. Which offscreen backend actually works"
# Render a single triangle. Anything that comes back non-black means the
# backend is live; an exception means it isn't. Cheaper than finding out
# halfway through a 600-frame render.
for backend in egl osmesa; do
  PYOPENGL_PLATFORM=$backend python - "$backend" <<'PY' 2>/dev/null
import sys, os, numpy as np
backend = sys.argv[1]
try:
    import trimesh, pyrender
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    scene = pyrender.Scene(bg_color=[0, 0, 0, 1], ambient_light=[.5, .5, .5])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3)
    pose = np.eye(4); pose[2, 3] = 3
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3), pose=pose)
    r = pyrender.OffscreenRenderer(64, 64)
    color, _ = r.render(scene)
    r.delete()
    print(f"  {backend:<7} ✅ works  (mean pixel {color.mean():.1f})")
except Exception as e:
    print(f"  {backend:<7} ❌ {type(e).__name__}: {str(e)[:70]}")
PY
done
echo "  (use the first one marked ✅: export PYOPENGL_PLATFORM=<that>)"

say "5. SMPL-X body model"
# The renderer needs the mesh topology, not just the skeleton. PantoMatrix
# The URL inside motion_rep_transfer.py is stale (its HF Space was replaced
# and now 404s); the file lives in the emage_evaltools repo, which is also
# the folder name the repo's own renderer looks in.
DEST=/root/PantoMatrix/emage_evaltools/smplx_models/smplx
mkdir -p "$DEST"
URL="https://huggingface.co/H-Liu1997/emage_evaltools/resolve/main/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
if [ ! -f "$DEST/SMPLX_NEUTRAL_2020.npz" ]; then
  echo "  downloading (167MB)…"
  wget -q --show-progress -O "$DEST/SMPLX_NEUTRAL_2020.npz" "$URL" \
    || { echo "  ❌ failed; retry with the mirror:";
         echo "     wget -O $DEST/SMPLX_NEUTRAL_2020.npz ${URL/huggingface.co/hf-mirror.com}"; }
fi
# smplx.create() looks for the undated name; fast_render reads the dated one.
[ -f "$DEST/SMPLX_NEUTRAL_2020.npz" ] && ln -sf SMPLX_NEUTRAL_2020.npz "$DEST/SMPLX_NEUTRAL.npz"
ls -lh "$DEST" 2>/dev/null | tail -3

say "6. How the official script wants to be called"
# Its own argparse is the authority on this — reading it beats guessing flags.
cd /root/PantoMatrix
python - <<'PY'
import re, pathlib
src = pathlib.Path("test_emage_audio.py").read_text()
for m in re.finditer(r"add_argument\((.*?)\)\s*$", src, re.M):
    print("  --", m.group(1)[:110])
i = src.find("if __name__")
print("\n  --- entry point ---")
print("  " + "\n  ".join(src[i:i+600].splitlines()))
PY

say "Next"
cat <<'TXT'
  If a backend came back ✅ in step 4, render with:

     cd /root/PantoMatrix
     export PYOPENGL_PLATFORM=<egl or osmesa>
     python test_emage_audio.py <flags from step 6>

  If both failed, the GPU path needs the container's NVIDIA GL libraries and
  the CPU path needs a working libOSMesa — send the two error lines and we
  pick the fix from what they actually say, rather than installing more.
TXT
