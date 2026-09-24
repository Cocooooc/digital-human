"""
Find out exactly what EMAGE's visualization needs, before installing anything.

We deliberately skipped the rendering stack during setup — it was pulling in
pyrender → OpenGL → imageio and none of it was needed for inference. Now the
goal has changed: we want to *see* the generated motion to judge whether it's
worth building retargeting on top of. So look at what visualize_one actually
does and what it requires, rather than installing a pile of graphics packages
and hoping.

Run:
    conda activate pantomatrix
    python /root/inspect_visualization.py
"""
import importlib.util
import re
from pathlib import Path

REPO = Path("/root/PantoMatrix")


def section(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


# ---------------------------------------------------------------------------
section("1. What visualize_one() does")
# ---------------------------------------------------------------------------
# Read as text — importing is what drags in the rendering chain.
script = (REPO / "test_emage_audio.py").read_text()
i = script.find("def visualize_one")
j = script.find("\ndef ", i + 1)
print(script[i:j if j != -1 else i + 2500])


# ---------------------------------------------------------------------------
section("2. fast_render's imports and entry points")
# ---------------------------------------------------------------------------
fr = REPO / "emage_utils/fast_render.py"
if fr.exists():
    text = fr.read_text()
    print("--- imports ---")
    for line in text.splitlines():
        if re.match(r"^\s*(import|from)\s", line):
            print("  " + line.strip())

    print("\n--- functions it offers ---")
    for m in re.finditer(r"^def (\w+)\((.*?)\)", text, re.M):
        print(f"  {m.group(1)}({m.group(2)[:90]})")
else:
    print("fast_render.py not found")


# ---------------------------------------------------------------------------
section("3. Which of those packages are missing")
# ---------------------------------------------------------------------------
def have(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


render_pkgs = ["pyrender", "trimesh", "imageio", "imageio_ffmpeg", "matplotlib",
               "cv2", "pytorch3d", "smplx", "OpenGL", "moviepy"]
for p in render_pkgs:
    print(f"  {'✅' if have(p) else '❌'} {p}")


# ---------------------------------------------------------------------------
section("4. Does it need the SMPL-X body model files?")
# ---------------------------------------------------------------------------
# Turning 165 axis-angle numbers into something visible requires forward
# kinematics, which needs SMPL-X's rest pose and joint hierarchy — i.e. the
# body model files, which are a separate registered download. If the code
# references them, that's a prerequisite we have to deal with before any
# visualization (or retargeting) can work.
hits = []
for py in list(REPO.rglob("*.py")):
    if "test" not in py.name and "emage" not in str(py) and "utils" not in str(py):
        continue
    try:
        t = py.read_text()
    except Exception:
        continue
    for m in re.finditer(r"[\"']([^\"']*(?:smplx|SMPLX|SMPL_X)[^\"']*)[\"']", t):
        hits.append((py.relative_to(REPO), m.group(1)))

seen = set()
for path, ref in hits:
    if ref in seen:
        continue
    seen.add(ref)
    print(f"  {path}: {ref}")

print("\nLooking for the model files on disk:")
for pat in ["**/SMPLX*.npz", "**/smplx/*.npz", "**/*SMPLX*.pkl"]:
    for f in REPO.rglob(pat.replace("**/", "")):
        print(f"  found: {f}")
else:
    print("  (searched repo; also check /root for a smplx models folder)")

section("Summary")
print("""
What this tells us:

  If visualization only needs matplotlib/cv2/imageio → cheap, install and
  render a 2D skeleton video to judge motion quality.

  If it needs pyrender/pytorch3d → that's a 3D mesh render on a headless
  server: doable but fiddly (EGL/OSMesa), and it also needs the SMPL-X body
  model files, which require registering at smpl-x.is.tue.mpg.de.

  Either way the SMPL-X body model question matters beyond visualization:
  retargeting onto VRM also needs the joint hierarchy those files define.
""")
