"""
Find the SMPL-X rest skeleton, without downloading anything.

Why this matters: retargeting SMPL-X onto a VRM needs both rest poses. The VRM
half is readable in the browser for free. The SMPL-X half is 55 rest joint
positions — a few hundred numbers — and the question is whether PantoMatrix
already ships them somewhere, or whether that means registering at
smpl-x.is.tue.mpg.de for the full model files.

Worth asking before registering, because the answer is often yes: anything that
renders a mesh, computes a joint loss, or converts between representations has
to know the rest skeleton, and repos routinely vendor a small template for it.

Run:
    conda activate pantomatrix
    python /root/scout_smplx_rest.py
"""
import os
import re
import sys
from pathlib import Path

REPO = Path("/root/PantoMatrix")
SEARCH_ROOTS = [REPO, Path("/root/emage_weights"), Path("/root")]


def section(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


# ---------------------------------------------------------------------------
section("1. SMPL-X model files already on disk?")
# ---------------------------------------------------------------------------
# The real body model is SMPLX_NEUTRAL.npz/.pkl (~100MB+). If one of these is
# already here, everything else in this script is moot.
PATTERNS = ["SMPLX_*.npz", "SMPLX_*.pkl", "smplx_*.npz", "*SMPLX*.pkl",
            "SMPL_X*.npz", "*smplx*.npz"]
found_model = []
for root in [Path("/root")]:
    for pat in PATTERNS:
        for f in root.rglob(pat):
            if f.is_file() and f.stat().st_size > 1_000_000:
                found_model.append(f)
if found_model:
    for f in sorted(set(found_model)):
        print(f"  ✅ {f}  ({f.stat().st_size/1e6:.1f} MB)")
else:
    print("  ❌ no full SMPL-X body model on disk")


# ---------------------------------------------------------------------------
section("2. Small vendored templates (this is the likely win)")
# ---------------------------------------------------------------------------
# 55 joints x 3 floats is ~1-30KB depending on format. Look for small npy/npz/
# json/txt files whose name hints at joints, skeleton, template or t-pose.
HINT = re.compile(r"(joint|skelet|templ|tpose|t_pose|rest|bone|parent|kintree|"
                  r"mean_pose|beta|offset)", re.I)
candidates = []
for root in SEARCH_ROOTS:
    if not root.exists():
        continue
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".npy", ".npz", ".json", ".txt", ".pkl", ".yaml"):
            continue
        size = f.stat().st_size
        if size > 5_000_000 or size < 100:
            continue
        if HINT.search(f.name):
            candidates.append((f, size))
    if root == Path("/root"):
        break   # /root already covers the others; avoid duplicate walks

seen = set()
for f, size in sorted(candidates, key=lambda t: t[1]):
    if f in seen:
        continue
    seen.add(f)
    print(f"  {f}  ({size/1024:.1f} KB)")
if not candidates:
    print("  (nothing matched)")


# ---------------------------------------------------------------------------
section("3. Peek inside the numeric candidates")
# ---------------------------------------------------------------------------
# What we want has a 55 (or 127, or 10475 for vertices) somewhere in its shape.
import numpy as np  # noqa: E402

def describe(path):
    try:
        if path.suffix == ".npy":
            a = np.load(path, allow_pickle=True)
            return {"<array>": getattr(a, "shape", type(a))}
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=True) as z:
                return {k: getattr(z[k], "shape", "?") for k in z.files}
    except Exception as e:
        return {"error": str(e)[:80]}
    return None


INTERESTING = {55, 165, 127, 54, 22, 21, 52, 156}
for f, _ in sorted(set(candidates), key=lambda t: t[1]):
    if f.suffix not in (".npy", ".npz"):
        continue
    d = describe(f)
    if not d:
        continue
    print(f"\n  {f.name}")
    for k, shp in d.items():
        flag = ""
        if isinstance(shp, tuple) and any(n in INTERESTING for n in shp):
            flag = "   ← shape mentions a SMPL-X joint count"
        print(f"      {k}: {shp}{flag}")


# ---------------------------------------------------------------------------
section("4. What does the repo's own code expect?")
# ---------------------------------------------------------------------------
# If PantoMatrix's rendering path constructs smplx.create(...), the arguments
# tell us exactly which files it wants and where it looks for them.
hits = []
for py in REPO.rglob("*.py"):
    try:
        t = py.read_text(errors="ignore")
    except Exception:
        continue
    for m in re.finditer(r"^.*\b(smplx\.create|SMPLX\(|model_path|model_folder|"
                         r"J_regressor|v_template|kintree|parents)\b.*$", t, re.M):
        line = m.group(0).strip()
        if len(line) < 200:
            hits.append((py.relative_to(REPO), line))

seen_lines = set()
for path, line in hits[:60]:
    if line in seen_lines:
        continue
    seen_lines.add(line)
    print(f"  {path}: {line}")
if not hits:
    print("  (no references — the repo may never build a body model at all)")


# ---------------------------------------------------------------------------
section("5. Is the smplx package installed?")
# ---------------------------------------------------------------------------
try:
    import smplx
    print(f"  ✅ smplx {getattr(smplx, '__version__', '?')} at {Path(smplx.__file__).parent}")
    # The package vendors the kinematic tree even without the model data.
    for f in Path(smplx.__file__).parent.rglob("*"):
        if f.is_file() and f.suffix in (".npz", ".npy", ".pkl", ".json"):
            print(f"     bundled: {f.name} ({f.stat().st_size/1024:.1f} KB)")
except ImportError:
    print("  ❌ smplx not installed (`pip install smplx` is small — it's the")
    print("     model *data* that needs registration, not the code)")


section("What the answer means")
print("""
  Section 1 hit  → we already have everything; no registration needed.

  Section 2/3 hit on something shaped (55, 3) or (10475, 3)
                 → that's the rest skeleton; enough to compute the retarget
                   correction exactly, still no registration.

  Nothing        → register at smpl-x.is.tue.mpg.de and download SMPLX_NEUTRAL.
                   Free, academic use, takes a few minutes plus approval. Worth
                   doing regardless: the robot path almost certainly needs the
                   real body model too.
""")
