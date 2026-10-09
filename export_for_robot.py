"""
Package EMAGE motion for someone else's hardware.

The raw export is one (frames, 165) array. That is complete but not
self-describing: the first question anyone receiving it asks is "which three
numbers are the left elbow", and answering that in an email is how joint
mappings get silently wrong. So this writes the joint names and the parent
table into the file itself, alongside the numbers they describe.

It also adds two derived forms, because which one a robot wants depends on how
it is driven and guessing costs a round trip:

  joint_positions        world-space XYZ in metres — for visualising, for IK,
                         and for retargeting onto a body with different limb
                         lengths, where copying angles directly fails.
  joint_rotations_world  each joint's absolute orientation as a 3x3 — for a
                         robot whose joints are commanded in world or torso
                         frame rather than parent-relative.

All three are derived from one source, `poses_absolute`, and the script
checks before writing that the rotations it exports reproduce the positions
it exports to under a millimetre. A file whose two descriptions of the same
motion disagree is worse than a file with only one, because nothing errors —
the robot just does something slightly different depending on which field
its stack happens to read.

Run:
    conda activate pantomatrix
    python export_for_robot.py /root/render_test/out/clip5s_output.npz \
           --out-dir /root/for_robot
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

# SMPL-X's 55 joints, in the order the 165 numbers are packed.
JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "jaw", "left_eye", "right_eye",
]
for side in ("left", "right"):
    for finger in ("index", "middle", "pinky", "ring", "thumb"):
        JOINT_NAMES += [f"{side}_{finger}{k}" for k in (1, 2, 3)]

PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
    18, 19, 15, 15, 15,
    20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
]
assert len(JOINT_NAMES) == 55 and len(PARENTS) == 55

# The joints a legged or wheeled humanoid normally actuates — everything but
# the 30 finger joints and the face. Provided as an index list so the receiving
# side does not have to work out the slice themselves.
BODY_JOINTS = list(range(22))

ap = argparse.ArgumentParser()
ap.add_argument("npz", help="EMAGE output, or the *_body_smplx.npz export")
ap.add_argument("--out-dir", default="/root/for_robot")
ap.add_argument("--model-folder",
                default="/root/PantoMatrix/emage_evaltools/smplx_models")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--name", default="", help="basename for the outputs")
args = ap.parse_args()

src = Path(args.npz)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
stem = args.name or src.stem.replace("_output", "").replace("_body_smplx", "")

data = np.load(src, allow_pickle=True)
poses = np.asarray(data["poses"], dtype=np.float32)
n = poses.shape[0]
assert poses.shape[1] == 165, f"expected 165 per frame, got {poses.shape[1]}"
expr = (np.asarray(data["expressions"], dtype=np.float32)
        if "expressions" in data.files else None)
fps = int(data["fps"]) if "fps" in data.files else args.fps

print(f"{src.name}: {n} frames @ {fps}fps = {n/fps:.2f}s")


# ---------------------------------------------------------------------------
# Body model, loaded once for its constants
# ---------------------------------------------------------------------------
import smplx  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_model(batch):
    # flat_hand_mean=False matches both of PantoMatrix's own renderers, i.e.
    # the convention EMAGE's hand poses were produced in.
    return smplx.create(args.model_folder, model_type="smplx", gender="neutral",
                        ext="npz", num_betas=300, num_expression_coeffs=100,
                        use_pca=False, use_face_contour=False,
                        flat_hand_mean=False, batch_size=batch).to(device).eval()


ref = make_model(1)
# With flat_hand_mean=False, SMPL-X's forward pass does `full_pose += pose_mean`
# before anything else. pose_mean is zero for the body and a relaxed, slightly
# curled hand for the fingers. So EMAGE's finger values are offsets from that
# resting hand, not finger angles — and anything computed from the raw values
# without adding it back disagrees with what the body model actually does.
pose_mean = ref.pose_mean.detach().cpu().numpy().astype(np.float32)   # (165,)
assert pose_mean.shape == (165,), pose_mean.shape
poses_abs = poses + pose_mean[None, :]

hand_mean_deg = np.degrees(np.linalg.norm(pose_mean[75:].reshape(-1, 3), axis=1))
print(f"SMPL-X resting-hand offset: mean {hand_mean_deg.mean():.1f} deg, "
      f"max {hand_mean_deg.max():.1f} deg per finger joint  "
      f"(body joints: {np.abs(pose_mean[:66]).max():.1e})")

# Rest skeleton, for the self-check below. betas = 0, so v_shaped is the template.
J_rest = (ref.J_regressor @ ref.v_template).detach().cpu().numpy()     # (55, 3)
del ref


# ---------------------------------------------------------------------------
# World rotations, by walking the kinematic tree on the ABSOLUTE local poses
# ---------------------------------------------------------------------------
local = R.from_rotvec(poses_abs.reshape(-1, 3)).as_matrix().reshape(n, 55, 3, 3)
world = np.zeros_like(local)
for j, par in enumerate(PARENTS):
    world[:, j] = local[:, j] if par < 0 else np.matmul(world[:, par], local[:, j])
print(f"world rotations: {world.shape}")


# ---------------------------------------------------------------------------
# Joint positions, from the real SMPL-X body
# ---------------------------------------------------------------------------
# Expression is held at zero here. It moves the skeleton only by deforming the
# template the joints are regressed from, by fractions of a millimetre and only
# at the face — irrelevant to a robot, and it would make the rest skeleton
# frame-dependent, which breaks the exact self-check. The face is exported
# separately, unchanged.
positions = []
CHUNK = 60
for s0 in range(0, n, CHUNK):
    pc = torch.from_numpy(poses[s0:s0 + CHUNK]).to(device)
    b = pc.shape[0]
    z = lambda k: torch.zeros(b, k, device=device)
    m = make_model(b)
    with torch.no_grad():
        o = m(betas=z(300), transl=z(3), expression=z(100),
              global_orient=pc[:, 0:3], body_pose=pc[:, 3:66],
              jaw_pose=pc[:, 66:69], leye_pose=pc[:, 69:72],
              reye_pose=pc[:, 72:75],
              left_hand_pose=pc[:, 75:120], right_hand_pose=pc[:, 120:165])
    positions.append(o.joints[:, :55].cpu().numpy())
    del m
    torch.cuda.empty_cache()
positions = np.concatenate(positions, 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Self-check: do the exported rotations reproduce the exported positions?
# ---------------------------------------------------------------------------
# Rebuild every joint position independently from world rotations plus the
# rest skeleton, the way any receiving robot stack would, and compare with
# what SMPL-X computed. If the two disagree the file contradicts itself, and
# that must stop the export rather than ship.
fk = np.zeros((n, 55, 3), np.float64)
fk[:, 0] = J_rest[0]
for j in range(1, 55):
    par = PARENTS[j]
    offset = J_rest[j] - J_rest[par]
    fk[:, j] = fk[:, par] + np.einsum("nij,j->ni", world[:, par], offset)
err_mm = np.linalg.norm(fk - positions, axis=2) * 1000
body_err, hand_err = err_mm[:, :22].max(), err_mm[:, 25:].max()
print(f"self-check  rotations -> positions:  body max {body_err:.3f} mm,  "
      f"fingers max {hand_err:.3f} mm")

# The same check without the resting-hand offset, kept so the reason for it is
# visible in the log rather than just asserted in a comment.
local_raw = R.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(n, 55, 3, 3)
world_raw = np.zeros_like(local_raw)
for j, par in enumerate(PARENTS):
    world_raw[:, j] = local_raw[:, j] if par < 0 else np.matmul(world_raw[:, par], local_raw[:, j])
fk_raw = np.zeros((n, 55, 3), np.float64)
fk_raw[:, 0] = J_rest[0]
for j in range(1, 55):
    par = PARENTS[j]
    fk_raw[:, j] = fk_raw[:, par] + np.einsum("nij,j->ni", world_raw[:, par], J_rest[j] - J_rest[par])
raw_hand_err = (np.linalg.norm(fk_raw - positions, axis=2) * 1000)[:, 25:].max()
print(f"            (without the hand offset, fingers would be off by {raw_hand_err:.1f} mm)")

if max(body_err, hand_err) > 1.0:
    raise SystemExit("❌ exported rotations and positions disagree by more than 1mm — "
                     "not writing a self-contradictory file")
print("✅ rotations and positions agree")

# Root-relative: the pelvis sits at its template rest location, not the
# origin. Shift so it genuinely is at the origin, which is what a robot's base
# frame wants and what the metadata now states.
pelvis_rest = positions[:, :1].copy()
positions = positions - pelvis_rest
print(f"joint positions: {positions.shape}, pelvis-relative "
      f"(template pelvis was at {pelvis_rest[0,0].round(3).tolist()} m)  "
      f"height span {positions[:,:,1].max()-positions[:,:,1].min():.2f} m")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
bundle = out_dir / f"{stem}_motion.npz"
np.savez_compressed(
    bundle,
    fps=fps, frames=n, duration_seconds=round(n / fps, 3),
    poses=poses,
    poses_absolute=poses_abs,
    joint_names=np.array(JOINT_NAMES),
    parents=np.array(PARENTS, dtype=np.int32),
    body_joint_indices=np.array(BODY_JOINTS, dtype=np.int32),
    joint_positions=positions,
    joint_rotations_world=world.astype(np.float32),
    expression=expr if expr is not None else np.zeros((n, 0), np.float32),
    jaw_pose=poses[:, 66:69],
    convention=("SMPL-X. poses_absolute = local axis-angle in radians, "
                "parent-relative, 55 joints x 3 = 165 per frame, packed in "
                "joint_names order: the true joint rotations, use these to drive "
                "a robot. poses = EMAGE's raw output, identical for the body but "
                "with fingers stored as offsets from SMPL-X's resting hand "
                "(flat_hand_mean=False). joint_rotations_world and "
                "joint_positions are both derived from poses_absolute and agree "
                "to under 1mm. Y up, +Z forward, right-handed. Metres. Pelvis "
                "at origin."),
)
print(f"\n  {bundle}")

# Plain .npy too: some pipelines will not unpack an npz.
for name, arr in (("poses_absolute", poses_abs), ("poses", poses),
                  ("joint_positions", positions),
                  ("joint_rotations_world", world.astype(np.float32))):
    f = out_dir / f"{stem}_{name}.npy"
    np.save(f, arr)
    print(f"  {f}   {arr.shape}")

meta = {
    "fps": fps, "frames": n, "duration_seconds": round(n / fps, 3),
    "convention": {
        "format": "SMPL-X",
        "poses_absolute": "local axis-angle, radians, parent-relative, (frames, 55, 3) flattened to (frames, 165). THE ONE TO USE.",
        "poses": "EMAGE raw output. Same as poses_absolute for joints 0-24; fingers (25-54) are offsets from SMPL-X's resting hand, not absolute angles.",
        "joint_positions": "world XYZ, metres, (frames, 55, 3)",
        "joint_rotations_world": "absolute 3x3 rotation per joint, (frames, 55, 3, 3)",
        "axes": "Y up, +Z forward, right-handed",
        "root": "pelvis at origin in joint_positions; global translation not applied; body orientation per frame is joint 0 (pelvis) rotation",
    },
    "joints": [{"index": i, "name": JOINT_NAMES[i], "parent": PARENTS[i]}
               for i in range(55)],
    "body_joint_indices": BODY_JOINTS,
    "notes": [
        "Joints 25-54 are the fingers; a robot without hands can ignore them.",
        "For fingers, use poses_absolute, not poses: the raw values are offsets from a curled resting hand.",
        "Joints 23/24 (eyes) are always zero - EMAGE does not drive the eyes.",
        "Face expression is FLAME's 100-coefficient basis, not ARKit's 52 named blendshapes.",
    ],
}
mf = out_dir / f"{stem}_meta.json"
mf.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(f"  {mf}")

print(f"""
  {n} frames, {n/fps:.1f}s

  Sanity check on the receiving side — these should read as a person:
    left_elbow  (18)  mean |angle| {np.degrees(np.linalg.norm(poses_abs[:,54:57],axis=1)).mean():5.1f} deg
    right_elbow (19)  mean |angle| {np.degrees(np.linalg.norm(poses_abs[:,57:60],axis=1)).mean():5.1f} deg
    head        (15)  mean |angle| {np.degrees(np.linalg.norm(poses_abs[:,45:48],axis=1)).mean():5.1f} deg

  If a mapped joint moves in the opposite direction, it is an axis-sign
  difference between the two skeletons, not bad data — joint_positions is
  the unambiguous reference to check against, since a position has no
  handedness convention to get wrong.
""")
