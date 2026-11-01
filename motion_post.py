"""
Motion post-processing for live conversation — three fixes on top of EMAGE.

  1. Idle motion   EMAGE only moves while there is speech. Between replies
                   (listening, thinking) the avatar froze on its last frame.
                   IdleMotion generates breathing, a slow weight shift and
                   small head/arm drift around a rest pose, as a pure
                   function of time, so it never repeats visibly and can be
                   asked for in chunks of any length.
  2. Amplitude     damp_to() shrinks each body joint's rotation toward ONE
                   fixed rest pose by a factor (default 0.7). Fixed matters:
                   damping each sentence toward its own mean would give every
                   sentence a different centre and add a jump at each seam.
  3. Seams         The live pipeline generates one sentence at a time, and
                   consecutive sentences start from unrelated poses: measured
                   21-37 degrees at a boundary, 3-5x a normal frame step.
                   blend_in() takes the offset between where the avatar is
                   and where the new segment starts, and fades that offset
                   out over ~0.33s. The new motion keeps its own velocity,
                   so it does not look like a pause-then-move.

Session ties them together and remembers the last output frame, so speech
-> speech, idle -> speech and speech -> idle are all continuous.

SMPL-X layout, 55 joints x 3 (axis-angle) = 165 per frame:
  0-21 body (0 = pelvis/global orientation), 22 jaw, 23/24 eyes,
  25-39 left hand, 40-54 right hand.

numpy + scipy only, so it runs in either conda env and in the browser-side
tests. Python 3.9 / numpy 1.23 compatible.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

FPS = 30
N_JOINTS = 55
BODY = list(range(22))
JAW, EYE_L, EYE_R = 22, 23, 24
HANDS = list(range(25, 55))
# Joints whose offset is faded at a seam. Jaw and eyes are left alone: the jaw
# is lip-sync and must follow the audio from its first frame; the eyes belong
# to the gaze model.
SEAM_JOINTS = BODY + HANDS


def _rv(poses, j):
    return poses[:, 3 * j:3 * j + 3]


def _rot(poses, j):
    return R.from_rotvec(_rv(poses, j))


# ---------------------------------------------------------------------------
# Root heading
# ---------------------------------------------------------------------------
def fix_root(poses, window_s=2.0):
    """Remove the slow heading drift of the pelvis, keep pitch/roll and quick
    sways. (Zeroing the whole global orientation, as the first renderers did,
    also drops EMAGE's ~19 deg backward pelvis tilt and tips the head down.)"""
    n = len(poses)
    if n == 0:
        return poses.copy()
    r = _rot(poses, 0)
    f = r.apply([0.0, 0.0, 1.0])
    heading = np.unwrap(np.arctan2(f[:, 0], f[:, 2]))
    k = max(1, int(window_s * FPS)) | 1
    k = min(k, (n // 2) * 2 + 1)
    slow = np.convolve(np.pad(heading, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    undo = R.from_rotvec(np.stack([np.zeros(n), -slow, np.zeros(n)], 1))
    out = poses.copy()
    out[:, 0:3] = (undo * r).as_rotvec()
    return out


# ---------------------------------------------------------------------------
# Rest pose and amplitude
# ---------------------------------------------------------------------------
def rest_pose_from(poses):
    """Per-joint mean rotation of a (preferably long) clip -> (165,).
    Jaw closed and eyes centred: a rest pose with the mouth half open would
    leave the idle avatar gaping."""
    p = fix_root(np.asarray(poses, np.float64))
    rest = np.zeros(3 * N_JOINTS)
    for j in range(N_JOINTS):
        rest[3 * j:3 * j + 3] = _rot(p, j).mean().as_rotvec()
    rest[3 * JAW:3 * JAW + 3] = 0.0
    rest[3 * EYE_L:3 * EYE_R + 3] = 0.0
    return rest.astype(np.float32)


def damp_to(poses, rest, scale, joints=BODY):
    """Shrink each joint's rotation toward `rest` by `scale` (1 = unchanged,
    0 = frozen at rest). Rotations are scaled as relative rotations from the
    rest pose, not as raw axis-angle: scaling raw axis-angle pulls every joint
    toward the SMPL-X T-pose and the arms swing up."""
    if scale == 1.0:
        return poses.copy()
    out = poses.copy()
    for j in joints:
        m = R.from_rotvec(rest[3 * j:3 * j + 3])
        delta = (m.inv() * _rot(poses, j)).as_rotvec() * scale
        out[:, 3 * j:3 * j + 3] = (m * R.from_rotvec(delta)).as_rotvec()
    return out


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------
def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def blend_in(poses, expr, prev_pose, prev_expr, frames=10, joints=SEAM_JOINTS,
             max_step_deg=4.0):
    """Start `poses` from `prev_pose` and fade the difference out.

    For each joint, D = prev * new[0]^-1 is the rotation that would put the
    new segment's first frame where the avatar currently is. Frame t gets
    D scaled by (1 - w(t)), w rising 0 -> 1 over `frames`, applied on top of
    the new motion. So frame 0 continues from the previous frame and by
    `frames` the segment is exactly as generated, with its own motion intact
    throughout (unlike a crossfade, which freezes and then catches up).

    The window stretches for big gaps: the fade's steepest frame moves
    1.5 x gap / frames, so it is lengthened until that stays under
    `max_step_deg` (a normal frame step is ~1-7 deg at p50-p99)."""
    n = len(poses)
    if n == 0 or prev_pose is None or frames <= 0:
        return poses.copy(), (None if expr is None else expr.copy())
    gap = max(np.degrees(np.linalg.norm(
        (R.from_rotvec(prev_pose[3 * j:3 * j + 3]) * R.from_rotvec(poses[0, 3 * j:3 * j + 3]).inv()).as_rotvec()))
        for j in joints)
    if max_step_deg:
        frames = max(frames, int(np.ceil(1.5 * gap / max_step_deg)))
    k = min(frames, n)
    w = _smoothstep((np.arange(k) + 1) / (k + 1))            # never exactly 0 or 1
    out = poses.copy()
    for j in joints:
        new0 = R.from_rotvec(poses[0, 3 * j:3 * j + 3])
        D = (R.from_rotvec(prev_pose[3 * j:3 * j + 3]) * new0.inv()).as_rotvec()
        Dt = R.from_rotvec(D[None, :] * (1 - w)[:, None])
        out[:k, 3 * j:3 * j + 3] = (Dt * R.from_rotvec(poses[:k, 3 * j:3 * j + 3])).as_rotvec()
    e = None
    if expr is not None:
        e = expr.copy()
        if prev_expr is not None and len(e):
            e[:k] += (prev_expr - expr[0])[None, :] * (1 - w)[:, None]
    return out, e


# ---------------------------------------------------------------------------
# Idle
# ---------------------------------------------------------------------------
class IdleMotion:
    """Breathing + weight shift + small drift around a rest pose.

    Every channel is a sum of a few sinusoids with fixed random frequencies
    and phases, i.e. a deterministic function of absolute time. Any chunk is
    continuous with the chunk before it, and with incommensurate frequencies
    the pattern does not visibly repeat.

    Amplitudes are deliberately small (1-3 degrees): idle motion that is
    noticeable on its own reads as fidgeting.
    """

    # (joint, axis, amplitude deg, kind). axis: 0 = pitch (x), 1 = yaw (y), 2 = roll (z)
    CHANNELS = [
        (3, 0, 1.2, "breath"), (6, 0, 1.5, "breath"), (9, 0, 1.8, "breath"),
        (13, 2, 1.2, "breath"), (14, 2, -1.2, "breath"),          # collars lift
        (0, 2, 1.5, "sway"), (3, 2, -0.9, "sway"),                 # weight shift
        (12, 0, 1.5, "drift"), (12, 1, 2.0, "drift"),              # neck
        (15, 0, 2.0, "drift"), (15, 1, 3.0, "drift"), (15, 2, 1.0, "drift"),
        (16, 2, 1.5, "drift"), (17, 2, 1.5, "drift"),              # shoulders
        (18, 1, 3.0, "drift"), (19, 1, 3.0, "drift"),              # elbows
        (20, 0, 3.0, "drift"), (21, 0, 3.0, "drift"),              # wrists
    ]
    BREATH_HZ = 0.23          # ~14 breaths a minute

    def __init__(self, rest, seed=0):
        self.rest = np.asarray(rest, np.float64)
        rng = np.random.default_rng(seed)
        self.params = []
        for _, _, _, kind in self.CHANNELS:
            if kind == "breath":
                self.params.append(([self.BREATH_HZ], [0.0], [1.0]))
            elif kind == "sway":
                self.params.append((rng.uniform(0.04, 0.09, 2), rng.uniform(0, 2 * np.pi, 2), [0.7, 0.3]))
            else:
                self.params.append((rng.uniform(0.05, 0.35, 3), rng.uniform(0, 2 * np.pi, 3), [0.5, 0.3, 0.2]))

    def frames(self, start_frame, n):
        t = (start_frame + np.arange(n)) / FPS
        out = np.tile(self.rest, (n, 1))
        offs = np.zeros((n, N_JOINTS, 3))
        for (j, ax, amp, _), (fr, ph, wt) in zip(self.CHANNELS, self.params):
            sig = sum(w * np.sin(2 * np.pi * f * t + p) for f, p, w in zip(fr, ph, wt))
            offs[:, j, ax] += np.radians(amp) * sig
        for j in set(c[0] for c in self.CHANNELS):
            base = R.from_rotvec(self.rest[3 * j:3 * j + 3])
            out[:, 3 * j:3 * j + 3] = (base * R.from_rotvec(offs[:, j])).as_rotvec()
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Session: the live stream
# ---------------------------------------------------------------------------
class Session:
    """One avatar's continuous output stream.

        s = Session(rest, scale=0.7)
        s.idle(60)                 # 2s idle
        s.speak(poses, expr)       # sentence 1, blended in from idle
        s.speak(poses2, expr2)     # sentence 2, blended in from sentence 1
        s.idle(90)                 # back to idle, blended out of speech

    Every call returns (poses, expressions) ready to play, and continues
    exactly from the last frame returned by any previous call.
    """

    def __init__(self, rest, scale=0.7, seam_frames=10, idle_in_frames=20, seed=0):
        self.rest = np.asarray(rest, np.float32)
        self.scale = scale
        self.seam_frames = seam_frames
        self.idle_in_frames = idle_in_frames
        self.idler = IdleMotion(self.rest, seed)
        self.clock = 0                      # frames emitted so far; drives idle time
        self.last_pose = None
        self.last_expr = None
        self._idle_fresh = True             # next idle chunk must blend out of speech

    def _emit(self, poses, expr):
        self.clock += len(poses)
        if len(poses):
            self.last_pose = poses[-1].copy()
            self.last_expr = expr[-1].copy()
        return poses.astype(np.float32), expr.astype(np.float32)

    def speak(self, poses, expr):
        p = fix_root(np.asarray(poses, np.float64))
        p = damp_to(p, self.rest, self.scale, BODY)
        e = np.asarray(expr, np.float64)
        p, e = blend_in(p, e, self.last_pose, self.last_expr, self.seam_frames)
        self._idle_fresh = True
        return self._emit(p, e)

    def idle(self, n):
        p = self.idler.frames(self.clock, n).astype(np.float64)
        e = np.zeros((n, 100))
        if self._idle_fresh and self.last_pose is not None:
            # Mouth and face settle back to neutral over the same window.
            p, e = blend_in(p, e, self.last_pose, self.last_expr, self.idle_in_frames,
                            joints=SEAM_JOINTS + [JAW])
        self._idle_fresh = False
        return self._emit(p, e)


# ---------------------------------------------------------------------------
# Measurements (used by demo_postprocess.py and the self-test)
# ---------------------------------------------------------------------------
UPPER = [13, 14, 16, 17, 18, 19, 20, 21]


def step_deg(poses, joints=UPPER):
    """Mean rotation between consecutive frames over `joints`, degrees, (n-1,)."""
    return np.mean([np.degrees((_rot(poses, j)[:-1].inv() * _rot(poses, j)[1:]).magnitude())
                    for j in joints], axis=0)


def deviation_p95(poses, rest, joints=UPPER):
    return {j: float(np.percentile(np.degrees(
        (R.from_rotvec(rest[3 * j:3 * j + 3]).inv() * _rot(poses, j)).magnitude()), 95))
        for j in joints}


if __name__ == "__main__":
    # Self-test on synthetic motion; no model needed.
    rng = np.random.default_rng(1)

    def fake_sentence(n, seed):
        g = np.random.default_rng(seed)
        t = np.arange(n)[:, None] / FPS
        base = g.normal(0, 0.35, 165)
        wobble = 0.25 * np.sin(2 * np.pi * g.uniform(0.3, 1.2, 165) * t + g.uniform(0, 6, 165))
        return (base + wobble).astype(np.float32), g.normal(0, 0.3, (n, 100)).astype(np.float32)

    sents = [fake_sentence(150, s) for s in range(4)]
    rest = rest_pose_from(np.concatenate([s[0] for s in sents]))

    raw = np.concatenate([fix_root(s[0]) for s in sents])
    s = Session(rest, scale=0.7)
    parts = [s.idle(60)[0]] + [s.speak(*x)[0] for x in sents] + [s.idle(90)[0]]
    out = np.concatenate(parts)

    st_raw = step_deg(raw)
    st_out = step_deg(out)
    bounds = np.cumsum([len(p) for p in parts])[:-1] - 1
    raw_b = [149, 299, 449]
    print("raw  seam steps", np.round(st_raw[raw_b], 1), "| normal p99", round(float(np.percentile(st_raw, 99)), 1))
    print("post seam steps", np.round(st_out[bounds], 1), "| normal p99", round(float(np.percentile(st_out, 99)), 1))
    idle_speed = step_deg(out[:60]).mean() * FPS
    print("idle speed deg/s", round(float(idle_speed), 2), "(frozen = 0)")
    d_raw = np.mean(list(deviation_p95(raw, rest).values()))
    d_out = np.mean(list(deviation_p95(np.concatenate(parts[1:-1]), rest).values()))
    print("p95 deviation from rest", round(d_raw, 1), "->", round(d_out, 1))
    assert max(st_out[bounds]) <= np.percentile(st_out, 99) * 1.5, "seam still jumps"
    worst = max(st_out[b:b + 40].max() for b in bounds)
    print("largest step inside any fade window", round(float(worst), 1), "deg")
    assert worst < max(np.percentile(st_raw, 99) * 1.5, 6.0), "fade itself too steep"
    assert idle_speed > 0.5, "idle frozen"
    assert d_out < d_raw, "damping did nothing"
    # idle chunks are continuous with each other
    s2 = Session(rest)
    a = s2.idle(45)[0]; b = s2.idle(45)[0]
    whole = Session(rest).idle(90)[0]
    assert np.allclose(np.concatenate([a, b]), whole, atol=1e-5), "idle chunks not continuous"
    print("self-test OK")
