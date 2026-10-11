"""Stylized seated girl figurine: knees up, arms hugging shins, head tilted skyward.

Front +Z, Y-up, millimeters. Signed distance-bound, not exact signed distance.
Sphere, capsule, cylinder, and rounded-box distances are exact and 1-Lipschitz. Ellipsoids use
(|(p-c)/r|-1)*min(r): the diagonal norm map has bound 1/min(r), cancelled by
scaling. Rotation about X is an isometry, so rX_* variants keep the bound.
Hard min preserves the bound; polynomial smooth-min has nonnegative partial
derivatives summing to one, so its gradient is a convex combination of input
gradients. Thus the field is globally 1-Lipschitz; finite samples are
regression checks, not proof.

SCENE records: (name, kind, pivot/center, parameters, blend_mm).
Kinds: ellipsoid, sphere, capsule, cylinder, rounded_box, rx_ellipsoid,
rx_sphere, rx_torus_arc. Cylinder parameters are (radius, half_height),
axis-aligned with Y. rx_* kinds evaluate in a frame rotated about X by `angle`
radians (right-hand rule): q = R_x(-angle) @ (p - pivot) - local_center.
A negative angle tips the face upward (nose +Z rotates toward +Y).

primitive_fields(points) -> (..., len(SCENE)); material(points) -> float RGB
(..., 3), approximate nearest-primitive diagnostic colors, not STL colors.
"""
import numpy as np

LIPSCHITZ = 1.0
DESCRIPTION = ("Stylized seated girl on a base disc: knees up, hugging shins, "
               "gazing upward; ~154 mm tall.")

TILT = -0.44  # head pitch (rad); negative tips the face skyward
HEAD_PIVOT = (0.0, 90.0, -10.0)

# name, kind, pivot, parameters, blend
SCENE = (
    ("hips", "ellipsoid", (0., 16., -6.), ((26., 16., 22.),), 0.),
    ("torso", "capsule", (0., 22., -8.), ((0., 60., -16.), 19.), 5.),
    ("neck", "capsule", (0., 62., -15.), ((0., 74., -13.), 7.), 3.),
    ("collar", "capsule", (0., 64., -14.), ((0., 74., -12.), 9.0), 2.),
    ("zipper", "capsule", (0., 36., 7.67), ((0., 54., 3.88), 1.6), 0.),
    ("head", "rx_ellipsoid", HEAD_PIVOT, (TILT, (0., 0., 0.), (19., 20.5, 19.)), 3.),
    ("haircap", "rx_ellipsoid", HEAD_PIVOT, (TILT, (0., 4., -5.), (20., 21.5, 20.)), 2.),
    ("eye_L", "rx_sphere", HEAD_PIVOT, (TILT, (-7., 3., 17.5), 2.6), 0.),
    ("eye_R", "rx_sphere", HEAD_PIVOT, (TILT, (7., 3., 17.5), 2.6), 0.),
    ("smile", "rx_torus_arc", HEAD_PIVOT, (TILT, (0., -3., 18.), 5., 1.4, (-2.6, -.54)), 0.),
    ("ponytail", "capsule", (-15., 86., -26.), ((-24., 52., -31.), 7.5), 3.),
    ("ponytail_end", "sphere", (-24., 50., -31.), (7.5,), 2.),
    ("thigh_L", "capsule", (-11., 20., 4.), ((-15., 46., 24.), 13.), 4.),
    ("thigh_R", "capsule", (11., 20., 4.), ((15., 46., 24.), 13.), 4.),
    ("knee_L", "sphere", (-15., 48., 26.), (11.5,), 3.),
    ("knee_R", "sphere", (15., 48., 26.), (11.5,), 3.),
    ("shin_L", "capsule", (-15., 46., 26.), ((-14., 10., 30.), 8.5), 3.),
    ("shin_R", "capsule", (15., 46., 26.), ((14., 10., 30.), 8.5), 3.),
    ("shoe_L", "rounded_box", (-14., 8., 40.), ((11., 7.5, 14.), 3.5), 2.),
    ("shoe_R", "rounded_box", (14., 8., 40.), ((11., 7.5, 14.), 3.5), 2.),
    ("charm_L1", "sphere", (-21., 14.0, 49.), (2.8,), 0.),
    ("charm_L2", "sphere", (-12., 14.5, 51.), (2.4,), 0.),
    ("charm_L3", "sphere", (-16., 15.0, 44.), (2.2,), 0.),
    ("charm_R1", "sphere", (21., 14.0, 49.), (2.8,), 0.),
    ("charm_R2", "sphere", (12., 14.5, 51.), (2.4,), 0.),
    ("charm_R3", "sphere", (16., 15.0, 44.), (2.2,), 0.),
    ("upperarm_L", "capsule", (-20., 56., -10.), ((-27., 36., 24.), 8.), 4.),
    ("upperarm_R", "capsule", (20., 56., -10.), ((27., 36., 24.), 8.), 4.),
    ("forearm_L", "capsule", (-27., 36., 24.), ((-6., 29., 47.), 7.), 4.),
    ("forearm_R", "capsule", (27., 36., 24.), ((6., 29., 47.), 7.), 4.),
    ("hands", "ellipsoid", (0., 28., 47.), ((10., 7.5, 8.5),), 3.),
    ("wristband", "capsule", (17.4, 32.8, 34.5), ((24.0, 35.0, 27.3), 7.8), 0.),
    ("base", "cylinder", (0., -1.5, 14.), (40., 2.5), 0.),
)

# skin, jacket white, shorts pink, shoe dark, hair dark, wristband light blue,
# eyes dark, charm yellow, charm coral, zipper gray, base gray
PALETTE = ((.96, .78, .66), (.94, .94, .95), (.97, .72, .78),
           (.16, .14, .15), (.13, .11, .11), (.55, .84, .87), (.06, .05, .05),
           (.98, .78, .15), (.92, .38, .32), (.45, .45, .48), (.82, .80, .77))
MATERIAL_IDS = {
    "hips": 1, "torso": 1, "neck": 0, "collar": 1, "zipper": 9,
    "head": 0, "haircap": 4, "eye_L": 6, "eye_R": 6, "smile": 6,
    "ponytail": 4, "ponytail_end": 4,
    "thigh_L": 2, "thigh_R": 2, "knee_L": 2, "knee_R": 2,
    "shin_L": 0, "shin_R": 0, "shoe_L": 3, "shoe_R": 3,
    "charm_L1": 7, "charm_L2": 8, "charm_L3": 7,
    "charm_R1": 7, "charm_R2": 8, "charm_R3": 7,
    "upperarm_L": 1, "upperarm_R": 1, "forearm_L": 1, "forearm_R": 1,
    "hands": 0, "wristband": 5, "base": 10,
}


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _rotx(p, angle):
    """Rotate points about X by angle (radians), right-hand rule."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack((x, c * y - s * z, s * y + c * z), axis=-1)


def _sphere(p, c, r):
    return np.linalg.norm(p - np.asarray(c), axis=-1) - r


def _ellipsoid(p, c, radii):
    """Conservative 1-Lipschitz distance-bound, finite at the center."""
    r = np.asarray(radii)
    return (np.linalg.norm((p - np.asarray(c)) / r, axis=-1) - 1.) * float(r.min())


def _rounded_box(p, c, half, rounding):
    """Exact rounded-box SDF; half is the OUTER extent."""
    half = np.asarray(half, dtype=float)
    c = np.asarray(c, dtype=float)
    if (half.shape != (3,) or c.shape != (3,) or not np.isfinite(half).all()
            or not np.isfinite(c).all() or not np.isfinite(rounding)
            or rounding <= 0 or rounding > half.min()):
        raise ValueError("rounded box needs finite half extents and 0 < rounding <= min(half)")
    q = np.abs(p - c) - (half - rounding)
    return (np.linalg.norm(np.maximum(q, 0), axis=-1)
            + np.minimum(q.max(axis=-1), 0) - rounding)


def _capsule(p, a, b, r):
    a, b = np.asarray(a), np.asarray(b)
    pa, ba = p - a, b - a
    denom = float(np.dot(ba, ba))
    if denom <= 0:
        raise ValueError("capsule needs distinct endpoints")
    h = np.clip(np.sum(pa * ba, axis=-1) / denom, 0., 1.)[..., None]
    return np.linalg.norm(pa - ba * h, axis=-1) - r


def _cylinder(p, c, r, hh):
    """Y-axis capped cylinder SDF; exact and 1-Lipschitz."""
    c = np.asarray(c, dtype=float)
    if (not np.isfinite(c).all() or not np.isfinite(r) or not np.isfinite(hh)
            or r <= 0 or hh <= 0):
        raise ValueError("cylinder needs finite center and positive radius/half-height")
    q = p - c
    d = np.stack((np.linalg.norm(q[..., [0, 2]], axis=-1) - r,
                  np.abs(q[..., 1]) - hh), axis=-1)
    return (np.minimum(d.max(axis=-1), 0.0)
            + np.linalg.norm(np.maximum(d, 0.0), axis=-1))


def _torus_arc(q, R, r, a0, a1):
    """Capped-arc distance of pre-transformed points, -pi <= a0 < a1 <= pi."""
    mid = .5 * (a0 + a1)
    ang = mid + (np.arctan2(q[..., 1], q[..., 0]) - mid + np.pi) % (2 * np.pi) - np.pi
    ang = np.clip(ang, a0, a1)
    return np.sqrt((q[..., 0] - R * np.cos(ang)) ** 2
                   + (q[..., 1] - R * np.sin(ang)) ** 2 + q[..., 2] ** 2) - r


def _rx_frame(p, pivot, angle, local_center):
    return _rotx(p - np.asarray(pivot), -angle) - np.asarray(local_center)


def _points(points):
    p = np.asarray(points)
    if (p.ndim < 1 or p.shape[-1] != 3
            or p.dtype.kind not in "fiu" or not np.isfinite(p).all()):
        raise ValueError("points must be a finite real array with shape (..., 3)")
    return p.astype(float, copy=False)


def _primitive(p, record):
    _name, kind, pivot, parameters, _ = record
    if kind == "sphere":
        return _sphere(p, pivot, parameters[0])
    if kind == "ellipsoid":
        return _ellipsoid(p, pivot, parameters[0])
    if kind == "capsule":
        return _capsule(p, pivot, *parameters)
    if kind == "cylinder":
        return _cylinder(p, pivot, *parameters)
    if kind == "rounded_box":
        return _rounded_box(p, pivot, *parameters)
    if kind == "rx_ellipsoid":
        angle, lc, radii = parameters
        return _ellipsoid(_rx_frame(p, pivot, angle, lc), (0., 0., 0.), radii)
    if kind == "rx_sphere":
        angle, lc, r = parameters
        q = _rx_frame(p, pivot, angle, lc)
        return np.linalg.norm(q, axis=-1) - r
    if kind == "rx_torus_arc":
        angle, lc, R, r, (a0, a1) = parameters
        return _torus_arc(_rx_frame(p, pivot, angle, lc), R, r, a0, a1)
    raise ValueError(f"unknown primitive kind: {kind}")


def primitive_fields(points):
    """Unsmoothed per-primitive fields in SCENE order, shape (..., N)."""
    p = _points(points)
    return np.stack([_primitive(p, record) for record in SCENE], axis=-1)


def sdf(points):
    """Signed distance-bound in mm: finite real points (..., 3) -> (...,)."""
    p = _points(points)
    d = _primitive(p, SCENE[0])
    for record in SCENE[1:]:
        value, blend = _primitive(p, record), record[-1]
        d = _smin(d, value, blend) if blend else np.minimum(d, value)
    return d


def material(points):
    """Diagnostic float RGB (..., 3) in [0,1]; no colors exported to STL."""
    nearest = np.argmin(primitive_fields(points), axis=-1)
    ids = np.asarray([MATERIAL_IDS[name] for name, *_ in SCENE])[nearest]
    return np.asarray(PALETTE)[ids]


def _abs_rotx_half(half, angle):
    """World AABB half-extents of a head-frame box with local half-extents."""
    c, s = abs(np.cos(angle)), abs(np.sin(angle))
    hx, hy, hz = half
    return np.array([hx, c * hy + s * hz, s * hy + c * hz])


def _scene_bounds():
    # smin >= min(a,b)-k/4: any final negative point has some primitive field
    # <= sum(k/4) from its insertion onward. Enclose those sublevel sets
    # analytically, plus 3 mm domain margin. rX kinds use the conservative
    # world AABB of the rotated local box.
    boxes, budget = [], 0.
    for _, kind, pivot, parameters, blend in reversed(SCENE):
        budget += blend / 4
        c = np.asarray(pivot, dtype=float)
        if kind == "ellipsoid":
            half = np.asarray(parameters[0]) + budget
        elif kind == "sphere":
            half = np.full(3, parameters[0] + budget)
        elif kind == "rounded_box":
            half = np.asarray(parameters[0]) + budget
        elif kind == "capsule":
            end, radius = parameters
            end = np.asarray(end, dtype=float)
            boxes.append((np.minimum(c, end) - radius - budget,
                          np.maximum(c, end) + radius + budget))
            continue
        elif kind == "cylinder":
            radius, hh = parameters
            half = np.array([radius, hh, radius]) + budget
        elif kind == "rx_ellipsoid":
            angle, lc, radii = parameters
            half = _abs_rotx_half(np.abs(lc) + np.asarray(radii), angle) + budget
        elif kind == "rx_sphere":
            angle, lc, r = parameters
            half = _abs_rotx_half(np.abs(lc) + r, angle) + budget
        elif kind == "rx_torus_arc":
            angle, lc, R, r, _ = parameters
            half = _abs_rotx_half(np.abs(lc) + np.array([R + r, R + r, r]), angle) + budget
        else:
            raise ValueError(f"unknown kind in bounds: {kind}")
        boxes.append((c - half, c + half))
    boxes = np.asarray(boxes)
    return (tuple(boxes[:, 0].min(axis=0) - 3), tuple(boxes[:, 1].max(axis=0) + 3))


BOUNDS = _scene_bounds()


def _glsl_literal(value):
    if isinstance(value, tuple):
        return f"vec{len(value)}(" + ", ".join(_glsl_literal(v) for v in value) + ")"
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


_GLSL_HELPERS = """vec3 rotX(vec3 q, float a) {
    float c = cos(a), s = sin(a);
    return vec3(q.x, c*q.y - s*q.z, s*q.y + c*q.z);
}
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}
float sdSphere(vec3 p, vec3 c, float r) { return length(p-c)-r; }
float sdEllipsoid(vec3 p, vec3 c, vec3 r) {
    return (length((p-c)/r)-1.0)*min(r.x,min(r.y,r.z));
}
float sdRoundedBox(vec3 p, vec3 c, vec3 halfExtent, float rounding) {
    vec3 q = abs(p-c) - (halfExtent-vec3(rounding));
    return length(max(q,vec3(0.0))) + min(max(q.x,max(q.y,q.z)),0.0) - rounding;
}
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p-a, ba = b-a;
    float h = clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0);
    return length(pa-ba*h)-r;
}
float sdCylinder(vec3 p, vec3 c, float r, float hh) {
    vec2 d = abs(vec2(length(p.xz-c.xz), p.y-c.y)) - vec2(r, hh);
    return min(max(d.x,d.y),0.0) + length(max(d,vec2(0.0)));
}
float sdRXEllipsoid(vec3 p, vec3 pivot, float ang, vec3 lc, vec3 r) {
    vec3 q = rotX(p-pivot, -ang) - lc;
    return (length(q/r)-1.0)*min(r.x,min(r.y,r.z));
}
float sdRXSphere(vec3 p, vec3 pivot, float ang, vec3 lc, float r) {
    return length(rotX(p-pivot, -ang) - lc) - r;
}
float sdRXTorusArc(vec3 p, vec3 pivot, float ang, vec3 lc,
                   float R, float r, vec2 arc) {
    vec3 q = rotX(p-pivot, -ang) - lc;
    float mid = 0.5*(arc.x+arc.y);
    float a = mid + mod(atan(q.y,q.x)-mid+3.141592653589793,
                        6.283185307179586)-3.141592653589793;
    a = clamp(a,arc.x,arc.y);
    return length(q-vec3(R*cos(a),R*sin(a),0.0))-r;
}
"""


def _scene_glsl():
    functions = {"sphere": "sdSphere", "ellipsoid": "sdEllipsoid",
                 "rounded_box": "sdRoundedBox", "capsule": "sdCapsule",
                 "cylinder": "sdCylinder",
                 "rx_ellipsoid": "sdRXEllipsoid", "rx_sphere": "sdRXSphere",
                 "rx_torus_arc": "sdRXTorusArc"}
    lines = [_GLSL_HELPERS, "float map(vec3 p) {"]
    for i, (name, kind, pivot, parameters, blend) in enumerate(SCENE):
        args = ", ".join(_glsl_literal(v) for v in (pivot, *parameters))
        value = f"{functions[kind]}(p, {args})"
        if i == 0:
            expression = f"float d = {value};"
        elif blend:
            expression = f"d = smin(d, {value}, {_glsl_literal(blend)});"
        else:
            expression = f"d = min(d, {value});"
        lines.append(f"    {expression} // {name}")
    return "\n".join([*lines, "    return d;", "}", ""])


GLSL = _scene_glsl()


def _checkpoints():
    pts = {
        "head center": (np.array([HEAD_PIVOT]), -1),
        "torso center": (np.array([[0., 40., -12.]]), -1),
        "knee_L": (np.array([[-15., 48., 26.]]), -1),
        "shoe_R inside": (np.array([[14., 8., 42.]]), -1),
        "hands center": (np.array([[0., 27., 40.]]), -1),
        "collar": (np.array([[0., 69., -13.]]), -1),
        "zipper mid": (np.array([[0., 45., 5.78]]), -1),
        "charm_L1": (np.array([[-21., 14.0, 49.]]), -1),
        "base center": (np.array([[0., -2., 14.]]), -1),
        "base below": (np.array([[0., -10., 14.]]), 1),
        "far above": (np.array([[0., 400., 0.]]), 1),
        "far side": (np.array([[400., 50., 0.]]), 1),
        "behind": (np.array([[0., 50., -400.]]), 1),
    }
    for label, (p, sign) in pts.items():
        v = float(sdf(p)[0])
        assert np.isfinite(v), label
        assert (v < 0) if sign < 0 else (v > 0), f"{label}: {v}"
    # Lipschitz spot check: |f(p)-f(q)| <= |p-q| on random pairs.
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(BOUNDS)
    a = rng.uniform(lo, hi, size=(2000, 3))
    b = a + rng.normal(0, 2.0, size=(2000, 3))
    fa, fb = sdf(a), sdf(b)
    ratio = np.abs(fa - fb) / np.maximum(np.linalg.norm(a - b, axis=1), 1e-9)
    assert np.isfinite(ratio).all() and ratio.max() <= 1.0 + 1e-6, ratio.max()
    # GLSL sanity: balanced braces and a map signature.
    assert GLSL.count("{") == GLSL.count("}")
    assert "float map(vec3 p)" in GLSL
    print(f"checkpoints ok; lipschitz spot max ratio {ratio.max():.4f}; "
          f"bounds {BOUNDS}")


if __name__ == "__main__":
    _checkpoints()
