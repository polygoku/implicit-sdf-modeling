"""Smooth Jolly reference figurine: viewer-right wave, front +Z, Y-up, mm.

Signed distance-bound, not exact signed distance. Sphere, capsule, rounded-box,
rounded-prism and capped arc distances are 1-Lipschitz. Ellipsoids use (|(p-c)/r|-1)*min(r):
the diagonal norm map has bound 1/min(r), cancelled by scaling. Hard min
preserves the bound; polynomial smooth-min has nonnegative partial derivatives
summing to one, so its gradient is a convex combination of input gradients,
including piecewise boundaries. Thus the field is globally 1-Lipschitz;
finite samples are regression checks, not proof.

SCENE records: (name, kind, center/start, parameters, blend_mm), union order.
Parameters: (radii,), (outer_half, rounding), (outer_half, corner, edge),
(end, radius), (radius,), or
(major_radius, tube_radius, angle_pair). NumPy and GLSL share these records.
primitive_fields(points) -> (..., len(SCENE)); material(points) -> float RGB
(..., 3), approximate nearest-primitive diagnostic colors, not blend-aware
textures or STL colors. Both use the same finite-real input contract as sdf.
"""
import numpy as np

LIPSCHITZ = 1.0
DESCRIPTION = "Reference-refined waving Jolly figurine, approximately 152 mm tall."
SCENE = (
    ("body", "ellipsoid", (0., -13., 0.), ((40., 46., 30.),), 0.),
    ("hood", "rounded_prism", (0., 42., 0.), ((38., 36., 32.), 26., 12.), 8.),
    ("face", "rounded_prism", (0., 43., 27.), ((28., 21., 7.), 17.5, 2.), 1.),
    ("left_eye", "sphere", (-14.5, 47., 33.5), (2.8,), 0.),
    ("right_eye", "sphere", (14.5, 47., 33.5), (2.8,), 0.),
    ("smile", "torus_arc", (0., 45., 34.3), (5., .65, (-2.47, -.67)), 0.),
    ("wave_upper", "capsule", (33., 15., 0.), ((47., 27., 2.), 10.), 4.),
    ("wave_forearm", "capsule", (47., 27., 2.), ((59., 39., 3.), 10.), 3.),
    ("wave_mitten", "ellipsoid", (61., 40., 3.), ((10., 11., 9.5),), 2.),
    ("hanging_upper", "capsule", (-33., 12., 0.), ((-47., -25., 2.), 9.5), 4.),
    ("hanging_forearm", "capsule", (-47., -25., 2.), ((-54., -42., 3.), 9.5), 3.),
    ("left_foot", "rounded_box", (-17., -61., 2.), ((12.5, 13., 18.), 5.), 3.),
    ("right_foot", "rounded_box", (17., -61., 2.), ((12.5, 13., 18.), 5.), 3.),
)
PALETTE = ((.86, .65, .66), (.98, .87, .79), (.055, .04, .045))
MATERIAL_IDS = tuple(1 if name == "face" else 2 if "eye" in name or name == "smile"
                     else 0 for name, *_ in SCENE)


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _sphere(p, c, r):
    return np.linalg.norm(p - np.asarray(c), axis=-1) - r


def _ellipsoid(p, c, radii):
    """Conservative 1-Lipschitz distance-bound, finite at the center."""
    r = np.asarray(radii)
    return (np.linalg.norm((p - np.asarray(c)) / r, axis=-1) - 1.) * float(r.min())


def _rounded_box(p, c, half, rounding):
    """Exact rounded-box SDF; half is the OUTER extent, not core extent."""
    half, c = np.asarray(half, dtype=float), np.asarray(c, dtype=float)
    if (half.shape != (3,) or c.shape != (3,) or not np.isfinite(half).all()
            or not np.isfinite(c).all() or not np.isfinite(rounding)
            or rounding <= 0 or rounding > half.min()):
        raise ValueError("rounded box needs finite half extents and 0 < rounding <= min(half)")
    q = np.abs(p - c) - (half - rounding)
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0) - rounding


def _rounded_prism(p, c, half, corner, edge):
    """Exact rounded XY prism with independent Z fillet; half is OUTER extent.

    The core is a rounded 2D rectangle times a Z interval, offset by edge.
    Its product SDF combines dxy and dz by outside norm / inside maximum.
    Their unit gradients occupy orthogonal XY and Z subspaces: outside the
    weights have squared sum one; inside one unit gradient is selected.
    At seams the limiting gradients' convex hull has norm <= 1. Subtracting
    the constant offset preserves this global 1-Lipschitz bound.
    """
    half, c = np.asarray(half, dtype=float), np.asarray(c, dtype=float)
    if (half.shape != (3,) or c.shape != (3,) or not np.isfinite(half).all()
            or not np.isfinite(c).all() or np.any(half <= 0)
            or not np.isfinite(corner) or not np.isfinite(edge)
            or not 0 < corner <= half[:2].min()
            or not 0 < edge < min(half[2], corner)
            or np.any(half - edge <= 0)):
        raise ValueError("rounded prism needs finite positive half extents, "
                         "0 < corner <= min(half.xy), 0 < edge < min(half.z, corner)")
    core_corner = corner - edge
    q = np.abs(p[..., :2] - c[:2]) - (half[:2] - edge - core_corner)
    dxy = (np.linalg.norm(np.maximum(q, 0), axis=-1)
           + np.minimum(q.max(axis=-1), 0) - core_corner)
    dz = np.abs(p[..., 2] - c[2]) - (half[2] - edge)
    d = np.stack((dxy, dz), axis=-1)
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(axis=-1), 0) - edge


def _capsule(p, a, b, r):
    a, b = np.asarray(a), np.asarray(b)
    pa, ba = p - a, b - a
    h = np.clip(np.sum(pa * ba, axis=-1) / np.dot(ba, ba), 0., 1.)[..., None]
    return np.linalg.norm(pa - ba * h, axis=-1) - r


def _torus_arc(p, c, R, r, a0, a1):
    """Exact capped-arc distance, -pi <= a0 < a1 <= pi, span < full circle.

    Midpoint unwrapping puts the cut at the true endpoint Voronoi boundary;
    naive atan2 clipping is discontinuous. Axis points are equidistant to
    every arc point, including NumPy atan2(0,0).
    """
    q = p - np.asarray(c)
    mid = .5 * (a0 + a1)
    ang = mid + (np.arctan2(q[..., 1], q[..., 0]) - mid + np.pi) % (2 * np.pi) - np.pi
    ang = np.clip(ang, a0, a1)
    return np.sqrt((q[..., 0] - R * np.cos(ang)) ** 2
                   + (q[..., 1] - R * np.sin(ang)) ** 2 + q[..., 2] ** 2) - r


def _points(points):
    p = np.asarray(points)
    if (p.ndim < 1 or p.shape[-1] != 3
            or p.dtype.kind not in "fiu" or not np.isfinite(p).all()):
        raise ValueError("points must be a finite real array with shape (..., 3)")
    return p.astype(float, copy=False)


def _primitive(p, record):
    _, kind, center, parameters, _ = record
    if kind == "torus_arc":
        R, r, angles = parameters
        return _torus_arc(p, center, R, r, *angles)
    return {"sphere": _sphere, "ellipsoid": _ellipsoid,
            "capsule": _capsule, "rounded_box": _rounded_box,
            "rounded_prism": _rounded_prism}[kind](p, center, *parameters)


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
    return np.asarray(PALETTE)[np.asarray(MATERIAL_IDS)[nearest]]


def _scene_bounds():
    # smin >= min(a,b)-k/4: any final negative point has some primitive field
    # <= sum(k/4) from its insertion onward. Analytically enclose those
    # sublevel sets, including ellipsoid scaling, plus 2 mm domain margin.
    boxes, budget = [], 0.
    for _, kind, center, parameters, blend in reversed(SCENE):
        budget += blend / 4
        c = np.asarray(center)
        if kind == "ellipsoid":
            radii = np.asarray(parameters[0])
            half = radii * (1 + budget / radii.min())
        elif kind in ("rounded_box", "rounded_prism"):
            half = np.asarray(parameters[0]) + budget
        elif kind == "sphere":
            half = np.full(3, parameters[0] + budget)
        elif kind == "torus_arc":
            R, r, _ = parameters
            half = np.array([R + r, R + r, r]) + budget
        else:
            end, radius = parameters
            boxes.append((np.minimum(c, end) - radius - budget,
                          np.maximum(c, end) + radius + budget))
            continue
        boxes.append((c - half, c + half))
    boxes = np.asarray(boxes)
    return (tuple(boxes[:, 0].min(axis=0) - 2), tuple(boxes[:, 1].max(axis=0) + 2))


BOUNDS = _scene_bounds()


def _glsl_literal(value):
    if isinstance(value, tuple):
        return f"vec{len(value)}(" + ", ".join(_glsl_literal(v) for v in value) + ")"
    return f"{value:.8f}"


_GLSL_HELPERS = """float smin(float a, float b, float k) {
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
float sdRoundedPrism(vec3 p, vec3 c, vec3 halfExtent, float corner, float edge) {
    vec2 coreXY = halfExtent.xy - vec2(edge);
    float coreCorner = corner - edge;
    float coreHalfZ = halfExtent.z - edge;
    vec2 q = abs(p.xy-c.xy) - (coreXY-vec2(coreCorner));
    float dxy = length(max(q,vec2(0.0))) + min(max(q.x,q.y),0.0) - coreCorner;
    float dz = abs(p.z-c.z) - coreHalfZ;
    vec2 d = vec2(dxy, dz);
    return length(max(d,vec2(0.0))) + min(max(d.x,d.y),0.0) - edge;
}
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p-a, ba = b-a;
    float h = clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0);
    return length(pa-ba*h)-r;
}
float sdTorusArc(vec3 p, vec3 c, float R, float r, vec2 ang) {
    vec3 q = p - c;
    if (q.x == 0.0 && q.y == 0.0) return length(vec2(R, q.z)) - r;
    float mid = 0.5*(ang.x+ang.y);
    float a = mid + mod(atan(q.y,q.x)-mid+3.141592653589793,
                        6.283185307179586)-3.141592653589793;
    a = clamp(a,ang.x,ang.y);
    return length(q-vec3(R*cos(a),R*sin(a),0.0))-r;
}
"""


def _scene_glsl():
    functions = {"sphere": "sdSphere", "ellipsoid": "sdEllipsoid",
                 "rounded_box": "sdRoundedBox", "capsule": "sdCapsule",
                 "torus_arc": "sdTorusArc", "rounded_prism": "sdRoundedPrism"}
    lines = [_GLSL_HELPERS, "float map(vec3 p) {"]
    for i, (name, kind, center, parameters, blend) in enumerate(SCENE):
        args = ", ".join(_glsl_literal(v) for v in (center, *parameters))
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
