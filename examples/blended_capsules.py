"""A smooth union of two crossing capsules, with all dimensions in mm.

The capsules intersect at the origin. Their polynomial smooth union is one
bounded solid; it is a conservative 1-Lipschitz distance *bound*, not an exact
Euclidean distance in the blend region. GLSL is hand-paired, not translated.
"""
import numpy as np

CAPSULES = (
    ((-9.0, -4.0, 0.0), (9.0, 4.0, 0.0), 3.0),
    ((-4.0, 9.0, 0.0), (4.0, -9.0, 0.0), 3.0),
)
BLEND_RADIUS = 2.5
BOUNDS = ((-14.0, -14.0, -6.0), (14.0, 14.0, 6.0))
LIPSCHITZ = 1.0
DESCRIPTION = "Two intersecting 3 mm radius capsules blended over 2.5 mm."


def capsule_sdf(points, a, b, radius):
    """Distance to a capped line segment with hemispherical ends."""
    points = np.asarray(points)
    if (points.ndim < 1 or points.shape[-1] != 3 or
            points.dtype.kind not in "fiu" or not np.isfinite(points).all()):
        raise ValueError("points must be a finite real array with shape (..., 3)")
    points = points.astype(float, copy=False)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (a.shape != (3,) or b.shape != (3,) or
            not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("capsule endpoints must be finite three-vectors")
    if np.ndim(radius) != 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("capsule radius must be finite and positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ba = b - a
        length_squared = np.dot(ba, ba)
    if not np.isfinite(length_squared) or length_squared <= 0:
        raise ValueError("capsule segment must have a finite nonzero squared length")
    pa = points - a
    h = np.clip(np.sum(pa * ba, axis=-1) / length_squared, 0.0, 1.0)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - radius


def smooth_min(a, b, k):
    """Polynomial smooth minimum with blend width k in distance units."""
    if np.ndim(k) != 0 or not np.isfinite(k) or k <= 0:
        raise ValueError("smooth-min blend radius must be finite and positive")
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def sdf(points):
    """Conservative signed distance bound for points with shape (..., 3)."""
    first = capsule_sdf(points, *CAPSULES[0])
    second = capsule_sdf(points, *CAPSULES[1])
    return smooth_min(first, second, BLEND_RADIUS)


# Constants are formatted from the exact same tuples used above.
_calls = []
for _a, _b, _r in CAPSULES:
    _va = "vec3(" + ", ".join(f"{x:.8f}" for x in _a) + ")"
    _vb = "vec3(" + ", ".join(f"{x:.8f}" for x in _b) + ")"
    _calls.append(f"sdCapsule(p, {_va}, {_vb}, {_r:.8f})")

GLSL = """float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    // Invalid helper arguments yield a positive sentinel, never a zero set.
    if (dot(ba, ba) <= 0.0 || r <= 0.0) return 1e20;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
float smoothMin(float a, float b, float k) {
    if (k <= 0.0) return 1e20;
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
""" + f"""float map(vec3 p) {{
    float a = {_calls[0]};
    float b = {_calls[1]};
    return smoothMin(a, b, {BLEND_RADIUS:.8f});
}}
"""
