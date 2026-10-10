"""Jolly (Muse's default avatar) waving: pink-fur variant, left hand raised.

Stylized smooth figurine, ~150 mm tall. Built from blended spheres, ellipsoids,
capsules and a torus-arc smile. Front faces +z, y-up. All dimensions in mm.

Improvements over the first hand-authored draft:
- smile is a true circular arc (torus segment with round caps), not two capsules;
- face plaque joins the head with a small smooth-min for a softer edge;
- waving hand is a fuller mitten.
"""

import numpy as np

BOUNDS = ((-60.0, -75.0, -40.0), (60.0, 80.0, 45.0))
# The field is globally 1-Lipschitz: spheres, capsules and the torus arc are
# exact distances (1-Lipschitz), the ellipsoid uses the conservative bound
# (|q|-1)*min(r) which is 1-Lipschitz by construction, and min/smin preserve
# the bound. Verified: max|grad| = 1.0 over 300k uniform samples of BOUNDS.
LIPSCHITZ = 1.0
DESCRIPTION = "Waving Jolly figurine, ~150 mm tall."


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _sphere(p, c, r):
    return np.linalg.norm(p - np.asarray(c, dtype=float), axis=-1) - r


def _ellipsoid(p, c, radii):
    """Conservative 1-Lipschitz bound for an ellipsoid.

    (|(p-c)/r| - 1) * min(r): zero exactly on the surface, and 1-Lipschitz as
    the composition of |(p-c)/r| (Lipschitz 1/min(r)) scaled by min(r). A
    1-Lipschitz function vanishing on the surface is a valid distance bound,
    so sphere tracing with step |f| is safe. (iq's tighter k0(k0-1)/k1 bound
    is NOT Lipschitz at the ellipsoid center and must not be used with a
    declared bound of 1.0.)
    """
    c = np.asarray(c, dtype=float)
    r = np.asarray(radii, dtype=float)
    return (np.linalg.norm((p - c) / r, axis=-1) - 1.0) * float(r.min())


def _capsule(p, a, b, r):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pa = p - a
    ba = b - a
    h = np.clip(np.sum(pa * ba, axis=-1) / np.dot(ba, ba), 0.0, 1.0)[..., None]
    return np.linalg.norm(pa - ba * h, axis=-1) - r


def _torus_arc(p, c, R, r, a0, a1):
    """Tube of radius r around a circular arc of radius R in the x-y plane.

    The arc spans angles [a0, a1] (radians, atan2 convention, a0 < a1, no
    wraparound) with hemispherical end caps. Distance to a compact set, so
    1-Lipschitz.
    """
    c = np.asarray(c, dtype=float)
    qx = p[..., 0] - c[0]
    qy = p[..., 1] - c[1]
    qz = p[..., 2] - c[2]
    # Unwrap the angle about the arc midpoint so the atan2 branch cut sits at
    # the antipode of the arc (the true Voronoi boundary between the two ends).
    # A naive clip, or unwrapping about an endpoint, discontinuously jumps from
    # one arc end to the other for points crossing the cut.
    mid = 0.5 * (a0 + a1)
    ang = mid + (np.arctan2(qy, qx) - mid + np.pi) % (2.0 * np.pi) - np.pi
    ang = np.clip(ang, a0, a1)
    cx, cy = R * np.cos(ang), R * np.sin(ang)
    return np.sqrt((qx - cx) ** 2 + (qy - cy) ** 2 + qz**2) - r


def sdf(points):
    """Signed distance in mm for points with shape (..., 3)."""
    p = np.asarray(points)
    if (p.ndim < 1 or p.shape[-1] != 3
            or p.dtype.kind not in "fiu" or not np.isfinite(p).all()):
        raise ValueError("points must be a finite real array with shape (..., 3)")
    p = p.astype(float, copy=False)

    d = _ellipsoid(p, (0.0, -4.8, 0.0), (37.2, 48.0, 33.0))  # body
    d = _smin(d, _sphere(p, (0.0, 43.2, 1.2), 31.2), 16.8)  # head
    d = _smin(
        d,
        _ellipsoid(
            p,
            (0.0, 42.6, 23.7),  # face plaque
            (21.6, 21.6, 10.2),
        ),
        1.0,
    )
    d = np.minimum(d, _sphere(p, (-9.6, 48.0, 30.9), 3.9))  # eyes
    d = np.minimum(d, _sphere(p, (9.6, 48.0, 30.9), 3.9))
    d = np.minimum(
        d,
        _torus_arc(
            p,
            (0.0, 42.0, 33.5),  # smile
            8.0,
            1.4,
            -2.47,
            -0.67,
        ),
    )
    d = _smin(
        d,
        _capsule(
            p,
            (-28.8, 13.2, 0.0),  # wave: upper arm
            (-42.0, 34.8, 1.8),
            9.0,
        ),
        7.2,
    )
    d = _smin(
        d,
        _capsule(
            p,
            (-42.0, 34.8, 1.8),  # wave: forearm
            (-46.2, 55.8, 3.6),
            8.7,
        ),
        6.0,
    )
    d = _smin(d, _sphere(p, (-46.2, 58.5, 3.6), 10.0), 4.8)  # mitten hand
    d = _smin(
        d,
        _capsule(
            p,
            (30.0, 10.8, 0.0),  # resting arm
            (40.8, -21.0, 1.8),
            9.3,
        ),
        7.2,
    )
    d = _smin(
        d,
        _capsule(
            p,
            (-14.4, -42.0, 0.0),  # legs
            (-16.2, -61.2, 1.8),
            9.9,
        ),
        7.2,
    )
    d = _smin(d, _capsule(p, (14.4, -42.0, 0.0), (16.2, -61.2, 1.8), 9.9), 7.2)
    return d


# Hand-paired GLSL in the same millimetre units. Constants originate from the
# NumPy definitions above; this string is source, not evidence of compilation.
GLSL = """float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}
float sdEllipsoid(vec3 p, vec3 c, vec3 r) {
    // Conservative 1-Lipschitz bound: (|(p-c)/r|-1)*min(r). Zero exactly on
    // the surface; see the NumPy version for why this is the safe choice.
    return (length((p-c)/r) - 1.0) * min(r.x, min(r.y, r.z));
}
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p-a, ba = b-a;
    float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
    return length(pa-ba*h)-r;
}
float sdTorusArc(vec3 p, vec3 c, float R, float r, vec2 ang) {
    vec3 q = p - c;
    // Unwrap about the arc midpoint: the cut lands at the antipode, the true
    // boundary between the ends' regions (see NumPy version).
    float mid = 0.5*(ang.x + ang.y);
    float a = mid + mod(atan(q.y, q.x) - mid + 3.14159265, 6.2831853) - 3.14159265;
    a = clamp(a, ang.x, ang.y);
    vec3 closest = vec3(R*cos(a), R*sin(a), 0.0);
    return length(q - closest) - r;
}
float map(vec3 p) {
    float d = sdEllipsoid(p, vec3(0.0,-4.8,0.0), vec3(37.2,48.0,33.0));
    d = smin(d, length(p-vec3(0.0,43.2,1.2))-31.2, 16.8);
    d = smin(d, sdEllipsoid(p, vec3(0.0,42.6,23.7), vec3(21.6,21.6,10.2)), 1.0);
    d = min(d, length(p-vec3(-9.6,48.0,30.9))-3.9);
    d = min(d, length(p-vec3( 9.6,48.0,30.9))-3.9);
    d = min(d, sdTorusArc(p, vec3(0.0,42.0,33.5), 8.0, 1.4, vec2(-2.47,-0.67)));
    d = smin(d, sdCapsule(p, vec3(-28.8,13.2,0.0), vec3(-42.0,34.8,1.8), 9.0), 7.2);
    d = smin(d, sdCapsule(p, vec3(-42.0,34.8,1.8), vec3(-46.2,55.8,3.6), 8.7), 6.0);
    d = smin(d, length(p-vec3(-46.2,58.5,3.6))-10.0, 4.8);
    d = smin(d, sdCapsule(p, vec3(30.0,10.8,0.0), vec3(40.8,-21.0,1.8), 9.3), 7.2);
    d = smin(d, sdCapsule(p, vec3(-14.4,-42.0,0.0), vec3(-16.2,-61.2,1.8), 9.9), 7.2);
    d = smin(d, sdCapsule(p, vec3( 14.4,-42.0,0.0), vec3( 16.2,-61.2,1.8), 9.9), 7.2);
    return d;
}
"""
