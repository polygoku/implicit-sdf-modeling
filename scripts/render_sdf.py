"""Vectorized CPU perspective renderer for an explicitly supplied NumPy SDF."""
from numbers import Integral, Real

import numpy as np


def _vector(value, name):
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite three-vector") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


def _unit(vector, name):
    # Scaling first avoids overflow for finite, very large coordinates.
    magnitude = np.max(np.abs(vector))
    if not np.isfinite(magnitude) or magnitude == 0:
        raise ValueError(f"{name} must have a finite nonzero direction")
    scaled = vector / magnitude
    return scaled / np.linalg.norm(scaled)


def _validate_options(width, height, max_steps, fov_degrees,
                      epsilon, max_distance, lipschitz):
    for name, value in (("width", width), ("height", height),
                        ("max_steps", max_steps)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("fov_degrees", fov_degrees), ("epsilon", epsilon),
                        ("max_distance", max_distance), ("lipschitz", lipschitz)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if fov_degrees >= 180:
        raise ValueError("fov_degrees must be less than 180")


def render(sdf, *, width=160, height=120, camera=(0, 0, 3),
           target=(0, 0, 0), up=(0, 1, 0), fov_degrees=45,
           epsilon=1e-3, max_steps=192, max_distance=100, lipschitz=1.0):
    """Return RGB, hit mask, and ray distances (infinity for misses).

    Pixel centers sample a vertical perspective FOV. Every ray starts at camera.
    The field's units, camera coordinates, and distance limits must agree.
    ``sdf`` receives active points of shape (N, 3) and must return finite real
    values of shape (N,). Invalid options or field values raise ValueError.
    Supply a positive conservative Lipschitz bound for the *supplied field*;
    underspecifying it can skip surfaces. Absolute field values permit marching
    from inside a solid to its exit. The far plane is inclusive, and exhausted
    rays are misses, even if more steps would have found an intersection.
    Hits have a flat color (no untested normal estimation); background is zero.
    """
    if not callable(sdf):
        raise ValueError("sdf must be callable")
    _validate_options(width, height, max_steps, fov_degrees,
                      epsilon, max_distance, lipschitz)
    origin = _vector(camera, "camera")
    target = _vector(target, "target")
    up = _unit(_vector(up, "up"), "up")
    with np.errstate(over="ignore", invalid="ignore"):
        forward = _unit(target - origin, "target - camera")
    right = _unit(np.cross(forward, up), "up cross view direction")
    vertical = np.cross(right, forward)
    scale = np.tan(np.deg2rad(fov_degrees) / 2)
    x = (2 * (np.arange(width) + 0.5) / width - 1) * scale * width / height
    y = (1 - 2 * (np.arange(height) + 0.5) / height) * scale
    xx, yy = np.meshgrid(x, y)
    directions = forward + xx[..., None] * right + yy[..., None] * vertical
    directions = directions.reshape(-1, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    travel = np.zeros(width * height, dtype=float)
    hit = np.zeros(width * height, dtype=bool)
    active = np.ones(width * height, dtype=bool)
    for _ in range(max_steps):
        indices = np.flatnonzero(active)
        if not indices.size:
            break
        points = origin + travel[indices, None] * directions[indices]
        if not np.all(np.isfinite(points)):
            raise ValueError("sdf sample coordinates must be finite")
        distances = np.asarray(sdf(points))
        if (distances.shape != (indices.size,) or
                distances.dtype.kind not in "fiu" or
                not np.all(np.isfinite(distances))):
            raise ValueError("sdf must return finite real distances with shape (N,)")
        distances = distances.astype(float, copy=False)
        reached = np.abs(distances) <= epsilon
        hit[indices[reached]] = True
        active[indices[reached]] = False
        remaining = indices[~reached]
        at_limit = travel[remaining] >= max_distance
        active[remaining[at_limit]] = False
        moving = remaining[~at_limit]
        # Overflow in a positive step simply means the step exceeds the far
        # plane. Cap it *before* addition; evaluate the far plane at most once.
        with np.errstate(over="ignore"):
            steps = np.abs(distances[~reached][~at_limit]) / lipschitz
        steps = np.minimum(steps, max_distance - travel[moving])
        travel[moving] += steps
    depths = np.where(hit, travel, np.inf).reshape(height, width)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    hit = hit.reshape(height, width)
    image[hit] = (180, 205, 230)
    return image, hit, depths
