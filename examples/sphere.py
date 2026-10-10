"""A 10 mm sphere: explicit NumPy implementation and matching GLSL source."""
import numpy as np

RADIUS = 10.0
BOUNDS = ((-12.0, -12.0, -12.0), (12.0, 12.0, 12.0))
LIPSCHITZ = 1.0
DESCRIPTION = "Sphere of radius 10 mm centered at the origin."


def sdf(points):
    """Signed distance in mm for points with shape (..., 3)."""
    points = np.asarray(points)
    if (points.ndim < 1 or points.shape[-1] != 3 or
            points.dtype.kind not in "fiu" or not np.isfinite(points).all()):
        raise ValueError("points must be a finite real array with shape (..., 3)")
    points = points.astype(float, copy=False)
    return np.linalg.norm(points, axis=-1) - RADIUS


# Hand-paired formulas; numeric constants originate here, not in duplicated GLSL.
# This string is source, not evidence of shader compilation or translation.
GLSL = f"""float map(vec3 p) {{
    return length(p) - {RADIUS:.8f};
}}
"""
