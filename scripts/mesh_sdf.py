"""Millimetre scalar-field meshing (negative inside, positive outside).

Scalar fields need not be true signed distances or distance bounds. Positive
sampled boundaries are necessary, not proof that the global model is complete.
The report checks mesh integrity and declared sampling, not self-intersections,
minimum wall thickness, printer/material tolerances or physical printability.
"""
import inspect
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def _sampling_report(pitch_mm, feature):
    if feature is not None:
        if (isinstance(feature, (bool, np.bool_))
                or not isinstance(feature, (int, float, np.integer, np.floating))
                or not np.isfinite(feature) or feature <= 0):
            raise ValueError("declared minimum feature must be finite positive mm")
        feature = float(feature)
    pitch = None
    if pitch_mm is not None:
        raw = np.asarray(pitch_mm)
        if raw.dtype.kind not in "fiu" or raw.shape not in ((), (3,)):
            raise ValueError("pitch must be a positive finite scalar or length-3 mm vector")
        pitch = np.atleast_1d(raw).astype(float)
        if not np.isfinite(pitch).all() or np.any(pitch <= 0):
            raise ValueError("pitch must be positive finite mm")
    met = None
    if feature is not None and pitch is not None:
        met = bool(pitch.max() <= feature/3)
        if not met:
            raise ValueError("under-resolved: max grid pitch must be <= declared min feature / 3")
    return {"pitch_mm": None if pitch is None else pitch.tolist(),
            "max_pitch_mm": None if pitch is None else float(pitch.max()),
            "declared_min_feature_mm": feature, "sampling_requirement_met": met}


def validate_mesh(mesh, *, pitch_mm=None, declared_min_feature_mm=None,
                  allow_multiple_components=False):
    """Return mesh integrity only, not physical printability certification."""
    if not isinstance(allow_multiple_components, (bool, np.bool_)):
        raise ValueError("allow_multiple_components must be boolean")
    sampling = _sampling_report(pitch_mm, declared_min_feature_mm)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("mesh must be a trimesh.Trimesh")
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)):
        raise ValueError("mesh must contain vertices and triangular faces")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh geometry must be finite")
    if faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("invalid face vertex indices")
    dimensions = mesh.extents
    if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("mesh dimensions must be finite and positive in mm")
    if not np.isfinite(mesh.area_faces).all() or np.any(mesh.area_faces <= 0):
        raise ValueError("nonfinite or degenerate triangles")
    if not mesh.unique_faces().all():
        raise ValueError("duplicate triangles")
    if not mesh.is_watertight:
        raise ValueError("mesh must be watertight")
    if not mesh.is_winding_consistent:
        raise ValueError("mesh must have consistent winding")
    components = mesh.split(only_watertight=False, repair=False)
    if len(components) != 1 and not allow_multiple_components:
        raise ValueError("multiple components require explicit opt-in")
    if any(not np.isfinite(part.volume) or part.volume <= 0 for part in components):
        raise ValueError("each component must have finite positive volume")
    return {
        **sampling,
        "dimensions_mm": mesh.extents.tolist(),
        "triangle_count": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "positive_volume": bool(mesh.volume > 0),
        "component_count": len(components),
        "physical_printability_verified": False,
        "wall_thickness_verified": False,
    }


def make_mesh(sdf, *, bounds=((-12, -12, -12), (12, 12, 12)),
              resolution=64, min_feature_mm=None, allow_multiple_components=False):
    """Sample sdf(points[..., 3]) by x-slice in unchanged mm coordinates.

    Each points slice has shape (N, N, 3), with a real finite (N, N) result.
    Explicit three-argument sdf(x, y, z) callbacks are also supported, with
    scalar x and broadcastable y/z of shapes (N, 1)/(1, N). Signature binding
    selects that legacy contract without catching exceptions inside the field.
    Only the scalar volume is dense; no dense XYZ coordinate volume is built.
    """
    if not callable(sdf):
        raise ValueError("sdf must be callable")
    if (isinstance(resolution, (bool, np.bool_))
            or not isinstance(resolution, (int, np.integer))
            or not 8 <= resolution <= 256):
        raise ValueError("resolution must be an integer in 8..256 (not bool)")
    try:
        bounds = np.asarray(bounds, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must be finite ordered (2, 3) coordinates in mm") from exc
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("bounds must be finite ordered (2, 3) coordinates in mm")
    lo, hi = bounds
    if not np.all(hi > lo):
        raise ValueError("each upper bound must exceed its lower bound")
    pitch = (hi-lo)/(resolution-1)
    _sampling_report(pitch, min_feature_mm)
    axes = [np.linspace(a, b, resolution) for a, b in zip(lo, hi)]
    xyz_callback = False
    try:
        signature = inspect.signature(sdf)
    except (ValueError, TypeError):
        # Uninspectable callables use the documented points-array contract.
        pass
    else:
        try:
            signature.bind(None)
        except TypeError:
            signature.bind(None, None, None)
            xyz_callback = True
    points = None
    if not xyz_callback:
        points = np.empty((resolution, resolution, 3), dtype=float)
        points[..., 1] = axes[1][:, None]
        points[..., 2] = axes[2][None, :]
    field = np.empty((resolution,) * 3, dtype=np.float32)
    for i, x in enumerate(axes[0]):
        if xyz_callback:
            values = sdf(x, axes[1][:, None], axes[2][None, :])
        else:
            assert points is not None
            points[..., 0] = x
            values = sdf(points)
        sample = np.asarray(values)
        if sample.shape != (resolution, resolution):
            raise ValueError("scalar field must return slice shape (resolution, resolution)")
        if sample.dtype.kind not in "fiu":
            raise ValueError("scalar field must contain real numeric values")
        if not np.isfinite(sample).all():
            raise ValueError("scalar field must be finite")
        if np.any(np.abs(sample) > np.finfo(np.float32).max):
            raise ValueError("scalar field must remain finite in marching-cubes float32")
        field[i] = sample
    if not field.min() < 0 < field.max():
        raise ValueError("scalar field must strictly cross zero: min < 0 < max")
    boundaries = (field[0], field[-1], field[:, 0], field[:, -1],
                  field[:, :, 0], field[:, :, -1])
    if any(not np.all(side > 0) for side in boundaries):
        raise ValueError("all six sampled domain boundaries must be strictly positive")
    vertices, faces, _, _ = marching_cubes(
        field, level=0, spacing=tuple((hi-lo)/(resolution-1)),
        gradient_direction="descent", allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices=vertices + lo, faces=faces, process=False)
    validate_mesh(mesh, allow_multiple_components=allow_multiple_components)
    return mesh


def export_stl(mesh, path, *, allow_multiple_components=False):
    """Write binary STL in unchanged mm coordinates and verify the reload."""
    before = validate_mesh(mesh, allow_multiple_components=allow_multiple_components)
    path = Path(path)
    path.write_bytes(trimesh.exchange.stl.export_stl(mesh))
    loaded = trimesh.load_mesh(path, file_type="stl", process=True,
                               merge_norm=True, merge_tex=True)
    after = validate_mesh(loaded, allow_multiple_components=allow_multiple_components)
    if after["triangle_count"] != before["triangle_count"]:
        raise ValueError("STL reload changed triangle count")
    if not np.allclose(after["dimensions_mm"], before["dimensions_mm"],
                       rtol=1e-6, atol=1e-5):
        raise ValueError("STL reload changed dimensions in mm")
    return {"path": str(path), "format": "binary_stl", "reload_verified": True,
            **after}
