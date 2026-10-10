"""Incremental TDD regression tests; execution evidence is in tool output."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

# Narrow upstream compatibility warning: skimage 0.26 reshapes via .shape.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning:skimage.measure._marching_cubes_lewiner")

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "mesh_sdf.py"

# TDD ledger (all RED/GREEN outputs recorded in execution transcript):
# 1 sphere: missing module RED; inward volume RED; corrected winding GREEN (1).
# 2 malformed bounds/resolution: 9 RED -> GREEN (13).
# 3 sampled scalar validity/domain boundaries: 15 RED -> GREEN (29).
# 4 JSON integrity report: missing validate_mesh RED -> GREEN (30).
# 5 topology/geometry/component validation: 11 RED -> GREEN (41).
# 6 meshing validates extracted components: 1 RED -> GREEN (42).
# 7 declared-feature sampling: 14 RED -> GREEN (56).
# 8 binary STL export/reload: missing export_stl RED -> GREEN (57).
# 9 malformed callback/opt-in: 3 RED -> GREEN (64); existing spacing,
#   scaled-field, slice allocation and pre-write validation regressions green.
# 10 binary STL multiple-component opt-in: missing keyword RED -> GREEN (65).


def api():
    assert MODULE.exists(), "meshing implementation is missing"
    spec = importlib.util.spec_from_file_location("mesh_sdf", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sphere(x, y, z):
    return np.sqrt(x*x + y*y + z*z) - 5


def test_point_array_field_contract_matches_renderer():
    def field(points):
        assert points.shape == (32, 32, 3)
        return np.linalg.norm(points, axis=-1) - 5
    mesh = api().make_mesh(field, resolution=32)
    assert mesh.is_watertight
    np.testing.assert_allclose(mesh.extents, [10, 10, 10], atol=.1)


def test_sphere_dimensions_outward_volume():
    mesh = api().make_mesh(sphere)
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume > 0
    np.testing.assert_allclose(mesh.extents, [10, 10, 10], atol=0.04)
    np.testing.assert_allclose(mesh.bounds.mean(axis=0), 0, atol=1e-5)


@pytest.mark.parametrize("kwargs", [
    {"resolution": True}, {"resolution": 7}, {"resolution": 257},
    {"resolution": 32.5}, {"resolution": float("nan")},
    {"resolution": float("inf")}, {"resolution": "64"},
    {"bounds": ((0, 0, 0), (0, 1, 1))},
    {"bounds": ((1, 0, 0), (0, 1, 1))},
    {"bounds": ((0, 0, 0), (float("inf"), 1, 1))},
    {"bounds": ((0, 0), (1, 1))}, {"bounds": "bad"},
])
def test_malformed_sampling_inputs(kwargs):
    with pytest.raises(ValueError):
        api().make_mesh(sphere, **kwargs)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_field(value):
    def field(x, y, z):
        result = sphere(x, y, z)
        result[0, 0] = value
        return result
    with pytest.raises(ValueError, match="finite"):
        api().make_mesh(field)


@pytest.mark.parametrize("result", [0.0, np.zeros((64,)), np.zeros((64, 64, 1)),
                                       np.zeros((64, 64), dtype=complex)])
def test_wrong_field_shape_or_type(result):
    with pytest.raises(ValueError, match="shape|real"):
        api().make_mesh(lambda x, y, z: result)


@pytest.mark.parametrize("value", [-1., 0., 1.])
def test_missing_strict_crossing(value):
    with pytest.raises(ValueError, match="cross"):
        api().make_mesh(lambda x, y, z: np.full((64, 64), value))


@pytest.mark.parametrize("axis,side", [(a,s) for a in range(3) for s in range(2)])
def test_clipped_domain_boundaries(axis, side):
    bounds = np.array([[-12., -12., -12.], [12., 12., 12.]])
    bounds[side, axis] = -4. if side == 0 else 4.
    with pytest.raises(ValueError, match="boundar"):
        api().make_mesh(sphere, bounds=bounds)


def test_validation_report_is_json_safe_and_not_print_certification():
    mesh = api().make_mesh(sphere)
    report = api().validate_mesh(mesh)
    json.dumps(report, allow_nan=False)
    np.testing.assert_allclose(report["dimensions_mm"], mesh.extents)
    assert report["triangle_count"] == len(mesh.faces)
    assert report["component_count"] == 1
    for flag in ("watertight", "winding_consistent", "positive_volume"):
        assert report[flag] is True
    assert report["physical_printability_verified"] is False
    assert report["wall_thickness_verified"] is False


@pytest.mark.parametrize("kind", ["empty", "open", "flipped", "inward", "degenerate",
                                  "nan", "inf", "flat", "bad-index", "duplicate"])
def test_invalid_mesh_rejected_without_repair(kind):
    mesh = trimesh.creation.box()
    if kind == "empty":
        mesh = trimesh.Trimesh(process=False)
    elif kind == "open":
        mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    elif kind == "flipped":
        mesh.faces[0] = mesh.faces[0][::-1]
    elif kind == "inward":
        mesh.invert()
    elif kind == "degenerate":
        mesh.faces[0] = [0, 0, 1]
    elif kind in ("nan", "inf"):
        mesh.vertices[0, 0] = float(kind)
    elif kind == "flat":
        mesh.vertices[:, 2] = 0
    elif kind == "bad-index":
        mesh.faces[0, 0] = len(mesh.vertices)
    elif kind == "duplicate":
        mesh.faces = np.vstack([mesh.faces, mesh.faces[0]])
    before = mesh.faces.copy()
    with pytest.raises(ValueError):
        api().validate_mesh(mesh)
    np.testing.assert_array_equal(mesh.faces, before)


def test_multiple_mesh_components_require_opt_in():
    first = trimesh.creation.box()
    second = first.copy()
    second.apply_translation([3, 0, 0])
    mesh = trimesh.util.concatenate([first, second])
    with pytest.raises(ValueError, match="component"):
        api().validate_mesh(mesh)
    assert api().validate_mesh(mesh, allow_multiple_components=True)["component_count"] == 2
    second.invert()
    with pytest.raises(ValueError, match="volume"):
        api().validate_mesh(trimesh.util.concatenate([first, second]),
                            allow_multiple_components=True)


def two_spheres(x, y, z):
    return np.minimum(np.sqrt((x-4)**2 + y*y + z*z)-2,
                      np.sqrt((x+4)**2 + y*y + z*z)-2)


def test_meshing_validates_components():
    with pytest.raises(ValueError, match="component"):
        api().make_mesh(two_spheres)
    mesh = api().make_mesh(two_spheres, allow_multiple_components=True)
    assert api().validate_mesh(mesh, allow_multiple_components=True)["component_count"] == 2


@pytest.mark.parametrize("feature", [0., -1., np.nan, np.inf, True, "small"])
def test_invalid_feature_declaration(feature):
    with pytest.raises(ValueError, match="feature"):
        api().make_mesh(sphere, min_feature_mm=feature)


def test_feature_sampling_uses_largest_axis_pitch_not_wall_measurement():
    with pytest.raises(ValueError, match="under-resolved"):
        api().make_mesh(sphere, resolution=32, min_feature_mm=1)
    mesh = api().make_mesh(sphere, resolution=32, min_feature_mm=3)
    report = api().validate_mesh(mesh, pitch_mm=[24/31]*3,
                                 declared_min_feature_mm=3)
    assert report["sampling_requirement_met"] is True
    assert report["declared_min_feature_mm"] == 3
    assert report["max_pitch_mm"] == pytest.approx(24/31)
    assert report["wall_thickness_verified"] is False
    with pytest.raises(ValueError, match="under-resolved"):
        api().validate_mesh(mesh, pitch_mm=[.1, 1.01, .1], declared_min_feature_mm=3)
    unknown = api().validate_mesh(mesh, declared_min_feature_mm=3)
    assert unknown["sampling_requirement_met"] is None


@pytest.mark.parametrize("pitch", [0., -1., np.nan, np.inf, True, [1, 2], [1, 0, 1]])
def test_invalid_validation_pitch(pitch):
    with pytest.raises(ValueError, match="pitch"):
        api().validate_mesh(trimesh.creation.box(), pitch_mm=pitch)


def test_binary_stl_export_is_reopened_and_verified(tmp_path):
    mesh = api().make_mesh(sphere)
    path = tmp_path / "sphere.stl"
    info = api().export_stl(mesh, path)
    json.dumps(info, allow_nan=False)
    assert info["path"] == str(path)
    assert info["format"] == "binary_stl"
    assert info["reload_verified"] is True
    raw = path.read_bytes()
    assert len(raw) == 84 + 50*len(mesh.faces)
    assert int.from_bytes(raw[80:84], "little") == len(mesh.faces)
    loaded = trimesh.load_mesh(path, file_type="stl", process=True,
                               merge_norm=True, merge_tex=True)
    report = api().validate_mesh(loaded)
    assert report["triangle_count"] == len(mesh.faces)
    np.testing.assert_allclose(loaded.extents, mesh.extents, atol=1e-5)


def test_scaled_scalar_field_preserves_zero_set():
    first = api().make_mesh(sphere, resolution=32)
    scaled = api().make_mesh(lambda x, y, z: 17*sphere(x, y, z), resolution=32)
    np.testing.assert_array_equal(first.faces, scaled.faces)
    np.testing.assert_allclose(first.vertices, scaled.vertices, atol=2e-6)


def test_anisotropic_spacing_and_nonzero_origin():
    def ellipsoid(x, y, z):
        return ((x-8)/3)**2 + ((y+5)/4)**2 + ((z-12)/5)**2 - 1
    mesh = api().make_mesh(ellipsoid, bounds=((1, -13, 2), (15, 3, 22)))
    np.testing.assert_allclose(mesh.extents, [6, 8, 10], atol=.04)
    np.testing.assert_allclose(mesh.bounds.mean(axis=0), [8, -5, 12], atol=2e-6)
    assert mesh.volume > 0


def test_sampling_callback_receives_only_slices_at_max_resolution():
    calls = []
    def sliced(x, y, z):
        assert np.ndim(x) == 0
        assert y.shape == (256, 1) and z.shape == (1, 256)
        calls.append(float(x))
        return sphere(x, y, z)
    mesh = api().make_mesh(sliced, resolution=np.int64(256))
    assert len(calls) == 256
    np.testing.assert_allclose(mesh.extents, 10, atol=.004)


def test_export_rejects_invalid_mesh_before_overwriting(tmp_path):
    path = tmp_path / "keep.stl"
    path.write_bytes(b"keep existing file")
    mesh = trimesh.creation.box()
    mesh.faces[0] = mesh.faces[0][::-1]
    with pytest.raises(ValueError):
        api().export_stl(mesh, path)
    assert path.read_bytes() == b"keep existing file"


@pytest.mark.parametrize("where", ["make", "validate"])
def test_component_opt_in_must_be_boolean(where):
    with pytest.raises(ValueError, match="boolean"):
        if where == "make":
            api().make_mesh(two_spheres, allow_multiple_components="false")
        else:
            api().validate_mesh(trimesh.creation.box(), allow_multiple_components="false")


def test_noncallable_field_rejected():
    with pytest.raises(ValueError, match="callable"):
        api().make_mesh(None)


def test_binary_stl_multiple_components_require_explicit_opt_in(tmp_path):
    mesh = api().make_mesh(two_spheres, allow_multiple_components=True)
    path = tmp_path / "two.stl"
    with pytest.raises(ValueError, match="component"):
        api().export_stl(mesh, path)
    assert not path.exists()
    info = api().export_stl(mesh, path, allow_multiple_components=True)
    assert info["component_count"] == 2 and info["reload_verified"] is True
