"""Reference geometry and direct distance-bound regressions (not GLSL execution)."""
import numpy as np
import pytest

from examples import jolly_waving as jolly


def test_glsl_arc_axis_returns_exact_distance_before_undefined_atan():
    helper = jolly.GLSL.split("float sdTorusArc", 1)[1].split("float map", 1)[0]
    guard = "if (q.x == 0.0 && q.y == 0.0) return length(vec2(R, q.z)) - r;"
    assert guard in helper
    assert helper.index(guard) < helper.index("atan(")
    points = np.array([[1, 2, 3], [1, 2, 15]])
    np.testing.assert_allclose(jolly._torus_arc(points, (1, 2, 3), 5, .65, -2.47, -.67),
                               np.hypot(5, [0, 12]) - .65)


def test_reference_pose_and_connected_wide_hood():
    # Wave on viewer-right, hanging hand on left, flat separated feet.
    assert jolly.sdf(np.array([61., 40., 3.])) < 0
    assert jolly.sdf(np.array([-54., -42., 3.])) < 0
    assert jolly.sdf(np.array([-61., 40., 3.])) > 0
    assert jolly.sdf(np.array([30., 22., 0.])) < 0
    assert jolly.sdf(np.array([25., 54., 32.])) < 0  # broad face corner
    np.testing.assert_allclose(jolly.sdf(np.array([[17, -74, 2], [20, -74, 6]])), 0,
                               atol=1e-12)
    assert jolly.sdf(np.array([0., -66., 2.])) > 0
    np.testing.assert_allclose(jolly.sdf(np.array([0., 78., 0.])), 0, atol=1e-12)


def test_shared_scene_fields_and_material_contract():
    assert hasattr(jolly, "SCENE"), "NumPy and GLSL need one shared scene"
    points = np.array([[[0., 43., 35.], [-14.5, 47., 36.8]],
                       [[61., 40., 3.], [0., -13., -30.]]])
    fields = jolly.primitive_fields(points)
    assert fields.shape == (2, 2, len(jolly.SCENE))
    result = fields[..., 0]
    for i, (_, _, _, _, blend) in enumerate(jolly.SCENE[1:], 1):
        result = (jolly._smin(result, fields[..., i], blend) if blend
                  else np.minimum(result, fields[..., i]))
    np.testing.assert_allclose(jolly.sdf(points), result)
    colors = jolly.material(points)
    assert colors.shape == points.shape and colors.dtype.kind == "f"
    np.testing.assert_allclose(colors, np.asarray(jolly.PALETTE)[[[1, 2], [0, 0]]])
    assert np.all((colors >= 0) & (colors <= 1))
    assert jolly.primitive_fields(np.empty((0, 3))).shape == (0, len(jolly.SCENE))
    assert np.shape(jolly.material(np.zeros(3))) == (3,)
    for _, _, center, parameters, blend in jolly.SCENE:
        for item in (center, *parameters, blend):
            assert jolly._glsl_literal(item) in jolly.GLSL


@pytest.mark.parametrize("half,rounding", [((1, 2, 3), 0), ((1, 2, 3), -1),
                                          ((1, 2, 3), 2), ((1, 2, 3), np.nan),
                                          ((1, 2, np.inf), 1), ((1, 2), 1)])
def test_rounded_box_rejects_invalid_geometry(half, rounding):
    assert hasattr(jolly, "_rounded_box")
    with pytest.raises(ValueError):
        jolly._rounded_box(np.zeros(3), (0, 0, 0), half, rounding)


def test_rounded_box_outer_half_extent_and_flat_sole():
    assert hasattr(jolly, "_rounded_box")
    p = np.array([[0, 0, 0], [12.5, 0, 0], [0, -13, 0], [2, -13, 4]])
    np.testing.assert_allclose(jolly._rounded_box(p, (0, 0, 0), (12.5, 13, 18), 5),
                               [-12.5, 0, 0, 0])


def test_arc_cuts_endpoints_and_axis_match_independent_nearest_set():
    R, r, a0, a1 = 5., .65, -2.47, -.67
    mid = (a0 + a1) / 2
    eps = 1e-8
    angles = np.array([-np.pi-eps, -np.pi+eps, mid+np.pi-eps,
                       mid+np.pi+eps, a0, a1, mid, a0-eps, a1+eps])
    q = np.column_stack((9*np.cos(angles), 9*np.sin(angles), np.full(len(angles), 2.)))
    q = np.vstack((q, [0, 0, 0], [0, 0, 12]))
    ends = R * np.array([[np.cos(a0), np.sin(a0), 0],
                         [np.cos(a1), np.sin(a1), 0]])
    expected = np.linalg.norm(q[:, None] - ends, axis=-1).min(axis=1)
    rho = np.linalg.norm(q[:, :2], axis=-1)
    in_sector = (q[:, 0]*np.cos(mid) + q[:, 1]*np.sin(mid)
                 >= rho*np.cos((a1-a0)/2))
    expected[in_sector] = np.hypot(rho[in_sector]-R, q[in_sector, 2])
    np.testing.assert_allclose(jolly._torus_arc(q, (0, 0, 0), R, r, a0, a1),
                               expected-r, atol=1e-12)
    endpoints = ends + [0, 0, r]
    np.testing.assert_allclose(jolly._torus_arc(endpoints, (0, 0, 0), R, r, a0, a1),
                               0, atol=1e-12)


def test_naive_arc_clipping_is_discontinuous_and_not_one_lipschitz():
    p = np.array([[-9., -1e-8, 2.], [-9., 1e-8, 2.]])
    a = np.clip(np.arctan2(p[:, 1], p[:, 0]), -2.47, -.67)
    naive = np.linalg.norm(p - np.column_stack((5*np.cos(a), 5*np.sin(a),
                                               np.zeros(2))), axis=-1) - .65
    correct = jolly._torus_arc(p, (0, 0, 0), 5, .65, -2.47, -.67)
    separation = np.linalg.norm(p[0]-p[1])
    assert abs(naive[0]-naive[1]) > 1
    assert abs(correct[0]-correct[1]) <= separation + 1e-12
    assert naive[1] > correct[1] + 1  # wrong endpoint across atan2 cut


def test_ellipsoid_center_surfaces_and_near_center_are_finite_bounds():
    center, radii = np.array([1., 2., 3.]), np.array([40., 46., 30.])
    directions = np.vstack((np.eye(3), -np.eye(3), [1, 2, 3]))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    surface = center + directions*radii
    np.testing.assert_allclose(jolly._ellipsoid(surface, center, radii), 0, atol=1e-12)
    np.testing.assert_allclose(jolly._ellipsoid(center, center, radii), -30.)
    nearby = center + np.eye(3)*1e-8
    d = jolly._ellipsoid(nearby, center, radii)
    assert np.isfinite(d).all()
    assert np.all(np.abs(d+30) <= 1e-8+1e-12)


@pytest.mark.parametrize("fn", [jolly.primitive_fields, jolly.material])
@pytest.mark.parametrize("points", [np.zeros(2), np.array(0), np.zeros((2, 4)),
                                    [[0, np.nan, 0]], [[0, np.inf, 0]], [[0, 1j, 0]]])
def test_diagnostic_apis_reject_invalid_points(fn, points):
    with pytest.raises(ValueError, match="points"):
        fn(points)


def test_rounded_prism_outer_axes_and_depth_independent_xy_rounding():
    assert hasattr(jolly, "_rounded_prism"), "XY corner must not be limited by Z depth"
    center = np.array([1., 2., 3.])
    half, corner, edge = (28., 21., 7.), 17.5, 2.
    axes = np.vstack((np.eye(3), -np.eye(3))) * half + center
    np.testing.assert_allclose(jolly._rounded_prism(axes, center, half, corner, edge),
                               0, atol=1e-12)
    # XY outline at the midplane has radius 17.5 despite only 7 mm half-depth.
    diagonal = np.array([28-corner, 21-corner, 0.])
    diagonal[:2] += corner / np.sqrt(2)
    np.testing.assert_allclose(jolly._rounded_prism(diagonal+center, center,
                                                  half, corner, edge), 0, atol=1e-12)
    assert jolly._rounded_prism(np.array([25., 18., 0.])+center,
                                center, half, corner, edge) > 0
    np.testing.assert_allclose(jolly._rounded_prism(center, center, half, corner, edge), -7)


@pytest.mark.parametrize("half,corner,edge", [
    ((28, 21, 7), 17.5, 0), ((28, 21, 7), 17.5, -1),
    ((28, 21, 7), 17.5, 7), ((28, 21, 7), 2, 2),
    ((28, 21, 7), 0, 1), ((28, 21, 7), 22, 2),
    ((28, 21, 7), np.nan, 2), ((28, 21, 7), 17.5, np.inf),
    ((28, 21, np.inf), 17.5, 2), ((28, 21, np.nan), 17.5, 2),
    ((28, 21, 0), 17.5, 2), ((-28, 21, 7), 17.5, 2),
    ((28, 21), 17.5, 2),
])
def test_rounded_prism_rejects_invalid_geometry(half, corner, edge):
    with pytest.raises(ValueError, match="rounded prism"):
        jolly._rounded_prism(np.zeros(3), (0, 0, 0), half, corner, edge)


@pytest.mark.parametrize("center", [(0, 0), (0, np.nan, 0), (0, 0, np.inf)])
def test_rounded_prism_rejects_invalid_center(center):
    with pytest.raises(ValueError, match="rounded prism"):
        jolly._rounded_prism(np.zeros(3), center, (28, 21, 7), 17.5, 2)


def test_face_projection_uses_shared_soft_prisms_not_thick_visor():
    records = {record[0]: record for record in jolly.SCENE}
    assert records["hood"] == ("hood", "rounded_prism", (0., 42., 0.),
                                ((38., 36., 32.), 26., 12.), 8.)
    assert records["face"] == ("face", "rounded_prism", (0., 43., 27.),
                                ((28., 21., 7.), 17.5, 2.), 1.)
    np.testing.assert_allclose(jolly._primitive(np.array([0., 43., 34.]), records["face"]), 0)
    np.testing.assert_allclose(jolly._primitive(np.array([0., 43., 32.]), records["hood"]), 0)
    np.testing.assert_allclose(jolly.sdf(np.array([0., 43., 34.])), 0, atol=1e-12)
    assert records["left_eye"][2:] == ((-14.5, 47., 33.5), (2.8,), 0.)
    assert records["right_eye"][2:] == ((14.5, 47., 33.5), (2.8,), 0.)
    assert records["smile"][2:] == ((0., 45., 34.3), (5., .65, (-2.47, -.67)), 0.)
    for name in ("hood", "face"):
        _, _, center, params, blend = records[name]
        args = ", ".join(jolly._glsl_literal(v) for v in (center, *params))
        assert f"sdRoundedPrism(p, {args})" in jolly.GLSL
        assert np.all(np.array(center) - np.array(params[0]) > np.array(jolly.BOUNDS)[0])
        assert np.all(np.array(center) + np.array(params[0]) < np.array(jolly.BOUNDS)[1])
    assert "vec2 d = vec2(dxy, dz);" in jolly.GLSL
    assert "length(max(d,vec2(0.0))) + min(max(d.x,d.y),0.0) - edge" in jolly.GLSL


@pytest.mark.parametrize("half,corner,edge", [((28., 21., 7.), 17.5, 2.),
                                              ((38., 36., 32.), 26., 12.),
                                              ((8., 8., 3.), 8., 1.)])
def test_rounded_prism_product_gradient_and_sampled_one_lipschitz(half, corner, edge):
    half = np.array(half)
    # Outside both factors: product gradient is (a*u_xy, b*u_z)/hypot(a,b).
    # Orthogonal unit vectors give norm exactly 1, not |a|+|b|.
    a, b = 3., 4.
    p = np.array([half[0]-edge+a, 0., half[2]-edge+b])
    def fn(x):
        return jolly._rounded_prism(x, (0, 0, 0), half, corner, edge)

    np.testing.assert_allclose(fn(p), np.hypot(a, b)-edge, atol=1e-12)
    eps = 1e-5
    offsets = np.eye(3)*eps
    gradient = (fn(p+offsets)-fn(p-offsets))/(2*eps)
    np.testing.assert_allclose(gradient, [a/5, 0, b/5], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(gradient), 1, atol=1e-9)
    # Exact fillet checkpoint: positive core XY/Z distances share edge radius.
    surface = np.array([half[0]-edge+edge/np.sqrt(2), 0.,
                        half[2]-edge+edge/np.sqrt(2)])
    np.testing.assert_allclose(fn(surface), 0, atol=1e-12)
    rng = np.random.default_rng(294)
    points = rng.uniform(-1.5*half, 1.5*half, (4000, 3))
    nearby = points + rng.normal(size=points.shape)*1e-5
    far = rng.uniform(-1.5*half, 1.5*half, points.shape)
    for other in (nearby, far):
        assert np.all(np.abs(fn(points)-fn(other)) <= np.linalg.norm(points-other, axis=-1)+1e-12)
    gradients = np.stack([(fn(points+offset)-fn(points-offset))/(2*eps)
                          for offset in offsets], axis=-1)
    assert np.max(np.linalg.norm(gradients, axis=-1)) <= 1+1e-8


def test_rounded_prism_bounds_include_exact_smooth_budget(monkeypatch):
    record = ("probe", "rounded_prism", (1., 2., 3.), ((28., 21., 7.), 17.5, 2.), 8.)
    monkeypatch.setattr(jolly, "SCENE", (record,))
    # 8/4 blend allowance plus the existing 2 mm domain margin.
    center, half = np.array(record[2]), np.array(record[3][0])
    np.testing.assert_allclose(jolly._scene_bounds(), [center-half-4, center+half+4])


# Same narrow upstream skimage/NumPy compatibility filter as test_mesh_sdf.
@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:"
    "DeprecationWarning:skimage.measure._marching_cubes_lewiner")
def test_refined_face_stays_one_closed_mesh_component_at_64(tmp_path):
    import trimesh

    from scripts.mesh_sdf import export_stl, make_mesh, validate_mesh

    mesh = make_mesh(jolly.sdf, bounds=jolly.BOUNDS, resolution=64)
    report = export_stl(mesh, tmp_path / "jolly-prism-64.stl")
    reopened = trimesh.load_mesh(tmp_path / "jolly-prism-64.stl", process=True)
    checked = validate_mesh(reopened)
    assert report["reload_verified"]
    assert checked["component_count"] == 1
    assert checked["watertight"] and checked["winding_consistent"] and checked["positive_volume"]
    np.testing.assert_allclose(checked["dimensions_mm"][1], 152., atol=.1)


def test_cli_help_lists_jolly_builtin(capsys):
    from scripts.sdf_pipeline import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "jolly-waving" in capsys.readouterr().out
