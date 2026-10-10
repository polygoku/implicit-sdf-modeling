"""Behavioral regression tests for the NumPy perspective marcher."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def renderer():
    path = Path(__file__).parents[1] / "scripts" / "render_sdf.py"
    assert path.exists(), "NumPy renderer must exist"
    spec = importlib.util.spec_from_file_location("renderer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render


def sphere(points):
    return np.linalg.norm(points, axis=-1) - 1.0


def test_perspective_sphere_has_visible_center_and_black_miss_corners():
    image, hit, depths = renderer()(sphere, width=41, height=31)
    assert image.shape == (31, 41, 3) and image.dtype == np.uint8
    assert hit.shape == (31, 41) and hit.dtype == np.bool_
    assert depths.shape == (31, 41) and depths.dtype.kind == "f"
    assert hit[15, 20]
    assert depths[15, 20] == pytest.approx(2.0, abs=0.002)
    assert image[15, 20].any()
    assert not hit[0, 0] and not hit[-1, -1]
    assert np.all(image[~hit] == 0)
    assert np.all(np.isinf(depths[~hit]))


@pytest.mark.parametrize("kwargs", [
    {"width": 0}, {"height": -1}, {"width": 1.5}, {"width": True},
    {"height": np.inf}, {"max_steps": 0}, {"max_steps": 1.5},
    {"max_steps": True}, {"epsilon": 0}, {"epsilon": -0.1},
    {"epsilon": np.nan}, {"max_distance": 0}, {"max_distance": np.inf},
    {"lipschitz": 0}, {"lipschitz": -1}, {"lipschitz": np.nan},
    {"lipschitz": np.inf}, {"fov_degrees": 0}, {"fov_degrees": 180},
    {"fov_degrees": np.nan}, {"camera": (0, 0)},
    {"camera": (0, np.inf, 3)}, {"target": (0, 0, 3)},
    {"target": (0, np.nan, 0)}, {"up": (0, 0, 0)},
    {"up": (0, 0, 1)}, {"up": (0, np.inf, 0)},
])
def test_invalid_render_arguments_are_rejected_before_field_call(kwargs):
    def field(points):
        pytest.fail("Invalid arguments must be rejected before SDF evaluation")
    with pytest.raises(ValueError):
        renderer()(field, **kwargs)


@pytest.mark.parametrize("field", [
    None,
    lambda p: 1.0,
    lambda p: np.zeros((len(p), 1)),
    lambda p: np.full(len(p), np.nan),
    lambda p: np.full(len(p), np.inf),
    lambda p: np.full(len(p), -np.inf),
    lambda p: np.full(len(p), 1j),
    lambda p: np.full(len(p), "bad"),
])
def test_bad_field_result_is_rejected(field):
    with pytest.raises(ValueError, match="sdf"):
        renderer()(field, width=3, height=3)


def test_extreme_finite_distances_do_not_overflow_or_shade_misses():
    calls = []
    def field(points):
        calls.append(points.copy())
        return np.full(len(points), 1e308)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        image, hit, depths = renderer()(field, width=3, height=3,
                                       lipschitz=1e-308, max_distance=4)
    assert not hit.any() and not image.any() and np.isinf(depths).all()
    assert len(calls) <= 2
    assert all(np.isfinite(points).all() for points in calls)
    assert all(np.all(np.linalg.norm(points - (0, 0, 3), axis=-1) <= 4.000001)
               for points in calls)


def test_inside_camera_marches_to_exit_not_immediate_negative_hit():
    image, hit, depths = renderer()(sphere, width=7, height=5,
                                   camera=(0, 0, 0), target=(0, 0, -1))
    assert hit.all() and image.any()
    np.testing.assert_allclose(depths, 1.0, atol=1e-12)


def test_camera_uses_common_origin_with_correct_off_axis_perspective():
    # An independent ray/sphere quadratic validates more than the center pixel.
    width, height, fov = 43, 33, 50
    _, hit, depths = renderer()(sphere, width=width, height=height,
                               fov_degrees=fov, epsilon=1e-6, max_steps=1000)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    scale = np.tan(np.deg2rad(fov) / 2)
    x = (2 * (xx + 0.5) / width - 1) * scale * width / height
    y = (1 - 2 * (yy + 0.5) / height) * scale
    dz = -1 / np.sqrt(x * x + y * y + 1)
    discriminant = (3 * dz) ** 2 - 8
    expected = discriminant > 0
    np.testing.assert_array_equal(hit, expected)
    expected_depth = -3 * dz[expected] - np.sqrt(discriminant[expected])
    np.testing.assert_allclose(depths[hit], expected_depth, atol=2e-5)


def test_rotated_and_translated_camera_aims_at_target():
    center = np.array((7, -3, 2))
    def moved_sphere(points):
        return sphere(points - center)
    image, hit, depths = renderer()(moved_sphere, width=11, height=9,
                                   camera=center + (3, 0, 0), target=center)
    assert hit[4, 5] and image[4, 5].any()
    assert depths[4, 5] == pytest.approx(2.0)
    assert not hit[0, 0]


@pytest.mark.parametrize("far,expected", [(1.9, False), (2.0, True), (2.1, True)])
def test_far_plane_is_inclusive_and_never_shades_beyond_limit(far, expected):
    image, hit, depths = renderer()(sphere, width=1, height=1, max_distance=far)
    assert bool(hit[0, 0]) is expected
    assert bool(image.any()) is expected
    if expected:
        assert depths[0, 0] <= far
    else:
        assert np.isinf(depths[0, 0])


def test_step_budget_and_active_ray_compaction():
    counts = []
    def recorded(points):
        counts.append(len(points))
        return sphere(points)
    image, hit, depths = renderer()(recorded, width=41, height=31, max_steps=8)
    assert len(counts) == 8
    assert counts[0] == 41 * 31 and counts[2] < counts[1]
    assert hit[15, 20] and np.all(image[~hit] == 0)
    assert np.isinf(depths[~hit]).all()
    image, hit, depths = renderer()(sphere, width=1, height=1, max_steps=1)
    assert not hit.any() and not image.any() and np.isinf(depths).all()


def test_lipschitz_divisor_handles_scaled_distance_bound():
    _, hit, depths = renderer()(lambda p: 4 * sphere(p), width=1, height=1,
                               lipschitz=4)
    assert hit[0, 0] and depths[0, 0] == pytest.approx(2.0)


def test_constant_negative_field_does_not_count_as_a_hit():
    calls = []
    def negative(points):
        calls.append(len(points))
        return -np.ones(len(points))
    image, hit, depths = renderer()(negative, width=3, height=3, max_steps=4)
    assert len(calls) == 4
    assert not hit.any() and not image.any() and np.isinf(depths).all()


def test_later_nonfinite_evaluation_is_rejected():
    calls = []
    def field(points):
        calls.append(len(points))
        return sphere(points) if len(calls) == 1 else np.full(len(points), np.nan)
    with pytest.raises(ValueError, match="sdf"):
        renderer()(field, width=3, height=3)
