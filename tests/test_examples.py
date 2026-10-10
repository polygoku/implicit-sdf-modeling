"""Tests for deliberately hand-paired NumPy and GLSL example models."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def model(name):
    path = Path(__file__).parents[1] / "examples" / f"{name}.py"
    assert path.exists(), f"Example {name} must exist"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sphere_contract_and_shared_glsl_radius():
    m = model("sphere")
    points = np.array([[[0, 0, 0], [10, 0, 0]], [[0, -12, 0], [0, 0, 6]]])
    np.testing.assert_allclose(m.sdf(points), [[-10, 0], [2, -4]])
    assert np.shape(m.sdf(np.array([0, 0, 0]))) == ()
    assert m.sdf(np.empty((0, 3))).shape == (0,)
    assert m.BOUNDS == ((-12.0, -12.0, -12.0), (12.0, 12.0, 12.0))
    assert m.RADIUS == 10.0 and m.LIPSCHITZ == 1.0
    assert "float map(vec3 p)" in m.GLSL
    assert f"length(p) - {m.RADIUS:.8f}" in m.GLSL


def test_blended_capsules_has_shared_constants_and_nontrivial_distances():
    m = model("blended_capsules")
    assert m.LIPSCHITZ == 1.0
    assert len(m.CAPSULES) == 2
    assert m.BLEND_RADIUS > 0
    a, b, radius = m.CAPSULES[0]
    points = np.array([a, b, (0, 0, 0), (0, 0, 5)])
    np.testing.assert_allclose(m.capsule_sdf(points, a, b, radius),
                               [-radius, -radius, -radius, 5 - radius])
    np.testing.assert_allclose(m.sdf(points)[2:],
                               [-radius - m.BLEND_RADIUS / 4,
                                5 - radius - m.BLEND_RADIUS / 4])
    assert m.sdf(points.reshape(2, 2, 3)).shape == (2, 2)
    assert np.shape(m.sdf(np.zeros(3))) == ()
    assert m.sdf(np.empty((0, 3))).shape == (0,)
    assert "float map(vec3 p)" in m.GLSL
    assert "float sdCapsule(" in m.GLSL and "float smoothMin(" in m.GLSL
    assert "dot(pa, ba) / dot(ba, ba)" in m.GLSL
    assert "mix(b, a, h) - k * h * (1.0 - h)" in m.GLSL
    for a, b, radius in m.CAPSULES:
        for endpoint in (a, b):
            literal = "vec3(" + ", ".join(f"{x:.8f}" for x in endpoint) + ")"
            assert literal in m.GLSL
        assert f"{radius:.8f}" in m.GLSL
    assert f"{m.BLEND_RADIUS:.8f}" in m.GLSL


@pytest.mark.parametrize("a,b,radius", [
    ((0, 0, 0), (0, 0, 0), 1),
    ((0, 0, 0), (1e-300, 0, 0), 1),
    ((0, 0, 0), (1e300, 0, 0), 1),
    ((0, 0), (1, 0, 0), 1),
    ((0, 0, 0), (np.nan, 0, 0), 1),
    ((0, 0, 0), (1, 0, 0), 0),
    ((0, 0, 0), (1, 0, 0), -1),
    ((0, 0, 0), (1, 0, 0), np.inf),
])
def test_capsule_rejects_degenerate_or_invalid_geometry(a, b, radius):
    with pytest.raises(ValueError):
        model("blended_capsules").capsule_sdf(np.zeros((1, 3)), a, b, radius)


@pytest.mark.parametrize("k", [0, -1, np.nan, np.inf])
def test_smooth_min_rejects_invalid_blend_radius(k):
    with pytest.raises(ValueError, match="blend"):
        model("blended_capsules").smooth_min(np.array([1.0]), np.array([2.0]), k)


def test_glsl_helpers_guard_their_divisors():
    source = model("blended_capsules").GLSL
    assert "if (dot(ba, ba) <= 0.0 || r <= 0.0) return 1e20;" in source
    assert "if (k <= 0.0) return 1e20;" in source


@pytest.mark.parametrize("name", ["sphere", "blended_capsules"])
@pytest.mark.parametrize("points", [
    np.zeros(2), np.zeros((2, 4)), np.zeros((3, 1)), np.array(0),
    np.array([[0, np.nan, 0]]), np.array([[0, 0, np.inf]]),
    np.array([[0, 1j, 0]]),
])
def test_model_rejects_invalid_point_arrays(name, points):
    with pytest.raises(ValueError, match="points"):
        model(name).sdf(points)


def test_example_and_script_packages_are_explicit_importable_packages():
    root = Path(__file__).parents[1]
    assert (root / "scripts" / "__init__.py").is_file()
    assert (root / "examples" / "__init__.py").is_file()
    from examples import blended_capsules, sphere
    from scripts.render_sdf import render
    for m in (sphere, blended_capsules):
        image, hits, depths = render(m.sdf, width=41, height=31,
                                    camera=(0, 0, 40), max_distance=80,
                                    lipschitz=m.LIPSCHITZ)
        assert hits[15, 20] and np.isfinite(depths[15, 20])
        assert image[hits].any() and not image[~hits].any()
        assert not hits[0, 0]


@pytest.mark.parametrize("name", ["sphere", "blended_capsules"])
def test_model_is_one_bounded_component_on_sample_grid(name):
    m = model(name)
    low, high = np.asarray(m.BOUNDS)
    axes = [np.linspace(a, b, 29) for a, b in zip(low, high)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    distances = m.sdf(points)
    assert np.isfinite(distances).all()
    for axis in range(3):
        assert np.all(np.take(distances, [0, -1], axis=axis) > 0)
    occupied = distances < 0
    assert occupied.any()
    seed = tuple(np.argwhere(occupied)[0])
    visited = {seed}
    pending = [seed]
    while pending:
        current = pending.pop()
        for axis in range(3):
            for step in (-1, 1):
                neighbor = list(current)
                neighbor[axis] += step
                neighbor = tuple(neighbor)
                if (all(0 <= i < 29 for i in neighbor) and occupied[neighbor]
                        and neighbor not in visited):
                    visited.add(neighbor)
                    pending.append(neighbor)
    assert len(visited) == np.count_nonzero(occupied)


@pytest.mark.parametrize("name", ["sphere", "blended_capsules"])
def test_model_gradient_respects_declared_lipschitz_bound(name):
    m = model(name)
    points = np.random.default_rng(472).uniform(*np.asarray(m.BOUNDS), (1000, 3))
    delta = 1e-5
    gradient = np.stack([(m.sdf(points + np.eye(3)[i] * delta) -
                          m.sdf(points - np.eye(3)[i] * delta)) / (2 * delta)
                         for i in range(3)], axis=-1)
    assert np.all(np.linalg.norm(gradient, axis=-1) <= m.LIPSCHITZ + 1e-6)


def test_capsule_projection_and_smooth_min_match_independent_scalar_reference():
    # Analytic CPU reference, not a claim that the GLSL string was executed.
    m = model("blended_capsules")
    points = np.random.default_rng(13).uniform(-20, 20, (200, 3))
    expected = []
    for p in points:
        distances = []
        for a, b, radius in m.CAPSULES:
            segment = [bj - aj for aj, bj in zip(a, b)]
            relative = [pj - aj for pj, aj in zip(p, a)]
            t = sum(x * y for x, y in zip(relative, segment)) / sum(x*x for x in segment)
            t = max(0.0, min(1.0, t))
            d = sum((x - t*y)**2 for x, y in zip(relative, segment)) ** 0.5 - radius
            distances.append(d)
        a, b = distances
        # Equivalent polynomial expressed differently from the implementation.
        correction = max(m.BLEND_RADIUS - abs(a - b), 0.0) ** 2 / (4 * m.BLEND_RADIUS)
        expected.append(min(a, b) - correction)
    np.testing.assert_allclose(m.sdf(points), expected, atol=1e-12)


def test_smooth_min_blends_only_near_the_join():
    m = model("blended_capsules")
    a, b = np.array([-10, 0, 1]), np.array([10, 0, 2])
    result = m.smooth_min(a, b, m.BLEND_RADIUS)
    np.testing.assert_allclose(result, m.smooth_min(b, a, m.BLEND_RADIUS))
    assert result[0] == -10
    assert result[1] == -m.BLEND_RADIUS / 4
    assert np.all(result <= np.minimum(a, b))
    assert np.all(result >= np.minimum(a, b) - m.BLEND_RADIUS / 4)
