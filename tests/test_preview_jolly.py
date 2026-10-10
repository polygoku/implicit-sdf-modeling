"""Numerical contracts for shaded visual-review previews (not GLSL execution)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.preview_jolly import normals


@pytest.mark.parametrize("step", [0, -1, True, np.nan, np.inf])
def test_normals_reject_invalid_step(step):
    with pytest.raises(ValueError):
        normals(lambda p: np.linalg.norm(p, axis=-1), np.array([[1., 0., 0.]]), step)


def test_surface_points_reconstruct_center_ray_without_miss_infinities():
    from scripts import preview_jolly
    depths = np.array([[np.inf, 2., np.inf]])
    hits = np.array([[False, True, False]])
    points = preview_jolly.surface_points(depths, hits, (0, 0, 3), (0, 0, 0), (0, 1, 0))
    np.testing.assert_allclose(points, [[0, 0, 1]], atol=1e-12)
    assert points.shape == (1, 3)


def test_preview_cli_produces_reference_comparison_and_four_shaded_views(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "preview"
    result = subprocess.run([sys.executable, str(root / "scripts/preview_jolly.py"),
                             "--output", str(output), "--size", "64"],
                            cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    for name in ("front", "side", "top", "three_quarter"):
        image = np.asarray(Image.open(output / f"shaded_{name}.png"))
        assert image.shape == (64, 64, 3)
        assert np.any(image) and np.any(np.all(image == 0, axis=-1))
    assert (output / "reference_comparison.png").is_file()
    report = json.loads((output / "preview.json").read_text())
    assert report["source_reference_sha256"] == "7728cfa5602bf3d557f0076709a9311b374235234a6c108492ae506b37c8a583"
    assert report["glsl_executed"] is False
    assert report["stl_materials_exported"] is False


@pytest.mark.parametrize("bad", ["nonfinite", "shape"])
def test_normals_reject_invalid_field_values(bad):
    def field(p):
        return np.full(p.shape[:-1] if bad == "nonfinite" else (7,), np.nan if bad == "nonfinite" else 0.)
    with pytest.raises(ValueError):
        normals(field, np.array([[1., 0., 0.]]))


def test_normals_of_sphere_are_outward():
    p = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    def field(q):
        return np.linalg.norm(q, axis=-1) - 1
    np.testing.assert_allclose(normals(field, p), p, atol=1e-8)
