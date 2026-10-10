"""End-to-end pipeline acceptance tests: artifacts and safe loading."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.sdf_pipeline import load_model

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "sdf_pipeline.py"


def run_cli(*args):
    return subprocess.run([sys.executable, str(CLI), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=90)


def test_custom_model_needs_a_float_map_signature(tmp_path):
    model = tmp_path / "missing_map.py"
    model.write_text("def sdf(p): return p[..., 0]\n"
                     "BOUNDS = ((-2,-2,-2),(2,2,2))\n"
                     "LIPSCHITZ = 1.0\nGLSL = '/* TODO */'\n")
    with pytest.raises(ValueError, match="float map"):
        load_model(str(model), trusted=True)


def test_untrusted_custom_model_is_not_executed(tmp_path):
    sentinel = tmp_path / "executed.txt"
    model = tmp_path / "untrusted.py"
    model.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n")
    out = tmp_path / "out"
    result = run_cli("--model", model, "--output", out)
    assert result.returncode != 0
    assert "--trust-model" in result.stderr
    assert not sentinel.exists()
    assert not out.exists()


def test_existing_output_is_preserved(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("original")
    result = run_cli("--model", "sphere", "--output", out)
    assert result.returncode != 0
    assert marker.read_text() == "original"
    assert not (out / "model.stl").exists()


def test_invalid_resolution_fails_without_artifacts(tmp_path):
    out = tmp_path / "out"
    result = run_cli("--output", out, "--resolution", 1000)
    assert result.returncode != 0
    assert not out.exists()


def test_cli_explicitly_allows_two_closed_components(tmp_path):
    model = tmp_path / "two.py"
    model.write_text('''import numpy as np
BOUNDS = ((-12,-12,-12),(12,12,12))
LIPSCHITZ = 1.0
GLSL = "float map(vec3 p) { return min(length(p-vec3(4,0,0))-2.0, length(p+vec3(4,0,0))-2.0); }"
def sdf(p):
    return np.minimum(np.linalg.norm(p-[4,0,0],axis=-1)-2,
                      np.linalg.norm(p+[4,0,0],axis=-1)-2)
''')
    out = tmp_path / "two"
    result = run_cli("--model", model, "--trust-model", "--output", out,
                     "--allow-multiple-components", "--resolution", 32,
                     "--width", 64, "--height", 48)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "validation.json").read_text())
    assert report["component_count"] == 2
    assert (out / "model.stl").is_file()


def test_sphere_cli_exports_verified_artifacts(tmp_path):
    out = tmp_path / "sphere"
    result = run_cli("--model", "sphere", "--output", out,
                     "--resolution", 32, "--width", 64, "--height", 48)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "validation.json").read_text())
    assert report["units"] == "mm"
    assert report["watertight"] is True
    assert report["winding_consistent"] is True
    assert report["positive_volume"] is True
    assert report["physical_printability_verified"] is False
    assert report["wall_thickness_verified"] is False
    assert np.allclose(report["dimensions_mm"], [20, 20, 20], atol=0.25)
    assert report["triangle_count"] > 0
    assert (out / "model.stl").stat().st_size > 84
    assert "float map(vec3 p)" in (out / "model.glsl").read_text()
    for view in ("front", "side", "top"):
        image = np.asarray(Image.open(out / f"preview_{view}.png"))
        assert image.shape == (48, 64, 3)
        assert np.any(image > 0)
        assert np.all(image[0, 0] == 0)
