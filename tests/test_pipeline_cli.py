"""End-to-end pipeline acceptance tests: artifacts and safe loading."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.sdf_pipeline import build, load_model

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


@pytest.mark.parametrize("target_kind", ["empty-directory", "populated-directory", "file"])
@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:"
    "DeprecationWarning:skimage.measure._marching_cubes_lewiner")
def test_publication_refuses_target_after_stale_check(tmp_path, monkeypatch, target_kind):
    out = tmp_path / "raced"
    original_exists = Path.exists
    checks = 0

    def stale_exists(path):
        nonlocal checks
        if path == out:
            checks += 1
            if checks == 2:
                if target_kind == "file":
                    out.write_text("original")
                else:
                    out.mkdir()
                    if target_kind == "populated-directory":
                        (out / "keep.txt").write_text("original")
                return False  # Simulate a stale final existence check.
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", stale_exists)
    model, bounds, lipschitz = load_model("sphere")
    with pytest.raises(ValueError, match="Output.*(?:exists|appeared)"):
        build(model, bounds, lipschitz, out, resolution=16, width=16, height=16)
    assert checks == 2
    if target_kind == "file":
        assert out.read_text() == "original"
    else:
        assert out.is_dir()
        expected = [] if target_kind == "empty-directory" else ["keep.txt"]
        assert sorted(path.name for path in out.iterdir()) == expected
        if expected:
            assert (out / "keep.txt").read_text() == "original"
    assert not list(tmp_path.glob(".sdf-build-*"))


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:"
    "DeprecationWarning:skimage.measure._marching_cubes_lewiner")
def test_publication_report_is_written_after_complete_artifacts(tmp_path, monkeypatch):
    out = tmp_path / "published"
    original_open = Path.open
    artifact_names = ["preview_front.png", "preview_side.png", "preview_top.png",
                      "model.stl", "model.glsl"]
    report_opened = False

    def checked_open(path, mode="r", *args, **kwargs):
        nonlocal report_opened
        if path == out / "validation.json" and mode == "xb":
            report_opened = True
            temporary, = tmp_path.glob(".sdf-build-*")
            for name in artifact_names:
                assert (out / name).is_file()
                assert (out / name).read_bytes() == (temporary / name).read_bytes()
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    model, bounds, lipschitz = load_model("sphere")
    report = build(model, bounds, lipschitz, out, resolution=16, width=16, height=16)
    assert report_opened
    assert json.loads((out / "validation.json").read_text()) == report
    assert not list(tmp_path.glob(".sdf-build-*"))


@pytest.mark.parametrize("failure", ["file-collision", "late-io-error"])
@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:"
    "DeprecationWarning:skimage.measure._marching_cubes_lewiner")
def test_publication_failure_preserves_user_files_without_report(tmp_path, monkeypatch, failure):
    out = tmp_path / "partial"
    original_open = Path.open
    injected = False

    def failing_open(path, mode="r", *args, **kwargs):
        nonlocal injected
        if path == out / "model.glsl" and mode == "xb":
            injected = True
            (out / "keep.txt").write_text("unrelated user file")
            if failure == "late-io-error":
                raise OSError("injected late I/O error")
            path.write_text("user model, never overwrite")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    model, bounds, lipschitz = load_model("sphere")
    with pytest.raises(OSError):
        build(model, bounds, lipschitz, out, resolution=16, width=16, height=16)
    assert injected
    assert out.is_dir()
    assert (out / "keep.txt").read_text() == "unrelated user file"
    assert (out / "model.stl").is_file()
    assert not (out / "validation.json").exists()
    if failure == "file-collision":
        assert (out / "model.glsl").read_text() == "user model, never overwrite"
    assert not list(tmp_path.glob(".sdf-build-*"))


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


def test_cli_small_positive_field_rescaling_preserves_previews(tmp_path):
    model = tmp_path / "scaled_sphere.py"
    model.write_text('''import numpy as np
BOUNDS = ((-12,-12,-12),(12,12,12))
LIPSCHITZ = 1e-6
GLSL = "float map(vec3 p) { return 1e-6 * (length(p)-10.0); }"
def sdf(p):
    return 1e-6 * (np.linalg.norm(p, axis=-1) - 10.0)
''')
    base = tmp_path / "base"
    scaled = tmp_path / "scaled"
    for name, out, trust in (("sphere", base, ()),
                             (model, scaled, ("--trust-model",))):
        result = run_cli("--model", name, *trust, "--output", out,
                         "--resolution", 32, "--width", 64, "--height", 48)
        assert result.returncode == 0, result.stderr
    base_report = json.loads((base / "validation.json").read_text())
    scaled_report = json.loads((scaled / "validation.json").read_text())
    assert scaled_report["preview_hit_pixels"] == base_report["preview_hit_pixels"]
    for view in ("front", "side", "top"):
        base_image = np.asarray(Image.open(base / f"preview_{view}.png"))
        scaled_image = np.asarray(Image.open(scaled / f"preview_{view}.png"))
        assert np.any(base_image > 0)
        assert np.any(scaled_image > 0)
        assert np.all(scaled_image[0, 0] == 0)
        assert np.array_equal(np.all(scaled_image == 0, axis=-1),
                              np.all(base_image == 0, axis=-1))


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
