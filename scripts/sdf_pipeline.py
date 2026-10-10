"""Build previews and a validated STL from a trusted NumPy/GLSL model pair.

Custom Python models execute code; --trust-model is an acknowledgement, not a sandbox.
World coordinates and exported vertex coordinates are millimeters.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_model(name: str, trusted: bool = False):
    builtins = {"sphere": "sphere.py", "blended-capsules": "blended_capsules.py"}
    if name in builtins:
        path = ROOT / "examples" / builtins[name]
    else:
        if not trusted:
            raise ValueError("Custom Python models execute code; inspect it and use --trust-model")
        path = Path(name).expanduser().resolve()
        if path.suffix != ".py" or not path.is_file():
            raise ValueError("A custom model must be an existing .py file")
    spec = importlib.util.spec_from_file_location("sdf_user_model", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the model")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    if not callable(getattr(model, "sdf", None)):
        raise ValueError("Model must define callable sdf(points)")
    if not isinstance(getattr(model, "GLSL", None), str) or not model.GLSL.strip():
        raise ValueError("Model must provide its corresponding GLSL source")
    if not re.search(r"\bfloat\s+map\s*\(\s*vec3\s+\w+\s*\)", model.GLSL):
        raise ValueError("GLSL must contain a float map(vec3 p) signature; this is not compilation")
    bounds = np.asarray(getattr(model, "BOUNDS", None), dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("BOUNDS must contain two finite 3D corners in millimeters")
    if np.any(bounds[1] <= bounds[0]):
        raise ValueError("BOUNDS upper corner must be greater than the lower corner")
    raw_lipschitz = getattr(model, "LIPSCHITZ", None)
    if raw_lipschitz is None or isinstance(raw_lipschitz, bool):
        raise ValueError("Model must declare a positive finite LIPSCHITZ bound")
    try:
        lipschitz = float(raw_lipschitz)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model must declare a positive finite LIPSCHITZ bound") from exc
    if not math.isfinite(lipschitz) or lipschitz <= 0:
        raise ValueError("Model must declare a positive finite LIPSCHITZ bound")
    return model, bounds, lipschitz


def build(model, bounds, lipschitz, output, *, resolution, width, height,
          min_feature_mm=None, allow_multiple_components=False):
    # Imports here allow --help and model trust checks without loading heavy libraries.
    from scripts.mesh_sdf import export_stl, make_mesh, validate_mesh
    from scripts.render_sdf import render

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory to preserve previous artifacts")
    mesh = make_mesh(model.sdf, bounds=bounds, resolution=resolution,
                     min_feature_mm=min_feature_mm,
                     allow_multiple_components=allow_multiple_components)
    pitch = (bounds[1] - bounds[0]) / (resolution - 1)
    report = validate_mesh(mesh, pitch_mm=pitch,
                           declared_min_feature_mm=min_feature_mm,
                           allow_multiple_components=allow_multiple_components)
    report.update({"units": "mm", "resolution": resolution,
                   "sample_bounds_mm": bounds.tolist(),
                   "declared_lipschitz_bound": lipschitz,
                   "glsl_runtime_equivalence_verified": False,
                   "physical_printability_verified": False,
                   "wall_thickness_verified": False})
    center = (bounds[0] + bounds[1]) / 2
    radius = float(np.linalg.norm(bounds[1] - bounds[0]) / 2)
    half_vertical_fov = math.radians(45 / 2)
    half_horizontal_fov = math.atan(math.tan(half_vertical_fov) * width / height)
    distance = radius / math.sin(min(half_vertical_fov, half_horizontal_fov)) * 1.15
    views = {"front": ((0, 0, 1), (0, 1, 0)),
             "side": ((1, 0, 0), (0, 1, 0)),
             "top": ((0, 1, 0), (0, 0, -1))}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".sdf-build-", dir=output.parent))
    try:
        hit_counts = {}
        for name, (direction, up) in views.items():
            image, hits, _ = render(
                model.sdf, width=width, height=height,
                camera=center + np.asarray(direction) * distance,
                target=center, up=up, fov_degrees=45,
                # The renderer compares raw field values, not distances in mm.
                epsilon=max(1e-5, float(np.min(pitch)) * 0.02) * lipschitz,
                max_distance=distance + 2 * radius, lipschitz=lipschitz)
            if not hits.any():
                raise ValueError(f"The {name} preview has no surface hits; inspect field/tracing bounds")
            Image.fromarray(image).save(temporary / f"preview_{name}.png")
            hit_counts[name] = int(np.count_nonzero(hits))
        export_stl(mesh, temporary / "model.stl",
                   allow_multiple_components=allow_multiple_components)
        (temporary / "model.glsl").write_text(model.GLSL, encoding="utf-8")
        report["preview_hit_pixels"] = hit_counts
        (temporary / "validation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if output.exists():
            raise ValueError("Output directory appeared during the build; refusing replacement")
        try:
            # mkdir is the portable, exclusive reservation; exists() is advisory only.
            output.mkdir()
        except FileExistsError as exc:
            raise ValueError("Output directory appeared during the build; refusing replacement") from exc
        # Reserve each file exclusively too: never replace concurrent user files.
        # Publish the readiness report last, after all other artifact files close.
        artifacts = [f"preview_{name}.png" for name in views]
        artifacts += ["model.stl", "model.glsl", "validation.json"]
        for name in artifacts:
            with (temporary / name).open("rb") as source, (output / name).open("xb") as target:
                shutil.copyfileobj(source, target)
        # On late I/O failure leave the reserved directory intact: deleting it could
        # remove unrelated files created concurrently by the user.
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sphere", help="sphere, blended-capsules, or trusted .py file")
    parser.add_argument("--trust-model", action="store_true", help="Allow executing an inspected custom Python model")
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory; never overwritten")
    parser.add_argument("--resolution", type=int, default=64, help="Grid samples per axis, 8..256")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--min-feature-mm", type=float,
                        help="Declared feature size; checks grid sampling, NOT measured wall thickness")
    parser.add_argument("--allow-multiple-components", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not 8 <= args.resolution <= 256:
            raise ValueError("Resolution must be between 8 and 256")
        if not 8 <= args.width <= 2048 or not 8 <= args.height <= 2048:
            raise ValueError("Preview dimensions must be between 8 and 2048")
        if args.min_feature_mm is not None and (
                not math.isfinite(args.min_feature_mm) or args.min_feature_mm <= 0):
            raise ValueError("Declared minimum feature must be positive and finite")
        model, bounds, lipschitz = load_model(args.model, args.trust_model)
        report = build(model, bounds, lipschitz, args.output,
                       resolution=args.resolution, width=args.width, height=args.height,
                       min_feature_mm=args.min_feature_mm,
                       allow_multiple_components=args.allow_multiple_components)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), "validation": report}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
