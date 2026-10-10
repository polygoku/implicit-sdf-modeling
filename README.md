# implicit-sdf-modeling

An agent skill and runnable reference pipeline for **LLM-assisted modeling with GLSL implicit surfaces**. Geometry is authored as a compact `float map(vec3 p)` function, paired with a trusted NumPy evaluator. The tools produce front/side/top previews, a validated binary STL, and a JSON report.

This is an iterative modeling workflow, not a one-shot photo-to-3D generator, a general GLSL interpreter, or a printability certification tool.

## Quick start

Requires Python 3.12+ and virtual-environment support. From the repository root:

```sh
python -m venv .venv
```

Use `.venv/bin/python` on Linux/macOS, or `.venv\\Scripts\\python.exe` on Windows, for the commands below. Do not install into an unrelated application environment.

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/sdf_pipeline.py --model sphere --output artifacts/sphere --resolution 64 --min-feature-mm 1.5
.venv/bin/python scripts/sdf_pipeline.py --model blended-capsules --output artifacts/capsules --resolution 64
```

`requirements.txt` is sufficient to run the pipeline; the development file adds tests/lint dependencies. On Debian-derived systems, a missing `ensurepip` error may require the system's Python venv component or another approved interpreter with venv support.

### Artifacts

Each new output directory receives:
- `model.glsl`: geometry source, with no lighting or textures.
- `preview_front.png`, `preview_side.png`, `preview_top.png`: CPU-raymarched views.
- `model.stl`: binary STL, serialized and reopened for integrity validation.
- `validation.json`: dimensions, triangle count, topology, sampling and stated validation limits.

Output directories must be new; an existing directory is never overwritten. World coordinates are **millimeters**. STL itself does not encode units, so confirm millimeters in your slicer.

## Custom models

Use the built-in sphere or blended capsules as a template. A Python model defines:

```python
import numpy as np

BOUNDS = ((-12.0, -12.0, -12.0), (12.0, 12.0, 12.0))  # mm
LIPSCHITZ = 1.0  # valid bound for the evaluator, not a guessed quality knob
GLSL = "float map(vec3 p) { return length(p) - 10.0; }"

def sdf(points):
    # points is an array (..., 3); return one scalar for every point.
    return np.linalg.norm(points, axis=-1) - 10.0
```

Inspect custom code before running:

```sh
.venv/bin/python scripts/sdf_pipeline.py --model path/to/custom.py --trust-model --output artifacts/custom --resolution 96
```

**Custom Python modules execute code. `--trust-model` is not a sandbox.** The shipped examples use shared constants for their corresponding NumPy/GLSL definitions, but this pipeline does not execute the GLSL on a GPU or prove arbitrary port equivalence. Keep helpers, constants, signs, transforms, and units consistent. Use a sandbox for untrusted generated/downloaded code.

## Validation scope

The pipeline checks finite scalar samples, zero-level crossing, positive domain boundaries, mesh integrity, winding, volume, connected components and STL round-trip integrity. Bounds must still be chosen to include the entire intended model: positive sampled boundaries cannot detect wholly separate objects outside the box.

`--min-feature-mm` checks whether grid pitch is at most one third of a **declared** feature size. It does not measure the model's minimum wall thickness. Higher grid resolution alone does not prove a thin feature has been preserved. The report explicitly keeps `wall_thickness_verified` and `physical_printability_verified` false.

Multiple disconnected surface components are rejected unless `--allow-multiple-components` is supplied. Every accepted shell must have positive signed volume; enclosed cavity shells are currently unsupported even with this flag. The component count describes connected mesh surfaces, not necessarily physical parts. This option does not make detached or floating parts printable. Self-intersections are not tested.

Before printing, review actual dimensions, minimum walls, gaps/clearances, overhangs, support/orientation, material and process settings in an appropriate slicer/CAD tool. The pipeline is not appropriate as the sole verification for load-bearing, safety-critical or tolerance-controlled engineering parts.

## Development

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m pip check
```

Tests include camera hit/miss behavior, invalid fields and parameters, sampling bounds, mesh dimensions/topology, scalar-field rescaling, binary STL round trips, and the complete command-line workflow. CI runs the same gates and an example build; a workflow file being present is not a claim that remote CI has passed.

## Layout

- `SKILL.md`: concise agent workflow and verification criteria.
- `references/sdf-primer.md`: distance bounds, GLSL primitives, porting and meshing guidance.
- `scripts/render_sdf.py`: perspective camera and bounded CPU tracing.
- `scripts/mesh_sdf.py`: sampling, mesh validation and STL export.
- `scripts/sdf_pipeline.py`: portable end-to-end CLI.
- `examples/`: shared-constant sphere and blended-capsule models.
- `tests/`: regression and artifact checks.

## Background and sources

The technique draws on constructive implicit modeling and Inigo Quilez's SDF work, and was inspired by Keenan Crane's implicit-model examples. A particular model's runtime or feedback count is not a benchmark for all SDF modeling.

- [Keenan Crane's primary 3D model repository](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/)
- [Inigo Quilez: distance functions](https://iquilezles.org/articles/distfunctions/)
- [Hart: sphere tracing and Lipschitz distance bounds](https://doi.org/10.1007/s003710050084)
- [scikit-image marching cubes reference](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes)
- [trimesh documentation](https://trimesh.org/)

Alternative CAD/reconstruction/generative tools can be better for other tasks. Prices, generation times and relative quality vary; this skill makes no fixed commercial-service cost or performance guarantees.

## License

MIT; see `LICENSE`. Attribution to an inspiration does not imply that its third-party assets are included or relicensed here.
