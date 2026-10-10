---
name: implicit-sdf-modeling
description: "Model implicit surfaces; preview and validate STL exports."
version: 0.2.0
author: Ky Fu (polygoku), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [3d-modeling, sdf, glsl, implicit-surfaces, stl, 3d-printing]
---

# Implicit SDF Modeling

Build compact, mathematically defined geometry as GLSL `float map(vec3 p)`, with a corresponding trusted NumPy evaluator for previews and meshing. The included tools render three views, extract a zero-isosurface, validate mesh integrity, and export binary STL. They do not automatically interpret arbitrary GLSL or certify physical printability.

## When to Use
- Stylized forms, logos, parametric solids, constructive geometry, and portable implicit models.
- A user explicitly requests SDF/GLSL modeling or an STL derived from an implicit surface.
- Don't use as a substitute for dimensioned mechanical CAD, scan reconstruction, or photo-to-3D generation. Compare alternative methods against the actual task rather than promising fixed prices or generation times.

## Prerequisites
- Python 3.12+ with virtual-environment support; use an isolated project environment.
- Install `requirements.txt`; use `requirements-dev.txt` for testing and linting.
- Resolve `<repo>` to this skill's repository and `<venv-python>` to its environment's interpreter (`.venv/bin/python` on POSIX, `.venv\\Scripts\\python.exe` on Windows).
- If creating the environment fails because `ensurepip` is unavailable, use an approved interpreter with venv support or install the OS venv component with permission. Never alter an unrelated working agent/voice environment to get these dependencies.

## How to Run
Use `terminal` with the repository as its working directory:

```python
terminal(command="python -m venv .venv", workdir="<repo>", timeout=60)
terminal(command="<venv-python> -m pip install -r requirements-dev.txt", workdir="<repo>", timeout=180)
terminal(command="<venv-python> scripts/sdf_pipeline.py --model sphere --output artifacts/sphere --resolution 64 --min-feature-mm 1.5", workdir="<repo>", timeout=180)
terminal(command="<venv-python> scripts/sdf_pipeline.py --model blended-capsules --output artifacts/capsules --resolution 64", workdir="<repo>", timeout=180)
```

Each new output directory contains `model.glsl`, `model.stl`, three `preview_*.png` images, and `validation.json`. Existing output directories are never overwritten.

## Procedure
1. **Specify the physical intent.** Agree target dimensions in millimeters, intended process, and critical features. Choose finite bounds that enclose all modeled geometry; record any declared minimum feature size. Completion: units, bounds, and process assumptions are explicit.
2. **Draft a paired model.** Provide `sdf(points)`, `GLSL`, `BOUNDS`, and a positive finite `LIPSCHITZ` bound in a trusted Python module. Keep constants shared between representations. Read `references/sdf-primer.md` for safe primitives and the model contract. Completion: finite evaluations, known analytic checkpoints, and justified tracing bounds; do not claim an arbitrary GLSL/NumPy pair is equivalent without verification.
3. **Execute the pipeline.** Use a built-in example first. For a custom module, inspect its code before using `--model path/to/model.py --trust-model`; this flag acknowledges code execution and is not a sandbox. Completion: the command exits successfully and all named artifacts exist.
4. **Inspect every preview.** Use `vision_analyze` on front, side, and top views. Reject missing geometry, wrong proportions, clipping, or unwanted joins. Revisit the model or tracing bound, not just the mesh resolution. Completion: the required shape has been visually checked from all three views.
5. **Check the validation report.** Require finite, nondegenerate geometry; closed boundaries; consistent winding; positive volume; intended connected components; expected dimensions and triangle count. If `--min-feature-mm` is given, grid pitch must be no more than one third of that declared feature. Completion: mesh integrity and requested sampling checks pass. This does not measure actual wall thickness or guarantee globally complete geometry outside the sampling box.
6. **Deliver with limits.** Supply GLSL, PNGs, STL, and the report. State dimensions in mm, resolution, triangle count, connectivity, and remaining slicer/process checks. Physical printability and wall thickness remain explicitly unverified until separately assessed. Completion: the user receives the artifacts and an accurate validation scope.

## Pitfalls
- Sphere tracing needs a conservative distance step or a valid Lipschitz divisor. A merely 'close' field can overshoot; displacement, nonuniform scaling, and repetition need additional care.
- Marching cubes accepts scalar fields, not only true signed distances. Its sampling and finite-domain limitations are separate from raymarch correctness.
- Smooth blends change geometry, dimensions, and clearances. Boundaries positive on all six sampled faces are necessary, but cannot prove there are no separate objects outside the domain.
- STL has no intrinsic unit metadata. Treat this pipeline's vertex coordinates as mm and confirm the slicer's import units.
- No cloud API is required by the shipped pipeline. Do not upload a private design or source image to another service without authorization.

## Verification

```python
terminal(command="<venv-python> -m pytest -q", workdir="<repo>", timeout=180)
terminal(command="<venv-python> -m ruff check .", workdir="<repo>", timeout=60)
terminal(command="<venv-python> -m pip check", workdir="<repo>", timeout=60)
```

Run a fresh complete example, reopen the exported STL, inspect the previews, and verify the report. A successful script invocation alone is not a design acceptance test. See the README for installation, artifacts, limits, and primary references.
