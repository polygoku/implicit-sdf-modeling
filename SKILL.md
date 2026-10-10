---
name: "implicit-sdf-modeling"
description: "Generate 3D models as LLM-written GLSL signed distance functions (closed-form implicit surfaces, a la Keenan Crane's 'Inigo the Dragon'): draft the SDF, raymarch a preview, mesh it via marching cubes, and export STL for printing. Use when the user wants a compact, resolution-independent, mathematically-defined 3D shape, or references implicit surfaces / Shadertoy-style SDF modeling."
---

# Implicit SDF Modeling

## Purpose
Turn a text description into a 3D model expressed as a closed-form implicit surface: a GLSL `map(vec3 p)` signed-distance function. Preview it with a CPU raymarcher, convert to a mesh with marching cubes, and export STL. Inspired by Keenan Crane's Oct 2026 experiment (27kb of LLM-generated GLSL defining a dragon — no mesh, NeRF, or 3DGS involved).

## When to use / when not to
- USE for: stylized/cartoon forms, logos, abstract or parametric shapes, tiny portable model definitions, Shadertoy-style art.
- DO NOT USE for photorealistic organic models from photos — diffusion mesh generators (Meshy, Tripo: ~3 min, ~50¢, fully automatic) are far faster, cheaper, and better there. Say so and offer that route instead.

## Workflow
1. **Draft the SDF.** Write (or prompt yourself to write) a GLSL `float map(vec3 p)` using SDF primitives and combinators; see `references/sdf-primer.md` for the operator cheat-sheet and prompt template. Keep it constructive: spheres, boxes, capsules, smooth-min blends, repetition, displacement. Iterate on the preview, not on faith.
2. **Preview.** Port `map()` 1:1 into the numpy raymarcher pattern in `references/sdf-primer.md` and render a small (e.g. 160×120) preview. Check silhouette from 2–3 angles before refining detail.
3. **Mesh.** Sample `map()` on a regular grid and run marching cubes (`skimage.measure.marching_cubes`; `pip install scikit-image` if missing), then write STL (`pip install numpy-stl`, or emit ASCII STL manually).
4. **Deliver.** Save the `.glsl`, preview PNGs, and `.stl` under `~/workspace/your_files/` (or the relevant project dir). Report dimensions and triangle count.

## Operating Rules
- Implicit surfaces are expensive to visualize: keep preview resolution low and marching-cubes grids modest (e.g. 128³); raise only for the final export.
- Never claim the first draft is correct — the reference experiment needed 27 rounds of human feedback. Render, look, revise.
- SDF must be a true signed distance (or close): broken distance bounds cause raymarch artifacts and meshing holes. Prefer smooth-min over hard unions for organic blends.
- Keep the GLSL portable (Shadertoy-compatible `map(vec3)` signature) so the model stays usable outside this environment.
