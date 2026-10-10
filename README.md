# implicit-sdf-modeling

An agent skill for **LLM-assisted 3D modeling as closed-form implicit surfaces** —
3D models expressed as compact GLSL signed-distance functions instead of meshes.

Inspired by Keenan Crane's October 2026 experiment
([@keenanisalive](https://x.com/keenanisalive)): a dragon defined by 27kb of
LLM-generated GLSL — no mesh, NeRF, or 3DGS involved.

## What it does

1. **Draft** a `float map(vec3 p)` SDF from primitives (spheres, boxes, capsules)
   combined with smooth-min blends.
2. **Preview** it with a numpy CPU raymarcher (see `references/sdf-primer.md`).
3. **Mesh** it via marching cubes and **export STL** for 3D printing.

## Layout

- `SKILL.md` — the skill: when to use it, the workflow, operating rules.
- `references/sdf-primer.md` — SDF operator cheat-sheet, LLM prompt template,
  numpy raymarch/marching-cubes patterns, and notes from Crane's experiment.

## Honest scope (from the source experiment)

- 27kb of GLSL took ~25h of model calls + 27 rounds of human feedback — this is
  an iterative, hands-on technique, not a one-shot generator.
- Rendering implicit surfaces is expensive; keep previews small.
- For photorealistic organic 3D from photos, diffusion mesh generators
  (Meshy, Tripo — ~3 min, ~50¢, fully automatic) are the better tool.
  This skill is for stylized, parametric, resolution-independent shapes.
