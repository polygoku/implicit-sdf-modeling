# SDF Primer and Model Contract

## Field, sign and units

Use a finite scalar field `f(p)` whose zero set is the intended surface: negative inside, positive outside. All model coordinates and dimensions in this pipeline are millimeters. The sphere `f(p) = length(p) - r` is an exact signed distance. Boolean compositions and smooth blends can preserve usable bounds without being exact signed distances everywhere.

**Raymarching and meshing have different requirements.** Sphere tracing must not step past a surface. A globally valid Lipschitz constant `L` gives the conservative step `abs(f(p))/L`; declare that bound instead of relying on the field being 'close enough.' Marching cubes samples a zero-isosurface and does not require exact distance values. Multiplying all field values by a positive constant preserves the zero set, but tracing requires scaling `L` by the same factor.

Warping, displacement and nonuniform scaling require additional analysis. For `f(p)=d(p)+A*sin(omega*x)`, if `d` is 1-Lipschitz, `1+abs(A*omega)` is a conservative Lipschitz bound. A safety factor chosen without a justified bound is not a proof of correct tracing. Finite-difference tests are useful diagnostics, not a proof of a global bound.

## GLSL primitives

These standard analytic forms are geometry only. Validate finite positive radii/box extents and nonzero repetition periods in the paired model. Guard a collapsed capsule segment and nonpositive blend radius explicitly:

```glsl
float sdSphere(vec3 p, float r) { return length(p) - r; }
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 ba = b - a;
    float denom = dot(ba, ba);
    if (denom <= 0.0) return length(p - a) - r;
    float h = clamp(dot(p - a, ba) / denom, 0.0, 1.0);
    return length(p - a - ba * h) - r;
}
float smin(float a, float b, float k) {
    if (k <= 0.0) return min(a, b);
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
float opUnion(float a, float b) { return min(a, b); }
float opIntersection(float a, float b) { return max(a, b); }
float opDifference(float a, float b) { return max(a, -b); }
```

The guarded GLSL definitions choose specific fallback semantics. A Python implementation may instead reject invalid inputs; document this difference and never claim equivalence on invalid parameters. Smooth-min changes the surface and can expand bounds. Finite repeating geometry needs an explicit finite clipping/enclosure construction; unbounded periodic models cannot be exported as complete finite STLs.

## Paired model contract

A trusted Python file must define:

- `sdf(points)`: NumPy-vectorized input of shape `(..., 3)` and output `(...)`; finite real values on the sampled domain.
- `GLSL`: self-contained helpers/constants and a `float map(vec3 p)` function.
- `BOUNDS`: two finite ordered 3D corners in mm, containing all intended geometry with a positive exterior margin.
- `LIPSCHITZ`: a finite positive upper bound appropriate for this evaluator.

Prefer a common set of constants that generates the GLSL values and drives NumPy evaluation. GLSL `mod` and floating-point behavior differ from some host-language operations; port signs, wrapping, transforms and vector axes deliberately. The pipeline does not run a GLSL interpreter or guarantee arbitrary port equivalence. For modified complex models, validate the shader with an actual compiler and compare field samples or rendered views against the NumPy port.

### Drafting prompt

> Write geometry for [SUBJECT] using a self-contained GLSL `float map(vec3 p)` and a corresponding vectorized NumPy evaluator. Model dimensions are [DIMENSIONS] mm. Supply helper definitions, shared constants, finite enclosing BOUNDS, and a justified positive LIPSCHITZ bound. Use bounded primitives and constructive operations; do not introduce unbounded repetition or unjustified displacement. Identify expected features, analytic checkpoints and known limitations. Do not include lighting or textures. Keep generated Python code inspectable and free of network/file/subprocess side effects.

## Correct preview invocation

Use the shipped renderer rather than inventing a new camera. A perspective camera has one origin; each pixel points toward an image plane oriented around the target. Pointing `rd` along the camera-position vector sends rays away from geometry near the origin. Negating separate per-pixel origins would instead converge all rays on one point, also producing the wrong image.

```python
from pathlib import Path
from PIL import Image
from examples.sphere import sdf
from scripts.render_sdf import render

image, hits, depth = render(sdf, camera=(0, 0, 35), target=(0, 0, 0),
                            max_distance=80, lipschitz=1.0)
assert hits.any(), "No surface hit: check camera, bounds and evaluator"
out = Path("artifacts")
out.mkdir(parents=True, exist_ok=True)
Image.fromarray(image).save(out / "quick_preview.png")
```

The renderer uses an absolute hit tolerance, active rays, a bounded iteration count, finite field checks and a far limit. Misses are background, not fake shaded surfaces. Large `L` bounds can require more iterations; an iteration cap can cause false misses. Inspect more than one camera angle.

## Mesh extraction and integrity checks

```python
from examples.sphere import sdf, BOUNDS
from scripts.mesh_sdf import make_mesh, validate_mesh, export_stl

mesh = make_mesh(sdf, bounds=BOUNDS, resolution=64, min_feature_mm=1.5)
report = validate_mesh(mesh)
export_stl(mesh, "artifacts/sphere.stl")
```

For a grid of `N` samples across `[lo, hi]`, spacing is `(hi-lo)/(N-1)` on each axis. Marching-cubes vertices must be translated by `lo`, not a hardcoded offset. The sampled field must be finite with `min(f)<0<max(f)`, and all six sampled boundary faces must remain positive. These are necessary checks for the sampled domain; they cannot prove that a separate component was not placed entirely outside it.

Checks cover nondegenerate geometry, closure, consistent winding, positive volume, components, dimensions and binary-STL readback. If a minimum feature is declared, require at least three grid intervals per feature (`max(spacing) <= feature/3`); this is a sampling policy, not measured wall thickness or guaranteed preservation of every detail.

An STL carries no unit metadata. Report mm explicitly and inspect the slicer's interpreted size, wall thickness, gaps, supports, orientation and process settings. Successful mesh validation must not be described as guaranteed physical printability.

## Primary references

- [Inigo Quilez's distance-function collection](https://iquilezles.org/articles/distfunctions/)
- [Hart's sphere-tracing paper](https://doi.org/10.1007/s003710050084)
- [scikit-image marching cubes API](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes)
- [Keenan Crane's model repository](https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/)

The cited works are background, not bundled licensed assets. Specific model-generation prices, timings and iteration counts should be tied to a dated primary account; none is promised by this workflow.
