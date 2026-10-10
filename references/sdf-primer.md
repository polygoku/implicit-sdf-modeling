# SDF Primer & Prompt Template

## Operator cheat-sheet (GLSL)
```glsl
float sdSphere(vec3 p, float r) { return length(p) - r; }
float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0); }
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
  return length(pa - ba*h) - r;
}
// smooth-min (polynomial), k = blend radius
float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}
float opUnion(float a, float b) { return min(a,b); }
// repetition
vec3 opRep(vec3 p, vec3 c) { return mod(p + 0.5*c, c) - 0.5*c; }
```

## LLM prompt template (for drafting `map()`)
> Write a GLSL `float map(vec3 p)` signed-distance function for [SUBJECT].
> Rules: only SDF primitives (sphere/box/capsule/torus) combined with min/max/smin;
> keep every part a valid distance bound; expose 2–4 tweakable constants at the top
> (e.g. BODY_R, BLEND_K); no textures, no lighting — geometry only.
> Return ONLY the GLSL function plus a one-line parts list.

## Numpy raymarch preview pattern
```python
import numpy as np
from PIL import Image

# def sdf(p): ...  # port of map(): p is (...,3) array, returns (...,) distances
def render(w=160, h=120):
    ys, xs = np.mgrid[0:h, 0:w]
    ro = np.stack([(xs/w-0.5)*2, (0.5-ys/h)*2, np.full_like(xs, 3.0)], -1)
    rd = ro / np.linalg.norm(ro, axis=-1, keepdims=True)
    t = np.zeros((h, w)); hit = np.zeros((h, w), bool)
    for _ in range(64):
        p = ro + rd * t[..., None]
        d = sdf(p)
        hit |= d < 1e-3
        t += np.where(hit, 0, d)
        if (d[~hit] > 4).all(): break
    img = (1 - np.clip(t/6, 0, 1)) * 255
    Image.fromarray(img.astype('uint8')).save('/tmp/sdf_preview.png')
```

## Meshing pattern
```python
import numpy as np
N = 128
xs = np.linspace(-2, 2, N)
X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
vol = sdf(np.stack([X, Y, Z], -1))          # may be slow; vectorize sdf with numpy ops
from skimage.measure import marching_cubes   # pip install scikit-image
verts, faces, _, _ = marching_cubes(vol, level=0.0,
                                    spacing=(4/(N-1),)*3)
verts -= 2.0                                 # recenter to world coords
# write STL (binary) with numpy-stl, or ASCII STL by hand
```

## Crane experiment notes (Oct 2026, @keenanisalive)
- 27kb GLSL, closed-form implicit dragon; 25h of model calls + 27 human-feedback rounds — not automatic.
- Rendering implicits is costly (45 min for a 12s clip on a custom Metal renderer); meshing was needed just to inspect it.
- Pipeline that worked: diffusion-generated 2D front/side reference images ("Nano Banana") → LLM-authored SDF reconciling both projections (ill-posed: many geometry/material/light/camera combos explain one image).
- Credit: technique borrows from Inigo Quilez's hand-built SDF art (iquilezles / shadertoy).
