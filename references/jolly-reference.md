# Jolly: authoritative visual reference

![User-supplied Jolly reference](images/jolly-source.jpg)

This is the source image supplied by the user for the Jolly refinement. Preserve its original bytes. SHA-256: `7728cfa5602bf3d557f0076709a9311b374235234a6c108492ae506b37c8a583`.

## Required workflow for every Jolly change

1. Open this image before changing the model; use it rather than memory or a previous generated render as the visual authority.
2. Compare the current and proposed front render against this image. Also inspect side, three-quarter and top views for accidental distortions and disconnected geometry.
3. Record visible improvements, remaining mismatches and inferred geometry. Do not describe a single photo as exact 3D ground truth.
4. Run the full test suite, lint, pipeline build and STL reopen checks. Keep printability and GLSL runtime-equivalence limits explicit.

## Visible constraints

- Front faces +Z; Y is up. The raised waving hand is on the **viewer's right**, therefore positive X in the model's front view.
- The hood and torso form a continuous, broad plush silhouette; avoid a spherical bobble-head and narrowed neck.
- The pale face is a wide, rounded rectangle rather than a circular disk. Eyes are small and widely spaced; the smile is restrained.
- The raised arm angles outward/upward. Its mitten top is below the hood top; the opposite arm hangs beside the torso.
- Feet are short rounded blocks with comparatively flat soles and a visible central gap, not long capsule-shaped legs.
- Pink fur, cream fabric, black glossy eyes and the black smile are appearance cues. Smooth solid geometry does not reproduce fur or fabric. Diagnostic preview colors are not STL materials.

## What remains inferred

The photo does not uniquely determine rear geometry, thickness, exact scale, hidden connections or tolerances. The approximately 150 mm figurine scale is a modeling choice, not a measured dimension of the photographed object. Side/back proportions are a conservative stylized interpretation. Visual comparisons are qualitative, not calibrated pixel-fitting measurements.

## Framing and silhouette refinement

The follow-up to the merged reference model reduces face XY half-extents from
`(28, 21)` to `(25.2, 18.9)` mm and raises its center from Y=43 to Y=45 mm.
This restores a visible pink band at the cheeks and below the chin while
retaining forehead framing. Face depth and eye placement are unchanged.
The smile's actual prior tube radius was 0.65 mm; it is now 0.5 mm, with its
center recessed by 0.15 mm. The hood/body blend is widened from 8 to 20 mm:
a smaller 12 mm blend still left a visible waist, so the final value also
regresses the central front silhouette profile rather than just one point.

These are design adjustments judged against the original photograph, not
measurements recovered from it. The smooth hood/face and block feet remain
stylized; fur, fabric, hidden depth and physical printability remain unresolved.
Preview colors remain diagnostic, not exported STL materials.

## Reference rights

The image was supplied by the user for visual-reference inclusion. Photographer and rights-holder attribution have not been supplied. This reference photograph is **not licensed by this repository's MIT code license**; its original rights are retained. Do not represent it as an original repository asset or imply ownership of the depicted character.
