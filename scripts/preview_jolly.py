"""Normal-shaded CPU reference comparisons; not GPU/GLSL equivalence tests.

Colors are diagnostic appearance only; STL geometry carries no materials.
Outputs reserve a new directory. A failure may leave incomplete outputs.
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def surface_points(depths, hits, camera, target, up, fov_degrees=45):
    """Reconstruct only hit positions using the renderer's perspective convention."""
    height, width = hits.shape
    origin = np.asarray(camera, dtype=float)
    forward = np.asarray(target, dtype=float) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=float))
    right /= np.linalg.norm(right)
    vertical = np.cross(right, forward)
    scale = np.tan(np.deg2rad(fov_degrees) / 2)
    x = (2 * (np.arange(width) + .5) / width - 1) * scale * width / height
    y = (1 - 2 * (np.arange(height) + .5) / height) * scale
    xx, yy = np.meshgrid(x, y)
    rays = forward + xx[..., None] * right + yy[..., None] * vertical
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return origin + depths[hits, None] * rays[hits]


def normals(sdf, points, step_mm=1e-3):
    """Central-difference outward normals; zero gradients remain zero."""
    p = np.asarray(points, dtype=float)
    if isinstance(step_mm, (bool, np.bool_)):
        raise ValueError("normal difference step must be finite positive mm")
    h = float(step_mm)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("normal difference step must be finite positive mm")
    def sample(q):
        value = np.asarray(sdf(q))
        if (value.shape != p.shape[:-1] or value.dtype.kind not in "fiu"
                or not np.isfinite(value).all()):
            raise ValueError("normal samples must be finite real scalars matching points")
        return value.astype(float, copy=False)
    gradient = np.stack([
        (sample(p + axis * h) - sample(p - axis * h)) / (2 * h)
        for axis in np.eye(3)
    ], axis=-1)
    length = np.linalg.norm(gradient, axis=-1, keepdims=True)
    return gradient / np.maximum(length, 1e-12)


def main(argv=None):
    from scripts.render_sdf import render
    from scripts.sdf_pipeline import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--size", type=int, default=480, help="Square view size, 32..1024")
    args = parser.parse_args(argv)
    if not 32 <= args.size <= 1024:
        parser.error("size must be in 32..1024")
    model, bounds, lipschitz = load_model("jolly-waving")
    reference = ROOT / "references/images/jolly-source.jpg"
    reference_bytes = reference.read_bytes()
    center = bounds.mean(axis=0)
    corners = np.array(list(itertools.product(*zip(bounds[0], bounds[1])))) - center
    radius = float(np.linalg.norm(bounds[1] - bounds[0]) / 2)
    views = {"front": ((0, 0, 1), (0, 1, 0)),
             "side": ((1, 0, 0), (0, 1, 0)),
             "top": ((0, 1, 0), (0, 0, -1)),
             "three_quarter": ((-.7, .15, 1), (0, 1, 0))}
    args.output.mkdir(parents=True, exist_ok=False)
    images, hit_counts = {}, {}
    light = np.array([-.4, .7, 1.])
    light /= np.linalg.norm(light)
    for name, (direction, up) in views.items():
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        right = np.cross(-direction, up)
        right /= np.linalg.norm(right)
        vertical = np.cross(right, -direction)
        half_frame = max(np.max(np.abs(corners @ right)),
                         np.max(np.abs(corners @ vertical)))
        distance = (half_frame / np.tan(np.deg2rad(22.5))
                    + np.max(np.abs(corners @ direction))) * 1.08
        camera = center + direction * distance
        _, hits, depths = render(model.sdf, width=args.size, height=args.size,
                                 camera=camera, target=center, up=up,
                                 epsilon=.008 * lipschitz, max_steps=256,
                                 max_distance=distance + 2 * radius, lipschitz=lipschitz)
        if not hits.any():
            raise ValueError(f"no geometry in {name} view")
        points = surface_points(depths, hits, camera, center, up)
        normal = normals(model.sdf, points)
        if callable(getattr(model, "material", None)):
            colors = np.asarray(model.material(points), dtype=float)
        else:
            colors = np.broadcast_to(np.array([.94, .66, .76]), points.shape)
        if (colors.shape != points.shape or not np.isfinite(colors).all()
                or np.any(colors < 0) or np.any(colors > 1)):
            raise ValueError("diagnostic material must return finite RGB in 0..1")
        brightness = .25 + .75 * np.maximum(normal @ light, 0)
        image = np.zeros((args.size, args.size, 3), dtype=np.uint8)
        image[hits] = np.clip(colors * brightness[:, None] * 255, 0, 255).astype(np.uint8)
        images[name] = Image.fromarray(image)
        images[name].save(args.output / f"shaded_{name}.png")
        hit_counts[name] = int(np.count_nonzero(hits))
    # Crop only the display copy of the original photo. Preserve source bytes.
    with Image.open(reference) as source:
        photo = source.convert("RGB").crop((350, 105, 960, 780))
    comparison = Image.new("RGB", (960, 700), "#17202b")
    draw = ImageDraw.Draw(comparison)
    for col, (label, image) in enumerate([("Source photograph", photo),
                                         ("Refined smooth model (CPU)", images["front"])]):
        if col == 1:
            box = image.getbbox()
            if box:
                image = image.crop(box)
        fitted = ImageOps.contain(image, (440, 630))
        comparison.paste(fitted, (col * 480 + (480 - fitted.width) // 2,
                                 40 + (630 - fitted.height) // 2))
        draw.text((col * 480 + 20, 12), label, fill="white")
    comparison.save(args.output / "reference_comparison.png")
    report = {"source_reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
              "model_source_sha256": hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
              "preview_hit_pixels": hit_counts, "glsl_executed": False,
              "stl_materials_exported": False, "physical_printability_verified": False,
              "comparison": "qualitative single-view reference; rear/depth geometry inferred"}
    (args.output / "preview.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
