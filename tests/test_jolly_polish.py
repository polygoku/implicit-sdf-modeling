"""Photo-framing regressions for the follow-up refinement."""
from typing import cast

import numpy as np

from examples import jolly_waving as jolly


def _front_surface_point(x, y):
    """Outer front surface at chosen solid hood/body rays, not a general tracer."""
    lower, upper = 0., jolly.BOUNDS[1][2]
    assert jolly.sdf(np.array([x, y, lower])) < 0
    assert jolly.sdf(np.array([x, y, upper])) > 0
    for _ in range(55):
        middle = (lower + upper) / 2
        if jolly.sdf(np.array([x, y, middle])) < 0:
            lower = middle
        else:
            upper = middle
    return np.array([x, y, (lower + upper) / 2])


def test_smaller_raised_face_restores_pink_cheeks_and_chin_frame():
    face = next(record for record in jolly.SCENE if record[0] == "face")
    parameters = cast(tuple[tuple[float, float, float], float, float], face[3])
    np.testing.assert_allclose(parameters[0][:2], np.array([28., 21.]) * .9)
    assert face[2][1] == 45.
    # World-space rays where the previous cream face occupied the pink frame.
    points = np.stack([_front_surface_point(x, y)
                       for x, y in [(-27., 45.), (27., 45.), (0., 24.), (0., 70.)]])
    np.testing.assert_allclose(jolly.sdf(points), 0., atol=1e-10)
    np.testing.assert_allclose(jolly.material(points),
                               np.broadcast_to(jolly.PALETTE[0], points.shape))


def test_delicate_smile_has_half_mm_radius_and_reduced_projection():
    smile = next(record for record in jolly.SCENE if record[0] == "smile")
    major, tube, angles = cast(tuple[float, float, tuple[float, float]], smile[3])
    assert tube == .5
    assert smile[2][2] == 34.15
    midpoint = (angles[0] + angles[1]) / 2
    top = np.asarray(smile[2]) + [major * np.cos(midpoint), major * np.sin(midpoint), tube]
    np.testing.assert_allclose(jolly._primitive(top, smile), 0, atol=1e-12)
    np.testing.assert_allclose(jolly.sdf(top), 0, atol=1e-12)
    # Face front stays at Z=34; new projection .65 mm, previously .95 mm.
    np.testing.assert_allclose(top[2] - 34., .65, atol=1e-12)


def test_head_body_blend_fills_previous_neck_silhouette_constriction():
    records = {record[0]: record for record in jolly.SCENE}
    # Exclude arms: a real silhouette checkpoint of the central hood/body union.
    points = np.array([[-37., 24., 0.], [37., 24., 0.]])
    body = jolly._primitive(points, records["body"])
    hood = jolly._primitive(points, records["hood"])
    core = jolly._smin(body, hood, records["hood"][-1])
    assert np.all(core < 0), "The narrower old join leaves both checkpoints outside"
    assert np.all(jolly.sdf(points) <= core)


def test_neck_core_profile_preserves_full_hood_side_width():
    records = {record[0]: record for record in jolly.SCENE}
    y = np.linspace(0., 42., 85)
    points = np.column_stack((np.full_like(y, 38.), y, np.zeros_like(y)))
    core = jolly._smin(jolly._primitive(points, records["body"]),
                       jolly._primitive(points, records["hood"]), records["hood"][-1])
    assert np.all(core < 0), "The front core must not pinch inside the hood side width"
