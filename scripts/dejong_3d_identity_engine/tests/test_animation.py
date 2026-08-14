from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dejong3d.animation import axis_gizmo_endpoints, build_animation, export_frames, frame_to_text, interpolate_spec, neutral_spec, surge_ease
from dejong3d.geometry import build_geometry, transform_and_project_with_depth
from dejong3d.models import RenderConfig, VariantSearchConfig
from dejong3d.render import SHADE_RAMP, project_points_with_depth, terminal_to_text, text_block
from dejong3d.search import build_canonical_assets


class AnimationTests(unittest.TestCase):
    def setUp(self):
        self.render = RenderConfig(term_width=36, term_height=18, supersample=2)
        self.search = VariantSearchConfig(candidates_per_code=2, modes=("surface",))

    def test_endpoints_are_exact(self):
        assets, target, frames, specs = build_animation(
            "test:node", render_cfg=self.render, search_cfg=self.search, frames=5, mode="surface"
        )
        self.assertTrue(np.array_equal(frames[0], assets["canonical_term"]))
        self.assertTrue(np.array_equal(frames[-1], target.term))

    def test_deterministic_target(self):
        _, target_a, frames_a, _ = build_animation(
            "test:determinism", render_cfg=self.render, search_cfg=self.search, frames=4, mode="surface"
        )
        _, target_b, frames_b, _ = build_animation(
            "test:determinism", render_cfg=self.render, search_cfg=self.search, frames=4, mode="surface"
        )
        self.assertEqual(target_a.manifest_record()["spec"], target_b.manifest_record()["spec"])
        self.assertTrue(np.array_equal(frames_a[-1], frames_b[-1]))

    def test_interpolation_hits_target(self):
        _, target, _, _ = build_animation(
            "test:interp", render_cfg=self.render, search_cfg=self.search, frames=3, mode="surface"
        )
        start = neutral_spec(target.spec)
        end = interpolate_spec(start, target.spec, 1.0)
        self.assertAlmostEqual(end.yaw_deg, target.spec.yaw_deg)
        self.assertAlmostEqual(end.pitch_deg, target.spec.pitch_deg)
        self.assertAlmostEqual(end.depth, target.spec.depth)
        self.assertAlmostEqual(end.twist, target.spec.twist)

    def test_default_canonical_matches_selected_mark(self):
        selected = Path(__file__).parents[2] / "poormans_shape_engine" / "selected_main" / "poormans_hpc_main_shade.txt"
        if not selected.exists():
            self.skipTest("selected poormans mark is not available in this checkout")
        render = RenderConfig()
        assets = build_canonical_assets(render)
        actual = text_block(terminal_to_text(assets["canonical_term"], SHADE_RAMP, render.gamma), render.term_width)
        expected = selected.read_text(encoding="utf-8").rstrip("\n")
        self.assertEqual(actual, expected)

    def test_axis_gizmo_can_overlay_plain_text(self):
        _, target, _, _ = build_animation(
            "test:axis", render_cfg=self.render, search_cfg=self.search, frames=3, mode="surface"
        )
        term = np.zeros((self.render.term_height, self.render.term_width), dtype=np.float64)
        spec = replace(target.spec, yaw_deg=35.0, pitch_deg=18.0, roll_deg=-8.0)
        text = frame_to_text(term, self.render, "shade", spec, axis_gizmo=True, axis_color=False)
        self.assertIn("+", text)
        self.assertIn("X", text)
        self.assertIn("Y", text)
        self.assertIn("Z", text)

    def test_axis_gizmo_endpoints_match_front_orientation(self):
        _, target, _, _ = build_animation(
            "test:axis-front", render_cfg=self.render, search_cfg=self.search, frames=3, mode="surface"
        )
        spec = replace(target.spec, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)
        endpoints = axis_gizmo_endpoints(spec, self.render)
        ox, oy = endpoints["o"]
        self.assertGreater(endpoints["x"][0], ox)
        self.assertAlmostEqual(endpoints["x"][1], oy)
        self.assertAlmostEqual(endpoints["y"][0], ox)
        self.assertLess(endpoints["y"][1], oy)
        self.assertAlmostEqual(endpoints["z"][0], ox)
        self.assertAlmostEqual(endpoints["z"][1], oy)

    def test_surge_ease_has_fast_middle(self):
        left = surge_ease(0.51) - surge_ease(0.49)
        early = surge_ease(0.11) - surge_ease(0.09)
        late = surge_ease(0.91) - surge_ease(0.89)
        self.assertGreater(left, early * 3.0)
        self.assertGreater(left, late * 3.0)

    def test_axis_html_export_does_not_corrupt_spans(self):
        _, target, frames, specs = build_animation(
            "test:html-axis", render_cfg=self.render, search_cfg=self.search, frames=3, mode="surface"
        )
        with TemporaryDirectory() as tmp:
            path = export_frames(Path(tmp), frames, specs, target, self.render, 20.0, axis_gizmo=True)
            html = path.read_text(encoding="utf-8")
        self.assertNotIn("axis-<span", html)
        self.assertIn("axis-x", html)
        self.assertIn("axis-y", html)
        self.assertIn("axis-z", html)

    def test_orbit_embedding_has_one_3d_point_per_canonical_point(self):
        search = replace(self.search, candidates_per_code=1, modes=("orbit_embedding",))
        assets, target, _, _ = build_animation(
            "test:true-3d-count", render_cfg=self.render, search_cfg=search, frames=3, mode="orbit_embedding"
        )
        cloud, weights = build_geometry(assets["base_points"], assets["densities"], target.spec)
        self.assertEqual(len(cloud), len(assets["base_points"]))
        self.assertEqual(len(weights), len(assets["base_points"]))

    def test_orbit_embedding_front_projection_preserves_canonical_xy(self):
        search = replace(self.search, candidates_per_code=1, modes=("orbit_embedding",))
        assets, target, _, _ = build_animation(
            "test:true-3d-front", render_cfg=self.render, search_cfg=search, frames=3, mode="orbit_embedding"
        )
        front = replace(
            target.spec,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            perspective=0.0,
        )
        cloud, _ = build_geometry(assets["base_points"], assets["densities"], front)
        projected, _, _ = transform_and_project_with_depth(cloud, front)
        self.assertTrue(np.allclose(projected, assets["base_points"]))

    def test_depth_normalization_does_not_amplify_tiny_depth(self):
        points = np.array([[-0.5, 0.0, -1e-5], [0.5, 0.0, 1e-5]], dtype=np.float64)
        _, _, depth = project_points_with_depth(points, perspective=0.0, focal=3.0, camera_z=3.0)
        self.assertLess(float(depth.max() - depth.min()), 1e-3)


if __name__ == "__main__":
    unittest.main()
