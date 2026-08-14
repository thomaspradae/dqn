from __future__ import annotations

import unittest

import numpy as np

from dejong3d.scoring import binary_metrics
from native3d.integrate import cloud_metrics, generate_orbit, normalize_cloud, object_quality, rk4_orbit
from native3d.models import CameraPose
from native3d.export import _PAGE
from native3d.search import project_cloud
from native3d.systems import FAMILIES
from generate_transient_lab import PAGE as TRANSIENT_PAGE, catalog as transient_catalog


class Native3DTests(unittest.TestCase):
    def test_gallery_exposes_interactive_camera_and_snapshots(self):
        self.assertIn("pointermove", _PAGE)
        self.assertIn("terminalProjection", _PAGE)
        self.assertIn("Save view", _PAGE)
        self.assertIn("Download JSON", _PAGE)
        self.assertIn("Shift+drag", _PAGE)

    def test_transient_lab_covers_each_evolution_protocol(self):
        systems = transient_catalog()
        kinds = {item["kind"] for item in systems}
        self.assertTrue({"map2d", "map3d", "ode3d", "curve3d"}.issubset(kinds))
        self.assertIn("De Jong 2D - poormans #009", {item["label"] for item in systems})
        self.assertIn("ensemble transient", {item["protocol"] for item in systems})
        self.assertIn("parametric construction", {item["protocol"] for item in systems})
        self.assertIn("Record WebM", TRANSIENT_PAGE)
        self.assertIn("exposure", TRANSIENT_PAGE)

    def test_binary_perimeter_does_not_wrap_unsigned_values(self):
        term = np.zeros((12, 18), dtype=np.float64)
        term[3:9, 5:13] = 1.0
        metrics = binary_metrics(term, 0.2)
        self.assertGreater(metrics["perimeter_ratio"], 0.0)
        self.assertLess(metrics["perimeter_ratio"], 4.0)

    def test_aizawa_default_is_genuinely_volumetric(self):
        family = FAMILIES["aizawa"]
        points = normalize_cloud(rk4_orbit(family, family.defaults, family.initial, samples=3000))
        metrics = cloud_metrics(points)
        _, accepted = object_quality(metrics)
        self.assertTrue(accepted)
        self.assertGreater(metrics["lambda3_ratio"], 0.10)
        self.assertGreater(metrics["local_lambda3_ratio"], 0.010)

    def test_curled_sheet_fails_local_volume_filter(self):
        t = np.linspace(0.0, 12.0 * np.pi, 5000)
        sheet = np.column_stack((np.cos(t), np.sin(t), 0.04 * t + 0.02 * np.sin(7.0 * t)))
        metrics = cloud_metrics(normalize_cloud(sheet))
        _, accepted = object_quality(metrics)
        self.assertFalse(accepted)
        self.assertLess(metrics["local_lambda3_ratio"], 0.010)

    def test_camera_changes_projection_not_object(self):
        family = FAMILIES["halvorsen"]
        points = normalize_cloud(rk4_orbit(family, family.defaults, family.initial, samples=2000))
        original = np.array(points, copy=True)
        front, _ = project_cloud(points, CameraPose(0.0, 0.0, 0.0))
        turned, _ = project_cloud(points, CameraPose(24.0, 62.0, -8.0))
        self.assertTrue(np.array_equal(points, original))
        self.assertFalse(np.allclose(front, turned))

    def test_cyclic_dejong_is_native_volume(self):
        family = FAMILIES["dejong3d_cyclic"]
        points = normalize_cloud(generate_orbit(family, family.defaults, family.initial, samples=4000))
        metrics = cloud_metrics(points)
        _, accepted = object_quality(metrics, family.topology)
        self.assertTrue(accepted)
        self.assertGreater(metrics["local_lambda3_ratio"], 0.10)

    def test_lissajous_is_accepted_as_nonplanar_curve_not_volume(self):
        family = FAMILIES["lissajous3d"]
        points = normalize_cloud(generate_orbit(family, family.defaults, family.initial, samples=4000))
        metrics = cloud_metrics(points)
        _, curve_accepted = object_quality(metrics, "curve")
        _, volume_accepted = object_quality(metrics, "volume")
        self.assertTrue(curve_accepted)
        self.assertFalse(volume_accepted)
        self.assertGreater(metrics["lambda3_ratio"], 0.10)
        self.assertLess(metrics["box_dimension"], 1.65)


if __name__ == "__main__":
    unittest.main()
