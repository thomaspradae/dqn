import random
import unittest

import numpy as np

from artmarks.families import get_families


class FamilyTest(unittest.TestCase):
    def test_each_family_returns_finite_points(self):
        for family in get_families():
            with self.subTest(family=family.name):
                params = family.sample_params(random.Random(1234))
                pts = family.generate_points(params, max(1200, int(family.default_iterations * 0.05)))
                self.assertEqual(pts.ndim, 2)
                self.assertEqual(pts.shape[1], 2)
                self.assertGreater(len(pts), 50)
                self.assertTrue(np.isfinite(pts).all())


if __name__ == "__main__":
    unittest.main()
