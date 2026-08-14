import unittest

import numpy as np

from artmarks.families import get_families
from artmarks.models import RenderConfig, SearchConfig
from artmarks.render import terminal_to_text
from artmarks.search import search_marks


class SmokeTest(unittest.TestCase):
    def test_all_families_generate_and_search(self):
        render = RenderConfig(term_width=30, term_height=15, supersample=2, blur_passes=0)
        search = SearchConfig(seed="smoke", per_family=2, count=4, iteration_scale=0.12, seed_each_family=False)
        selected, pool = search_marks(get_families(), render, search)
        self.assertGreater(len(pool), 0)
        self.assertGreater(len(selected), 0)
        self.assertEqual(selected[0].term.shape, (15, 30))
        self.assertTrue(np.isfinite(selected[0].term).all())
        self.assertEqual(len(terminal_to_text(selected[0].term)), 15)


if __name__ == "__main__":
    unittest.main()
