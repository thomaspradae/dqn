import unittest
from unittest.mock import patch

from eval_checkpoint import resize_interpolation_for_run


class EvalPreprocessingMetadataTests(unittest.TestCase):
    def test_bilinear_metadata_is_used_by_evaluation(self):
        run = {"run_dir": "/tmp/v4"}
        with patch(
            "eval_checkpoint.read_remote_file",
            return_value='{"resize_interpolation": "bilinear"}',
        ):
            self.assertEqual(resize_interpolation_for_run(None, run), "bilinear")

    def test_missing_or_invalid_metadata_preserves_legacy_area_default(self):
        run = {"run_dir": "/tmp/v3"}
        with patch("eval_checkpoint.read_remote_file", return_value="not-json"):
            self.assertEqual(resize_interpolation_for_run(None, run), "area")
        with patch(
            "eval_checkpoint.read_remote_file",
            return_value='{"resize_interpolation": "nearest"}',
        ):
            self.assertEqual(resize_interpolation_for_run(None, run), "area")


if __name__ == "__main__":
    unittest.main()
