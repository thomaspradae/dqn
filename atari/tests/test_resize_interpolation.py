import unittest

import cv2
import numpy as np

from eval import preprocess_frame as eval_preprocess_frame
from train_nature import preprocess_frame as train_preprocess_frame


class ResizeInterpolationTests(unittest.TestCase):
    def test_bilinear_matches_opencv_reference_in_training_and_eval(self):
        obs = np.zeros((210, 160, 3), dtype=np.uint8)
        obs[:, :, 0] = np.arange(160, dtype=np.uint8)
        obs[:, :, 1] = np.arange(210, dtype=np.uint8)[:, None]
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2YUV)[:, :, 0]
        expected = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_LINEAR)

        np.testing.assert_array_equal(
            train_preprocess_frame(obs, resize_interpolation="bilinear"), expected
        )
        np.testing.assert_array_equal(
            eval_preprocess_frame(obs, resize_interpolation="bilinear"), expected
        )


if __name__ == "__main__":
    unittest.main()
