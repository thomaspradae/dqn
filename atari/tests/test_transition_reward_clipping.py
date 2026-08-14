import unittest

from train_nature import clip_transition_reward


class TransitionRewardClippingTests(unittest.TestCase):
    def test_action_repeat_transition_reward_is_bounded(self):
        for total_raw_reward in range(-16, 17):
            with self.subTest(total_raw_reward=total_raw_reward):
                reward = clip_transition_reward(total_raw_reward)
                self.assertGreaterEqual(reward, -1.0)
                self.assertLessEqual(reward, 1.0)

    def test_action_repeat_transition_reward_is_clipped_after_aggregation(self):
        self.assertEqual(clip_transition_reward(3.0), 1.0)
        self.assertEqual(clip_transition_reward(-3.0), -1.0)
        self.assertEqual(clip_transition_reward(0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
