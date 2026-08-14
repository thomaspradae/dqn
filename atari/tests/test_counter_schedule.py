import unittest
from types import SimpleNamespace

from train_nature import (
    counter_smoke_summary,
    epsilon_for_agent_step,
    should_learn,
    should_update_target,
)


class CounterScheduleTests(unittest.TestCase):
    def test_epsilon_decay_uses_agent_steps_after_learning_start(self):
        args = SimpleNamespace(
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=1_000_000,
            learning_starts=50_000,
        )
        self.assertEqual(epsilon_for_agent_step(args, 0), 1.0)
        self.assertEqual(epsilon_for_agent_step(args, 50_000), 1.0)
        self.assertAlmostEqual(epsilon_for_agent_step(args, 550_000), 0.55)
        self.assertAlmostEqual(epsilon_for_agent_step(args, 1_050_000), 0.1)

    def test_learning_and_target_schedules_use_decision_steps(self):
        self.assertFalse(should_learn(50_000, 50_000, 4))
        self.assertTrue(should_learn(50_004, 50_000, 4))
        self.assertTrue(should_update_target(1, 10_000))
        self.assertFalse(should_update_target(10_000, 10_000))
        self.assertTrue(should_update_target(10_001, 10_000))

    def test_1000_decision_counter_smoke_totals(self):
        summary = counter_smoke_summary(
            agent_steps=1_000,
            frame_skip=4,
            learning_starts=0,
            train_freq=4,
            target_update_freq=100,
        )
        self.assertEqual(
            summary,
            {
                "agent_decisions": 1_000,
                "nominal_raw_action_frames": 4_000,
                "optimizer_updates": 249,
                "target_copies": 10,
            },
        )

    def test_trace_exposes_pre_and_post_increment_conventions(self):
        summary = counter_smoke_summary(20, 4, 0, 4, 10, trace=True)
        self.assertEqual(summary["update_pre_increment_steps"], [4, 8, 12, 16])
        self.assertEqual(summary["target_post_increment_steps"], [1, 11])


if __name__ == "__main__":
    unittest.main()
