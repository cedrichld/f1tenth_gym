"""RewardConfig: pluggable per-step reward (survival / progress / custom)."""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import (
    EnvConfig,
    SimulationConfig,
    RewardConfig,
    RewardMode,
    LoopCounterMode,
)


def _mk(**reward):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            simulation_config=SimulationConfig(max_laps=None),
            reward_config=RewardConfig(**reward),
            render_enabled=False,
        ),
    )


class TestRewardConfig(unittest.TestCase):
    def test_survival_is_timestep_and_info_has_signals(self):
        env = _mk()  # default SURVIVAL
        env.reset(seed=1)
        _, r, _, _, info = env.step(np.array([[0.0, 3.0]], dtype=np.float32))
        self.assertAlmostEqual(r, env.unwrapped.timestep, places=6)
        self.assertIn("progress", info)
        self.assertIn("collisions", info)
        self.assertEqual(info["progress"].shape, (1,))
        env.close()

    def test_progress_reward_tracks_arclength(self):
        env = _mk(mode=RewardMode.PROGRESS, progress_weight=1.0)
        env.reset(seed=1)
        total = 0.0
        for _ in range(50):
            _, r, _, _, info = env.step(np.array([[0.0, 4.0]], dtype=np.float32))
            total += r
        # accelerating from rest to ~4 m/s over 0.5 s -> ~1 m of forward arclength
        self.assertGreater(total, 0.3)
        self.assertLess(total, 2.5)
        self.assertAlmostEqual(r, float(info["progress"][0]), places=6)
        env.close()

    def test_collision_penalty_applied(self):
        env = _mk(mode=RewardMode.PROGRESS, progress_weight=0.0, collision_penalty=5.0)
        env.reset(seed=1)
        for _ in range(400):
            _, r, terminated, _, info = env.step(np.array([[0.4, 7.0]], dtype=np.float32))
            if info["collisions"][0] > 0:
                self.assertAlmostEqual(r, -5.0, places=5)
                break
        else:
            self.fail("no collision produced")
        env.close()

    def test_custom_reward_fn(self):
        seen = {}

        def rf(obs, action, info, terminated, truncated):
            seen["called"] = True
            self.assertIn("progress", info)
            return 42.0

        env = _mk(mode=RewardMode.CUSTOM, reward_fn=rf)
        env.reset(seed=1)
        _, r, _, _, _ = env.step(np.array([[0.0, 2.0]], dtype=np.float32))
        self.assertEqual(r, 42.0)
        self.assertTrue(seen.get("called"))
        env.close()

    def test_validation(self):
        with self.assertRaises(ValueError):
            RewardConfig(mode=RewardMode.CUSTOM)  # no reward_fn
        with self.assertRaises(ValueError):
            RewardConfig(collision_penalty=-1.0)
        with self.assertRaises(ValueError):
            EnvConfig(
                simulation_config=SimulationConfig(
                    compute_frenet_frame=False, loop_counter=LoopCounterMode.WINDING_ANGLE
                ),
                reward_config=RewardConfig(mode=RewardMode.PROGRESS),
            )


if __name__ == "__main__":
    unittest.main()
