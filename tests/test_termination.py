"""Termination and truncation configuration (TerminationConfig, upstream #90)."""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig, TerminationConfig


def _mk(num_agents=1, **termination):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            num_agents=num_agents,
            simulation_config=SimulationConfig(max_laps=None),
            termination_config=TerminationConfig(**termination),
            render_enabled=False,
        ),
    )


class TestTerminationConfig(unittest.TestCase):
    def test_max_episode_steps_truncates(self):
        env = _mk(max_episode_steps=10)
        env.reset(seed=1)
        for i in range(1, 15):
            _, _, terminated, truncated, _ = env.step(np.array([[0.0, 1.0]], dtype=np.float32))
            if i < 10:
                self.assertFalse(truncated)
            if truncated:
                self.assertEqual(i, 10)
                self.assertFalse(terminated, "time-limit is truncation, not termination")
                break
        else:
            self.fail("never truncated")
        env.close()

    def test_default_never_truncates(self):
        env = _mk()  # max_episode_steps None
        env.reset(seed=1)
        for _ in range(30):
            _, _, _, truncated, info = env.step(np.array([[0.0, 1.0]], dtype=np.float32))
            self.assertFalse(truncated)
        self.assertIn("collisions", info)
        self.assertEqual(info["collisions"].shape, (1,))
        env.close()

    def test_terminate_on_collision_false(self):
        env = _mk(terminate_on_collision=False)
        env.reset(seed=1)
        saw_collision = False
        for _ in range(200):
            _, _, terminated, _, info = env.step(np.array([[0.4, 7.0]], dtype=np.float32))
            saw_collision = saw_collision or bool(info["collisions"][0])
            self.assertFalse(terminated, "terminated despite terminate_on_collision=False")
        self.assertTrue(saw_collision, "test setup: expected a collision")
        env.close()

    def test_default_terminates_on_ego_collision(self):
        env = _mk()
        env.reset(seed=1)
        for _ in range(200):
            _, _, terminated, _, _ = env.step(np.array([[0.4, 7.0]], dtype=np.float32))
            if terminated:
                break
        else:
            self.fail("default config did not terminate on collision")
        env.close()

    def test_collision_agents_policy(self):
        # "ego" ignores an opponent-only collision; "any" terminates on it.
        for policy, expected in (("ego", False), ("any", True)):
            env = _mk(num_agents=2, collision_agents=policy)
            env.reset(seed=1)
            env.unwrapped.sim.state.collisions[:] = 0.0
            env.unwrapped.sim.state.collisions[1] = 1.0  # opponent only (ego is idx 0)
            self.assertEqual(env.unwrapped._check_done(), expected, f"policy={policy}")
            env.close()

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            TerminationConfig(max_episode_steps=0)
        with self.assertRaises(ValueError):
            TerminationConfig(collision_agents="nobody")


if __name__ == "__main__":
    unittest.main()
