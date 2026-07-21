"""ResetConfig now exposes the spawn-spacing / shuffle / lateral params that
were previously hardcoded in the reset functions."""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, ResetConfig, SimulationConfig


def _spacing(reset_config, seeds=range(5)):
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            num_agents=2,
            simulation_config=SimulationConfig(max_laps=None),
            reset_config=reset_config,
            render_enabled=False,
        ),
    )
    dists = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        p0 = obs["agent_0"]["std_state"][:2]
        p1 = obs["agent_1"]["std_state"][:2]
        dists.append(float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])))
    env.close()
    return float(np.mean(dists))


class TestResetConfig(unittest.TestCase):
    def test_min_max_dist_change_spawn_spacing(self):
        default = _spacing(ResetConfig())                       # 1.5..2.5 m
        wide = _spacing(ResetConfig(min_dist=5.0, max_dist=6.0))  # 5..6 m
        self.assertLess(default, 3.5)
        self.assertGreater(wide, 4.5)
        self.assertGreater(wide, default + 2.0)

    def test_shuffle_override_on_static_strategy(self):
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                num_agents=3,
                simulation_config=SimulationConfig(max_laps=None),
                reset_config=ResetConfig(shuffle=True),  # override a STATIC strategy
                render_enabled=False,
            ),
        )
        env.reset(seed=1)
        env.close()

    def test_reset_kwargs_only_non_none(self):
        self.assertEqual(ResetConfig().reset_kwargs(), {})
        self.assertEqual(
            ResetConfig(min_dist=1.0, shuffle=True).reset_kwargs(),
            {"min_dist": 1.0, "shuffle": True},
        )

    def test_validation(self):
        with self.assertRaises(ValueError):
            ResetConfig(min_dist=5.0, max_dist=3.0)
        with self.assertRaises(ValueError):
            ResetConfig(max_dist=0.0)


if __name__ == "__main__":
    unittest.main()
