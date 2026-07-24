"""Wrappers: single-agent flat interface for RL libraries."""
import unittest

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
from gymnasium.utils.env_checker import check_env

from f1tenth_gym.envs.env_config import (
    EnvConfig, ObservationConfig, SimulationConfig, TerminationConfig,
)
from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.wrappers import SingleAgentWrapper


def _base(**kw):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
            simulation_config=SimulationConfig(max_laps=None),
            termination_config=TerminationConfig(max_episode_steps=50),
            render_enabled=False,
            **kw,
        ),
    )


class TestSingleAgentWrapper(unittest.TestCase):
    def test_unwraps_obs_and_action(self):
        env = SingleAgentWrapper(_base())
        self.assertEqual(env.action_space.shape, (2,))
        obs, _ = env.reset(seed=1)
        self.assertIsInstance(obs, dict)
        self.assertNotIn("agent_0", obs)  # already unwrapped
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        self.assertEqual(np.ndim(r), 0)
        env.close()

    def test_flatten_gives_finite_box(self):
        env = FlattenObservation(SingleAgentWrapper(_base()))
        self.assertIsInstance(env.observation_space, gym.spaces.Box)
        self.assertTrue(np.all(np.isfinite(env.observation_space.low)))
        self.assertTrue(np.all(np.isfinite(env.observation_space.high)))
        env.close()

    def test_passes_gymnasium_check_env(self):
        env = FlattenObservation(SingleAgentWrapper(_base()))
        check_env(env, skip_render_check=True)
        env.close()

    def test_requires_single_agent(self):
        with self.assertRaises(ValueError):
            SingleAgentWrapper(gym.make("f1tenth_gym:f1tenth-v0",
                                        config=EnvConfig(num_agents=2, render_enabled=False)))


if __name__ == "__main__":
    unittest.main()
