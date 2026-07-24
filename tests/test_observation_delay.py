"""ObservationDelayWrapper: fixed-step sensor/perception latency."""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig
from f1tenth_gym.envs.wrappers import ObservationDelayWrapper


def _mk():
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            simulation_config=SimulationConfig(max_laps=None), render_enabled=False
        ),
    )


def _x(obs):
    return float(obs["agent_0"]["std_state"][0])


ACTION = np.array([[0.1, 4.0]], dtype=np.float32)


def _base_sequence(steps, seed=0):
    """Undelayed obs: index 0 is the reset obs, index t is obs after step t."""
    env = _mk()
    obs, _ = env.reset(seed=seed)
    seq = [_x(obs)]
    for _ in range(steps):
        obs, *_ = env.step(ACTION)
        seq.append(_x(obs))
    env.close()
    return seq


class TestObservationDelay(unittest.TestCase):
    def test_zero_delay_passthrough(self):
        base = _base_sequence(10)
        env = ObservationDelayWrapper(_mk(), delay_steps=0)
        obs, _ = env.reset(seed=0)
        got = [_x(obs)]
        for _ in range(10):
            obs, *_ = env.step(ACTION)
            got.append(_x(obs))
        env.close()
        self.assertEqual(base, got)

    def test_delay_returns_stale_observation(self):
        k = 3
        steps = 12
        base = _base_sequence(steps)
        env = ObservationDelayWrapper(_mk(), delay_steps=k)
        obs, _ = env.reset(seed=0)
        self.assertEqual(_x(obs), base[0])  # reset returns reset obs
        for t in range(1, steps + 1):
            obs, *_ = env.step(ACTION)
            expected = base[max(0, t - k)]  # padded with reset obs until t>k
            self.assertEqual(_x(obs), expected, f"step {t}")
        env.close()

    def test_reward_and_termination_are_current(self):
        # delayed obs, but reward/terminated must track the true (undelayed) run
        base_env = _mk()
        base_env.reset(seed=1)
        base_r, base_term = [], []
        for _ in range(20):
            _, r, term, trunc, _ = base_env.step(ACTION)
            base_r.append(r)
            base_term.append(term)
        base_env.close()

        env = ObservationDelayWrapper(_mk(), delay_steps=4)
        env.reset(seed=1)
        for i in range(20):
            _, r, term, trunc, _ = env.step(ACTION)
            self.assertEqual(r, base_r[i])
            self.assertEqual(term, base_term[i])
        env.close()

    def test_no_aliasing(self):
        # mutating a returned observation must not corrupt buffered frames
        env = ObservationDelayWrapper(_mk(), delay_steps=2)
        env.reset(seed=0)
        obs, *_ = env.step(ACTION)
        obs["agent_0"]["std_state"][0] = 999.0
        obs2, *_ = env.step(ACTION)
        self.assertNotEqual(_x(obs2), 999.0)
        env.close()

    def test_validation(self):
        with self.assertRaises(ValueError):
            ObservationDelayWrapper(_mk(), delay_steps=-1)


if __name__ == "__main__":
    unittest.main()
