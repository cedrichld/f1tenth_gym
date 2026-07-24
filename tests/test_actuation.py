"""Actuation realism: command noise + longitudinal (throttle) delay.

These affect the *applied* command only; defaults (all 0) must leave behavior
byte-identical to the plain simulator.
"""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig, ControlConfig


def _traj(seed=1, steps=60, action=(0.05, 5.0), **cc):
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            simulation_config=SimulationConfig(max_laps=None),
            render_enabled=False,
            control_config=ControlConfig(**cc),
        ),
    )
    env.reset(seed=seed)
    a = np.array([list(action)], dtype=np.float32)
    speeds = []
    for _ in range(steps):
        obs, *_ = env.step(a)
        speeds.append(float(obs["agent_0"]["std_state"][3]))
    env.close()
    return np.array(speeds)


class TestActuation(unittest.TestCase):
    def test_defaults_byte_identical(self):
        # two default runs match exactly, and defaults change nothing
        np.testing.assert_array_equal(_traj(), _traj())

    def test_steer_noise_changes_and_reproducible(self):
        base = _traj()
        noisy = _traj(steer_noise_std=0.15)
        self.assertFalse(np.allclose(base, noisy))
        np.testing.assert_array_equal(noisy, _traj(steer_noise_std=0.15))

    def test_accl_noise_changes_and_reproducible(self):
        base = _traj()
        noisy = _traj(accl_noise_std=1.5)
        self.assertFalse(np.allclose(base, noisy))
        np.testing.assert_array_equal(noisy, _traj(accl_noise_std=1.5))

    def test_throttle_delay_changes_trajectory(self):
        base = _traj()
        delayed = _traj(throttle_delay_steps=8)
        self.assertFalse(np.allclose(base, delayed))
        # reproducible
        np.testing.assert_array_equal(delayed, _traj(throttle_delay_steps=8))

    def test_throttle_delay_lags_response(self):
        # with a large accel command from rest, an 8-step delay must keep the
        # vehicle slower over the first few steps than the no-delay case.
        base = _traj(action=(0.0, 8.0), steps=12)
        delayed = _traj(action=(0.0, 8.0), steps=12, throttle_delay_steps=8)
        self.assertLess(delayed[3], base[3])

    def test_validation(self):
        with self.assertRaises(ValueError):
            ControlConfig(throttle_delay_steps=-1)
        with self.assertRaises(ValueError):
            ControlConfig(steer_noise_std=-0.1)
        with self.assertRaises(ValueError):
            ControlConfig(accl_noise_std=-0.1)


if __name__ == "__main__":
    unittest.main()
