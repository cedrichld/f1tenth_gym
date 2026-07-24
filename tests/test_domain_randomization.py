"""DomainRandomizationConfig: per-episode vehicle-param randomization."""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import (
    EnvConfig, SimulationConfig, DomainRandomizationConfig,
)


def _mk(dr):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            simulation_config=SimulationConfig(max_laps=None),
            domain_randomization_config=dr,
            render_enabled=False,
        ),
    )


class TestDomainRandomization(unittest.TestCase):
    def test_disabled_keeps_params_constant(self):
        env = _mk(DomainRandomizationConfig())  # disabled
        vals = []
        for s in range(4):
            env.reset(seed=s)
            vals.append(float(env.unwrapped.sim.vehicle_params.m))
        self.assertEqual(len(set(vals)), 1)
        env.close()

    def test_samples_within_range_and_varies(self):
        env = _mk(DomainRandomizationConfig(
            enabled=True, param_ranges={"m": (3.0, 4.0), "mu": (0.9, 1.1)}
        ))
        ms, mus, arrays = [], [], []
        for s in range(8):
            env.reset(seed=s)
            p = env.unwrapped.sim.vehicle_params
            ms.append(float(p.m))
            mus.append(float(p.mu))
            arrays.append(env.unwrapped.sim.params_array.copy())
        self.assertTrue(all(3.0 <= m <= 4.0 for m in ms))
        self.assertTrue(all(0.9 <= mu <= 1.1 for mu in mus))
        self.assertGreater(len(set(ms)), 1, "mass did not vary")
        # the params_array (what the njit dynamics index) must actually change
        self.assertFalse(np.array_equal(arrays[0], arrays[1]))
        env.close()

    def test_reproducible_with_reset_seed(self):
        env = _mk(DomainRandomizationConfig(enabled=True, param_ranges={"m": (3.0, 4.0)}))
        env.reset(seed=7)
        a = env.unwrapped.sim.params_array.copy()
        env.reset(seed=7)
        b = env.unwrapped.sim.params_array.copy()
        np.testing.assert_array_equal(a, b)
        env.close()

    def test_validation(self):
        with self.assertRaises(ValueError):
            DomainRandomizationConfig(param_ranges={"not_a_param": (1.0, 2.0)})
        with self.assertRaises(ValueError):
            DomainRandomizationConfig(param_ranges={"m": (4.0, 3.0)})  # low > high


if __name__ == "__main__":
    unittest.main()
