"""FRENET_BASED lap-counter regressions.

Lap progress must be measured from the spawn arclength, not from the spline's
s=0 datum. Previously ``agents_prev_s`` was zero-filled at reset, so on the
first step a car spawned at s>0 counted its whole spawn arclength as progress
(``cumulative_s`` jumped to ~spawn s) and completed lap 1 early.
"""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig, LoopCounterMode


class TestFrenetLapCounter(unittest.TestCase):
    def test_lap_progress_measured_from_spawn(self):
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                simulation_config=SimulationConfig(
                    max_laps=None, loop_counter=LoopCounterMode.FRENET_BASED
                ),
                render_enabled=False,
            ),
        )
        cl = env.unwrapped.track.centerline
        cx = np.asarray(cl.xs, dtype=float)
        cy = np.asarray(cl.ys, dtype=float)
        yaw = np.arctan2(np.gradient(cy), np.gradient(cx))
        k = len(cx) // 2  # spawn roughly half a lap along the track (s well above 0)
        obs, _ = env.reset(options={"poses": np.array([[cx[k], cy[k], yaw[k]]], dtype=np.float32)})
        spawn_s = float(obs["agent_0"]["frenet_pose"][0])
        self.assertGreater(spawn_s, 20.0, "test setup: spawn should be well past s=0")

        # agents_prev_s must be seeded to the spawn s...
        self.assertAlmostEqual(float(env.unwrapped.agents_prev_s[0]), spawn_s, delta=1.0)

        # ...so a tiny forward move accrues ~no lap progress (not the spawn s).
        env.step(np.array([[0.0, 1.0]], dtype=np.float32))
        self.assertLess(
            abs(float(env.unwrapped.cumulative_s[0])), 1.0,
            "cumulative_s jumped by ~spawn_s: lap progress not measured from spawn",
        )
        self.assertEqual(int(env.unwrapped.lap_counts[0]), 0)
        env.close()


if __name__ == "__main__":
    unittest.main()
