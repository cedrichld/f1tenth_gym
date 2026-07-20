"""Multi-agent Frenet regression.

The Frenet arclength search uses a per-agent guess. Previously all agents
shared one ``Track.s_guess`` scalar, so during a step each agent's local search
was windowed around the *previous* agent's position -- an agent far from the
others would snap to the wrong arclength branch (a >100 m jump without moving).
"""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig


def _two_agents_far_apart():
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            num_agents=2,
            simulation_config=SimulationConfig(max_laps=None),
            render_enabled=False,
        ),
    )
    cl = env.unwrapped.track.centerline
    cx = np.asarray(cl.xs, dtype=float)
    cy = np.asarray(cl.ys, dtype=float)
    yaw = np.arctan2(np.gradient(cy), np.gradient(cx))
    n = len(cx)
    a, b = 10, n // 2  # far apart in arclength
    poses = np.array(
        [[cx[a], cy[a], yaw[a]], [cx[b], cy[b], yaw[b]]], dtype=np.float32
    )
    return env, poses


def _s(obs, i):
    return float(obs[f"agent_{i}"]["frenet_pose"][0])


class TestMultiAgentFrenet(unittest.TestCase):
    def test_stationary_agents_keep_their_own_s(self):
        env, poses = _two_agents_far_apart()
        obs, _ = env.reset(options={"poses": poses})
        s0_ref, s1_ref = _s(obs, 0), _s(obs, 1)
        self.assertGreater(abs(s0_ref - s1_ref), 50.0, "test setup: agents should be far apart in s")
        for _ in range(5):
            obs, *_ = env.step(np.zeros((2, 2), dtype=np.float32))  # nobody moves
            self.assertAlmostEqual(_s(obs, 0), s0_ref, delta=0.5, msg="agent 0 s drifted while stationary")
            self.assertAlmostEqual(_s(obs, 1), s1_ref, delta=0.5, msg="agent 1 s drifted while stationary")
        env.close()

    def test_moving_agents_track_smoothly(self):
        env, poses = _two_agents_far_apart()
        obs, _ = env.reset(options={"poses": poses})
        prev = [_s(obs, 0), _s(obs, 1)]
        s_frame_max = env.unwrapped.track.centerline.spline.s_frame_max
        for _ in range(60):
            obs, *_ = env.step(np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32))
            for i in range(2):
                s = _s(obs, i)
                # small per-step change, accounting for start/finish wraparound
                ds = abs(s - prev[i])
                ds = min(ds, abs(ds - s_frame_max))
                self.assertLess(ds, 5.0, f"agent {i} s jumped {ds:.1f} m in one step")
                prev[i] = s
        env.close()


if __name__ == "__main__":
    unittest.main()
