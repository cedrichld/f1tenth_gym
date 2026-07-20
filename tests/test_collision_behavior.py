"""Env-level collision behaviour regressions.

Covers two things:
* a collision must halt the car (zero velocity) but PRESERVE its yaw -- the old
  code did ``state[3:] = 0`` which also wiped yaw (index 4), snapping the
  heading to 0/east.
* a car centred in the track must never falsely collide (f1tenth/f1tenth_gym
  issue #91 "unexpected collision ... without any visible collision").
"""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig


def _env(map_name="Spielberg"):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            map_name=map_name,
            simulation_config=SimulationConfig(max_laps=None),
            render_enabled=False,
        ),
    )


class TestCollisionBehaviour(unittest.TestCase):
    def test_collision_preserves_yaw_and_halts(self):
        env = _env()
        env.reset(seed=1)
        prev_yaw = None
        for _ in range(400):
            obs, _, done, _, _ = env.step(np.array([[0.4, 7.0]], dtype=np.float32))
            s = obs["agent_0"]["std_state"]
            yaw, speed = float(s[4]), float(s[3])
            if obs["agent_0"]["collision"] > 0:
                # velocity halted...
                self.assertAlmostEqual(speed, 0.0, places=5, msg="collision did not halt the car")
                # ...but yaw preserved (not snapped to 0, continuous with prev step)
                self.assertNotEqual(yaw, 0.0, "collision zeroed the yaw")
                if prev_yaw is not None:
                    self.assertLess(
                        abs(yaw - prev_yaw), 0.3,
                        f"yaw jumped on collision ({prev_yaw:.3f} -> {yaw:.3f}); likely reset to 0",
                    )
                # state and standard_state must agree on the halt
                self.assertAlmostEqual(float(obs["agent_0"]["state"][3]), 0.0, places=5)
                break
            prev_yaw = yaw
        else:
            self.fail("no collision produced in 400 steps of hard cornering")
        env.close()

    def test_no_false_collision_at_centerline(self):
        """A stationary car centred on the track centreline must not collide (#91)."""
        env = _env()
        cl = env.unwrapped.track.centerline
        cx = np.asarray(cl.xs, dtype=float)
        cy = np.asarray(cl.ys, dtype=float)
        yaw = np.arctan2(np.gradient(cy), np.gradient(cx))
        n = len(cx)
        stride = max(1, n // 100)
        false_collisions = []
        for k in range(0, n, stride):
            env.reset(options={"poses": np.array([[cx[k], cy[k], yaw[k]]], dtype=np.float32)})
            obs, *_ = env.step(np.array([[0.0, 0.0]], dtype=np.float32))
            if obs["agent_0"]["collision"] > 0:
                false_collisions.append((k, float(obs["agent_0"]["scan"].min())))
        self.assertEqual(
            false_collisions, [],
            f"false collisions at centreline points (idx, min_scan): {false_collisions}",
        )
        env.close()


if __name__ == "__main__":
    unittest.main()


class TestCollisionModes(unittest.TestCase):
    def test_none_mode_disables_collisions(self):
        """CollisionCheckMode.NONE: driving into a wall never flags a collision (#124)."""
        from f1tenth_gym.envs.collision_models import CollisionCheckMode
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                simulation_config=SimulationConfig(max_laps=None),
                collision_check=CollisionCheckMode.NONE,
                render_enabled=False,
            ),
        )
        env.reset(seed=1)
        for _ in range(400):
            obs, _, done, _, _ = env.step(np.array([[0.4, 7.0]], dtype=np.float32))
            self.assertEqual(float(obs["agent_0"]["collision"]), 0.0, "NONE mode flagged a collision")
            self.assertFalse(done, "NONE mode terminated on collision")
        env.close()

    def test_lidar_disabled_does_not_crash(self):
        """lidar_config.enabled=False must not crash step() in any collision mode (#124)."""
        from f1tenth_gym.envs.collision_models import CollisionCheckMode
        from f1tenth_gym.envs.lidar import LiDARConfig
        for mode in (CollisionCheckMode.LIDAR_SCAN, CollisionCheckMode.BOUNDING_BOX, CollisionCheckMode.NONE):
            env = gym.make(
                "f1tenth_gym:f1tenth-v0",
                config=EnvConfig(
                    simulation_config=SimulationConfig(max_laps=None),
                    lidar_config=LiDARConfig(enabled=False),
                    collision_check=mode,
                    num_agents=2,
                    render_enabled=False,
                ),
            )
            env.reset(seed=1)
            for _ in range(20):
                env.step(np.zeros((2, 2), dtype=np.float32))
            env.close()
