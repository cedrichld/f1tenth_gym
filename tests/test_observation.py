import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig
from f1tenth_gym.envs.observation import (
    observation_factory,
    ObservationType,
)


class TestObservationInterface(unittest.TestCase):
    def test_direct_default_fields(self):
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                observation_config=ObservationConfig(type=ObservationType.DIRECT)
            ),
        )

        obs, _ = env.reset()
        direct_obs = observation_factory(env, type=ObservationType.DIRECT)
        space = direct_obs.space()
        sample = direct_obs.observe()

        expected_fields = {
            "scan",
            "std_state",
            "state",
            "collision",
            "lap_time",
            "lap_count",
            "sim_time",
        }
        if env.unwrapped.compute_frenet:
            expected_fields.add("frenet_pose")

        for agent_id in env.unwrapped.agent_ids:
            agent_space = space.spaces[agent_id]
            self.assertTrue(isinstance(agent_space, gym.spaces.Dict))
            self.assertEqual(set(agent_space.spaces.keys()), expected_fields)
            self.assertEqual(set(sample[agent_id].keys()), expected_fields)
            self.assertTrue(agent_space.contains(sample[agent_id]))
            self.assertIn(agent_id, obs)
        env.close()

    def test_feature_subset(self):
        features = ("pose_x", "pose_y", "pose_theta")
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                observation_config=ObservationConfig(
                    type=ObservationType.FEATURES,
                    features=features,
                )
            ),
        )

        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())

        for agent_id in env.unwrapped.agent_ids:
            self.assertEqual(set(obs[agent_id].keys()), set(features))
            self.assertTrue(all(isinstance(value, np.ndarray) for value in obs[agent_id].values()))

        sim_state = env.unwrapped.sim.state
        for idx, agent_id in enumerate(env.unwrapped.agent_ids):
            pose_x, pose_y, pose_theta = sim_state.poses[idx]
            agent_obs = obs[agent_id]
            observed = np.asarray([
                agent_obs["pose_x"],
                agent_obs["pose_y"],
                agent_obs["pose_theta"],
            ], dtype=np.float32)
            expected = np.asarray([pose_x, pose_y, pose_theta], dtype=np.float32)
            self.assertTrue(np.allclose(observed, expected))
        env.close()

    def test_presets_match_expected_fields(self):
        preset_expectations = {
            ObservationType.KINEMATIC_STATE: {"pose_x", "pose_y", "delta", "linear_vel_x", "pose_theta"},
            ObservationType.DYNAMIC_STATE: {
                "pose_x",
                "pose_y",
                "delta",
                "linear_vel_magnitude",
                "pose_theta",
                "ang_vel_z",
                "beta",
            },
            ObservationType.FRENET_DYNAMIC_STATE: {
                "pose_x",
                "pose_y",
                "delta",
                "linear_vel_x",
                "linear_vel_y",
                "pose_theta",
                "ang_vel_z",
                "beta",
            },
        }

        for obs_type, expected in preset_expectations.items():
            env = gym.make(
                "f1tenth_gym:f1tenth-v0",
                config=EnvConfig(
                    observation_config=ObservationConfig(type=obs_type)
                ),
            )
            obs_impl = observation_factory(env, type=obs_type)
            space = obs_impl.space()
            observation = obs_impl.observe()

            for agent_id in env.unwrapped.agent_ids:
                self.assertEqual(set(space.spaces[agent_id].spaces.keys()), expected)
                self.assertEqual(set(observation[agent_id].keys()), expected)
                self.assertTrue(space.spaces[agent_id].contains(observation[agent_id]))
            env.close()

    def test_invalid_feature_name(self):
        env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig())
        with self.assertRaises(ValueError):
            observation_factory(
                env,
                type=ObservationType.FEATURES,
                features=("unknown_feature",),
            )
        env.close()

    def test_invalid_observation_type(self):
        env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig())
        with self.assertRaises(TypeError):
            observation_factory(env, type="unsupported")
        env.close()

    def test_space_contains_observation(self):
        obs_types = [
            ObservationType.DIRECT,
            ObservationType.KINEMATIC_STATE,
            ObservationType.DYNAMIC_STATE,
        ]
        env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig())
        env.reset()

        for obs_type in obs_types:
            obs_impl = observation_factory(env, type=obs_type)
            space = obs_impl.space()
            observation = obs_impl.observe()
            self.assertTrue(space.contains(observation))
        env.close()



class TestScanObservationCopy(unittest.TestCase):
    def test_scan_is_a_copy_not_a_live_view(self):
        """obs['scan'] must be a copy: the sim scan buffer is overwritten in
        place every step, so a view would silently corrupt a stored scan
        (e.g. an RL replay buffer)."""
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(observation_config=ObservationConfig(type=ObservationType.DIRECT)),
        )
        obs, _ = env.reset(seed=1)
        env.step(np.array([[0.1, 4.0]], dtype=np.float32))
        obs, *_ = env.step(np.array([[0.1, 4.0]], dtype=np.float32))
        scan = obs["agent_0"]["scan"]
        self.assertFalse(
            np.shares_memory(scan, env.unwrapped.sim.state.scans),
            "obs scan aliases the live sim buffer",
        )
        snapshot = scan.copy()
        env.step(np.array([[0.2, 4.0]], dtype=np.float32))  # overwrites the buffer
        np.testing.assert_array_equal(scan, snapshot, "stored scan mutated after a later step")
        env.close()


class TestFiniteObsBounds(unittest.TestCase):
    def test_bounds_finite_and_not_violated(self):
        """Observation spaces must be finite (not +/-1e30) and contain the
        actual observations under aggressive driving, so gymnasium
        normalization / check_env work."""
        import numpy as _np
        for ot in (ObservationType.DIRECT, ObservationType.KINEMATIC_STATE,
                   ObservationType.DYNAMIC_STATE, ObservationType.FRENET_DYNAMIC_STATE):
            from f1tenth_gym.envs.env_config import SimulationConfig
            env = gym.make(
                "f1tenth_gym:f1tenth-v0",
                config=EnvConfig(
                    observation_config=ObservationConfig(type=ot),
                    simulation_config=SimulationConfig(max_laps=None),
                    render_enabled=False,
                ),
            )
            sp = env.observation_space["agent_0"]
            for box in sp.spaces.values():
                self.assertTrue(_np.all(_np.isfinite(box.low)), f"{ot.name}: non-finite low")
                self.assertTrue(_np.all(_np.isfinite(box.high)), f"{ot.name}: non-finite high")
            obs, _ = env.reset(seed=2)
            rng = _np.random.default_rng(0)
            for _ in range(200):
                for k, v in obs["agent_0"].items():
                    self.assertTrue(sp[k].contains(_np.asarray(v, _np.float32)),
                                    f"{ot.name}: {k}={_np.asarray(v)} outside {sp[k]}")
                a = _np.array([[rng.uniform(-0.4, 0.4), rng.uniform(0, 15)]], _np.float32)
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    obs, _ = env.reset()
            env.close()

    def _rollout_in_space(self, obstype, action_fn, steps=300, seed=3):
        import numpy as _np
        from f1tenth_gym.envs.env_config import SimulationConfig
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                observation_config=ObservationConfig(type=obstype),
                simulation_config=SimulationConfig(max_laps=None),
                render_enabled=False,
            ),
        )
        sp = env.observation_space["agent_0"]
        obs, _ = env.reset(seed=seed)
        mags = []
        for t in range(steps):
            obs, _, term, trunc, _ = env.step(action_fn(t))
            if "linear_vel_magnitude" in obs["agent_0"]:
                mags.append(float(obs["agent_0"]["linear_vel_magnitude"]))
            for k, v in obs["agent_0"].items():
                self.assertTrue(sp[k].contains(_np.asarray(v, _np.float32)),
                                f"{obstype.name}: {k}={_np.asarray(v)} outside {sp[k]}")
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        return mags

    def test_reverse_keeps_magnitude_nonnegative_and_in_space(self):
        # linear_vel_magnitude must be a true (non-negative) magnitude, not the
        # signed speed -- else reversing puts it below its declared low of 0.
        import numpy as _np
        mags = self._rollout_in_space(
            ObservationType.DYNAMIC_STATE,
            lambda t: _np.array([[0.0, -5.0]], _np.float32),  # full reverse
        )
        self.assertTrue(all(m >= 0.0 for m in mags), "magnitude went negative")
        self.assertGreater(max(mags), 1.0, "car never actually reversed")

    def test_hard_steer_and_accel_stay_in_space(self):
        # full-lock steering + max accel: the actuator constraints zero the rate
        # only AFTER crossing the limit, so delta/speed overshoot by one step.
        # The bounds must include that margin.
        import numpy as _np
        for ot in (ObservationType.DIRECT, ObservationType.DYNAMIC_STATE,
                   ObservationType.KINEMATIC_STATE):
            self._rollout_in_space(
                ot,
                lambda t: _np.array([[0.4189 if t % 2 else -0.4189, 20.0]], _np.float32),
            )
