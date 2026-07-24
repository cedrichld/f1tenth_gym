from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base import Observation, scan_space

__all__ = ["FullObservation"]

_BASE_FIELDS: tuple[str, ...] = (
    "scan",
    "std_state",
    "state",
    "collision",
    "lap_time",
    "lap_count",
    "sim_time",
    "frenet_pose",
)

_DERIVED_FIELDS: tuple[str, ...] = (
    "pose_x",
    "pose_y",
    "pose_theta",
    "linear_vel_x",
    "linear_vel_y",
    "linear_vel_magnitude",
    "ang_vel_z",
    "delta",
    "beta",
)

_ALL_FIELDS = set(_BASE_FIELDS) | set(_DERIVED_FIELDS)


def _scalar_box(low: float, high: float) -> gym.Space:
    return gym.spaces.Box(low=low, high=high, shape=(), dtype=np.float32)


_PI = float(np.pi)
# Finite fallbacks for quantities without a tight physical cap. Chosen generous
# enough not to be violated in practice, but finite so gymnasium normalization
# and check_env work (they can't with the old +/-1e30).
_TIME_LIMIT = 1.0e6  # seconds / lap count
_EY_LIMIT = 20.0     # metres of lateral Frenet deviation
_BETA_LIMIT = _PI    # slip angle (physically < pi/2; pi is a safe cap)


class FullObservation(Observation):
    """Observation provider that exposes simulator state with optional feature filtering."""

    def __init__(self, env, fields: tuple[str, ...] | None = None):
        super().__init__(env)
        if fields is None:
            self._fields: tuple[str, ...] | None = None
        else:
            normalized = tuple(fields)
            if not normalized:
                raise ValueError("FullObservation requires at least one field")
            unknown = next((item for item in normalized if item not in _ALL_FIELDS), None)
            if unknown is not None:
                raise ValueError(f"Unknown observation field: {unknown!r}")
            self._fields = normalized

    def _selected_fields(self) -> tuple[str, ...]:
        if self._fields is None:
            if self.env.unwrapped.compute_frenet:
                return _BASE_FIELDS
            return tuple(field for field in _BASE_FIELDS if field != "frenet_pose")
        if "frenet_pose" in self._fields and not self.env.unwrapped.compute_frenet:
            raise ValueError("frenet_pose requested but environment does not compute the Frenet frame")
        return self._fields

    def _physical_bounds(self) -> dict:
        """Finite, roughly-physical bounds for the observation fields, derived
        from the vehicle limits and track extents."""
        env = self.env.unwrapped
        p = env.vehicle_params
        v_min, v_max = float(p.v_min), float(p.v_max)
        s_min, s_max = float(p.s_min), float(p.s_max)
        spd_hi = max(abs(v_min), abs(v_max))
        wheelbase = max(float(p.lf) + float(p.lr), 0.1)
        # kinematic max yaw rate = v*tan(steer)/wheelbase, with a safety factor
        steer_hi = max(abs(s_min), abs(s_max))
        yaw_rate = 1.5 * spd_hi * float(np.tan(steer_hi)) / wheelbase

        track = env.track
        cl = getattr(track, "centerline", None) if track is not None else None
        if cl is not None:
            xs = np.asarray(cl.xs, dtype=float)
            ys = np.asarray(cl.ys, dtype=float)
            m = 5.0  # metres of margin beyond the centerline extent
            x_lo, x_hi = float(xs.min()) - m, float(xs.max()) + m
            y_lo, y_hi = float(ys.min()) - m, float(ys.max()) + m
            s_arc = float(cl.spline.s_frame_max)
        else:
            x_lo = y_lo = -1.0e4
            x_hi = y_hi = 1.0e4
            s_arc = _TIME_LIMIT
        return dict(
            v_min=v_min, v_max=v_max, spd_hi=spd_hi, s_min=s_min, s_max=s_max,
            yaw_rate=yaw_rate, x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi, s_arc=s_arc,
        )

    def _field_space(self, field: str, sim, b: dict) -> gym.Space:
        f32 = np.float32
        Box = gym.spaces.Box
        if field == "scan":
            return scan_space(sim)
        if field == "collision":
            return _scalar_box(0.0, 1.0)
        if field in ("lap_time", "lap_count", "sim_time"):
            return _scalar_box(0.0, _TIME_LIMIT)
        if field == "pose_x":
            return _scalar_box(b["x_lo"], b["x_hi"])
        if field == "pose_y":
            return _scalar_box(b["y_lo"], b["y_hi"])
        if field in ("pose_theta", "beta"):
            return _scalar_box(-_PI, _PI)
        if field == "linear_vel_x":
            return _scalar_box(b["v_min"], b["v_max"])
        if field == "linear_vel_y":
            return _scalar_box(-b["spd_hi"], b["spd_hi"])
        if field == "linear_vel_magnitude":
            return _scalar_box(0.0, b["spd_hi"])
        if field == "ang_vel_z":
            return _scalar_box(-b["yaw_rate"], b["yaw_rate"])
        if field == "delta":
            return _scalar_box(b["s_min"], b["s_max"])
        if field == "frenet_pose":  # [s, ey, ephi]
            return Box(
                low=np.array([0.0, -_EY_LIMIT, -_PI], f32),
                high=np.array([b["s_arc"], _EY_LIMIT, _PI], f32),
                dtype=f32,
            )
        if field == "std_state":  # [X, Y, steering, speed, yaw, yaw_rate, beta]
            lo = np.array([b["x_lo"], b["y_lo"], b["s_min"], b["v_min"], -_PI, -b["yaw_rate"], -_BETA_LIMIT], f32)
            hi = np.array([b["x_hi"], b["y_hi"], b["s_max"], b["v_max"], _PI, b["yaw_rate"], _BETA_LIMIT], f32)
            return Box(low=lo, high=hi, dtype=f32)
        if field == "state":  # KS: [x,y,delta,v,yaw]; ST adds [yaw_rate, beta]
            lo = [b["x_lo"], b["y_lo"], b["s_min"], b["v_min"], -_PI, -b["yaw_rate"], -_BETA_LIMIT]
            hi = [b["x_hi"], b["y_hi"], b["s_max"], b["v_max"], _PI, b["yaw_rate"], _BETA_LIMIT]
            n = sim.state_dim
            while len(lo) < n:  # pad any extra dims (e.g. MB) with a finite fallback
                lo.append(-_TIME_LIMIT)
                hi.append(_TIME_LIMIT)
            return Box(low=np.array(lo[:n], f32), high=np.array(hi[:n], f32), dtype=f32)
        raise ValueError(f"no space builder for observation field {field!r}")

    def space(self) -> gym.Space:
        sim = self._sim
        bounds = self._physical_bounds()
        fields = self._selected_fields()
        agent_spaces: dict[str, gym.Space] = {}
        for agent_id in self.env.unwrapped.agent_ids:
            agent_spaces[agent_id] = gym.spaces.Dict(
                {field: self._field_space(field, sim, bounds) for field in fields}
            )
        return gym.spaces.Dict(agent_spaces)

    def observe(self):
        sim = self._sim
        state = self._state
        fields = self._selected_fields()
        needs_derived = any(field in _DERIVED_FIELDS for field in fields)

        beam_count = sim.scan_num_beams
        observations: dict[str, dict[str, np.ndarray]] = {}

        for idx, agent_id in enumerate(self.env.unwrapped.agent_ids):
            scan = (
                state.scans[idx, :beam_count]
                if beam_count > 0
                else np.empty((0,), dtype=np.float32)
            )
            std_state = state.standard_state[idx]
            base_values: dict[str, np.ndarray] = {
                # copy: scan is a view into the live sim buffer, which is
                # overwritten in place every step. copy=False would alias it,
                # silently corrupting stored scans (e.g. RL replay buffers).
                "scan": scan.astype(np.float32),
                "std_state": std_state.astype(np.float32),
                "state": state.state[idx].astype(np.float32),
                "collision": np.asarray(state.collisions[idx], dtype=np.float32),
                "lap_time": np.asarray(self.env.unwrapped.lap_times[idx], dtype=np.float32),
                "lap_count": np.asarray(self.env.unwrapped.lap_counts[idx], dtype=np.float32),
                "sim_time": np.asarray(self.env.unwrapped.sim_time, dtype=np.float32),
            }
            if self.env.unwrapped.compute_frenet:
                base_values["frenet_pose"] = state.frenet[idx].astype(np.float32)

            derived_values: dict[str, np.ndarray] = {}
            if needs_derived:
                speed = std_state[3]
                beta = std_state[6]
                vx = speed * np.cos(beta)
                vy = speed * np.sin(beta)
                derived_values = {
                    "pose_x": np.asarray(std_state[0], dtype=np.float32),
                    "pose_y": np.asarray(std_state[1], dtype=np.float32),
                    "pose_theta": np.asarray(std_state[4], dtype=np.float32),
                    "linear_vel_x": np.asarray(vx, dtype=np.float32),
                    "linear_vel_y": np.asarray(vy, dtype=np.float32),
                    "linear_vel_magnitude": np.asarray(speed, dtype=np.float32),
                    "ang_vel_z": np.asarray(std_state[5], dtype=np.float32),
                    "delta": np.asarray(std_state[2], dtype=np.float32),
                    "beta": np.asarray(beta, dtype=np.float32),
                }

            agent_obs: dict[str, np.ndarray] = {}
            for field in fields:
                if field in base_values:
                    agent_obs[field] = base_values[field]
                else:
                    agent_obs[field] = derived_values[field]
            observations[agent_id] = agent_obs

        return observations



