from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .dynamic_models import (
    VehicleParameters,
)
from .simulator import F110Simulator
from .env_config import (
    EnvConfig,
    LoopCounterMode,
    RewardMode,
)
from .integrators import integrator_from_type
from .action import (
    get_action_space,
    from_single_to_multi_action_space,
)
from .observation import ObservationType, observation_factory
from .reset import make_reset_fn
from .rendering import make_renderer
from .track import Track


_INF = float("inf")


class RenderClock:
    """Decouples rendering from stepping.

    The user owns the step loop and calls ``env.render()`` explicitly, so all
    pacing/frame-emission decisions live here and are driven from
    ``F110Env.render()``. There are two independent clocks:

    * A **sim-time frame accumulator** (advanced by ``timestep`` once per
      ``step()``) that decides which steps emit a *distinct* rgb_array frame,
      so recorded video stays smooth regardless of how fast the sim runs.
    * A **wall-clock display gate** that, in human modes, caps redraws to at
      most ``render_fps`` per wall-second -- so stepping the dynamics faster
      than real time never forces more frames.

    Human-mode pacing holds ``sim_time / wall_time == real_time_factor`` using
    an absolute anchor so it self-corrects and never accumulates drift.
    ``real_time_factor == inf`` disables pacing (free-run).
    """

    def __init__(self, render_fps: float, real_time_factor: float, timestep: float):
        self.render_fps = float(render_fps)
        self.frame_period = 1.0 / self.render_fps  # sim seconds (video) / wall seconds (human cap)
        self.rtf = float(real_time_factor)
        self.timestep = float(timestep)
        self.reset()

    def reset(self) -> None:
        self._accum = 0.0
        self._frame_is_new = True  # first frame after reset always emits
        self._wall_anchor = None
        self._sim_anchor = None
        self._last_draw_wall = None

    def advance(self) -> bool:
        """Advance the sim-time accumulator by one timestep. Call once per ``step()``."""
        self._accum += self.timestep
        if self._accum + 1e-9 >= self.frame_period:
            self._accum -= self.frame_period
            if self._accum >= self.frame_period:
                # timestep coarser than a frame period: emit every step, no backlog
                self._accum = 0.0
            self._frame_is_new = True
        else:
            self._frame_is_new = False
        return self._frame_is_new

    @property
    def frame_is_new(self) -> bool:
        return self._frame_is_new

    def display_due(self) -> bool:
        """Human-mode wall-clock gate: True at most ``render_fps`` times per wall-second."""
        now = time.perf_counter()
        if self._last_draw_wall is None or (now - self._last_draw_wall) >= self.frame_period - 1e-4:
            self._last_draw_wall = now
            return True
        return False

    def pace(self, sim_time: float) -> None:
        """Sleep to hold ``sim/wall == rtf`` (human modes only). No-op if ``rtf == inf``."""
        if self.rtf == _INF:
            return
        now = time.perf_counter()
        if self._wall_anchor is None:
            self._wall_anchor, self._sim_anchor = now, sim_time
            return
        target = self._wall_anchor + (sim_time - self._sim_anchor) / self.rtf
        slack = target - now
        if slack > 0:
            time.sleep(slack)
        elif slack < -0.25:
            # fell far behind (e.g. a slow step or the sim can't keep up): re-anchor
            self._wall_anchor, self._sim_anchor = now, sim_time

    def set_rtf(self, rtf: float) -> None:
        self.rtf = float(rtf)
        self._wall_anchor = None  # re-anchor so a mid-episode change has no catch-up spike


class F110Env(gym.Env):
    """
    OpenAI Gym environment for F1TENTH autonomous racing.

    Simulates 1/10th scale autonomous race cars with realistic physics,
    LiDAR sensing, and collision detection.

    Attributes:
        track: The racing track used for simulation.
        sim: The underlying F110Simulator instance.
        num_agents: Number of agents in the environment.
        ego_idx: Index of the ego agent.
        vehicle_params: Vehicle parameters for dynamics.
    """

    metadata = {"render_modes": ["human", "human_fast", "rgb_array", "unlimited"], "render_fps": 100}

    def __init__(
        self,
        config: EnvConfig = EnvConfig(),
        render_mode=None,
    ):
        super().__init__()
        if isinstance(config, EnvConfig):
            resolved_config = config
        else:
            raise TypeError("config must be an EnvConfig instance")

        # `metadata` is a class attribute; copy it per-instance so that mutating
        # metadata["render_fps"] on one env never aliases another env's value.
        self.metadata = dict(type(self).metadata)

        self.env_config = resolved_config
        self.render_mode = render_mode
        self.renderer = None
        self.render_config = None
        self.render_obs = None

        self._apply_env_config()
        self._initialize_components()

    def configure(self, config: EnvConfig | None) -> None:
        """
        Reconfigure the environment with new settings.

        Args:
            config: New environment configuration, or None to skip.
        """
        if config is None:
            return
        if not isinstance(config, EnvConfig):
            raise TypeError("config must be an EnvConfig or None")

        self.env_config = config
        self._apply_env_config()
        self._initialize_components()

    def _apply_env_config(self) -> None:
        cfg = self.env_config

        self.seed = cfg.seed
        self.map = cfg.map_name
        self.map_scale = cfg.map_scale

        self.vehicle_params = cfg.params

        self.num_agents = cfg.num_agents
        self.ego_idx = cfg.ego_index

        self.control_cfg = cfg.control_config
        self.simulation_cfg = cfg.simulation_config
        self.observation_cfg = cfg.observation_config
        self.reset_cfg = cfg.reset_config
        self.lidar_cfg = cfg.lidar_config
        self.render_cfg = cfg.render_config
        self.termination_cfg = cfg.termination_config
        self.reward_cfg = cfg.reward_config
        self.dr_cfg = cfg.domain_randomization_config

        self.max_episode_steps = self.termination_cfg.max_episode_steps
        self.terminate_on_collision = self.termination_cfg.terminate_on_collision
        self.collision_agents = self.termination_cfg.collision_agents

        self.longitudinal_action_type = self.control_cfg.longitudinal_mode
        self.steer_action_type = self.control_cfg.steering_mode
        self.steer_delay_steps = self.control_cfg.steer_delay_steps

        self.timestep = self.simulation_cfg.timestep
        self.integrator_fn = integrator_from_type(self.simulation_cfg.integrator)
        self.model = self.simulation_cfg.dynamics_model
        self.loop_counter_mode = self.simulation_cfg.loop_counter
        self.compute_frenet = self.simulation_cfg.compute_frenet_frame
        self.max_laps = self.simulation_cfg.max_laps

        self.collision_check_mode = cfg.collision_check
        self.render_enabled = cfg.render_enabled

    def _resolve_track(self) -> Track:
        map_source = self.map
        if isinstance(map_source, Track):
            return map_source
        if isinstance(map_source, (str, Path)):
            map_str = str(map_source)
            map_path = Path(map_source)
            if "/" in map_str or "\\" in map_str or map_path.suffix:
                return Track.from_track_path(map_path, track_scale=self.map_scale)
            return Track.from_track_name(map_str, track_scale=self.map_scale)
        raise TypeError("map must be a Track instance or a path/name string")

    def _initialize_components(self) -> None:
        if self.render_enabled and self.renderer is not None:
            self.renderer.close()
        self.renderer = None
        self.render_config = None

        self.track = self._resolve_track()

        self.sim = F110Simulator(
            env_config=self.env_config,
            vehicle_params=self.vehicle_params,
            model=self.model,
            dynamics_fn=self.model.f_dynamics,
            integrator_fn=self.integrator_fn,
            longitudinal_type=self.longitudinal_action_type,
            steering_type=self.steer_action_type,
            track=self.track,
            seed=self.seed,
        )
        # NOTE: F110Simulator.__init__ already sets the map from `track` above
        # (and builds the scan cache from it), so no explicit set_map is needed.

        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        if self.loop_counter_mode is LoopCounterMode.FRENET_BASED:
            self.cumulative_s = np.zeros((self.num_agents,))
            self.agents_prev_s = np.zeros((self.num_agents,))
        elif self.loop_counter_mode is LoopCounterMode.WINDING_ANGLE:
            self.cumulative_angle = np.zeros((self.num_agents,))
            self.agents_prev_angle = np.zeros((self.num_agents, 2))
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_times_finish = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        # per-agent previous Frenet s for the progress reward (independent of the
        # lap counter so it works under any LoopCounterMode)
        self._reward_prev_s = np.zeros((self.num_agents,))
        self.sim_time = 0.0

        if self.loop_counter_mode is LoopCounterMode.WINDING_ANGLE and self.track is not None:
            cl = self.track.centerline
            # Polygon area centroid (Shoelace formula) — true centroid of the enclosed area,
            # guaranteed inside for convex tracks and robust for typical racing circuits.
            xs = np.asarray(cl.xs, dtype=np.float64)
            ys = np.asarray(cl.ys, dtype=np.float64)
            x = np.append(xs, xs[0])
            y = np.append(ys, ys[0])
            cross = x[:-1] * y[1:] - x[1:] * y[:-1]
            area = 0.5 * np.sum(cross)
            cx = np.sum((x[:-1] + x[1:]) * cross) / (6.0 * area)
            cy = np.sum((y[:-1] + y[1:]) * cross) / (6.0 * area)
            self._winding_point = np.array([float(cx), float(cy)])
            # Determine CW vs CCW from the first two raceline points relative to winding point.
            rl = self.track.raceline
            fp = np.array([rl.xs[0], rl.ys[0]]) - self._winding_point
            sp = np.array([rl.xs[1], rl.ys[1]]) - self._winding_point
            self._winding_direction = float(np.sign(
                np.arctan2(fp[0] * sp[1] - fp[1] * sp[0], fp.dot(sp))
            ))
        else:
            self._winding_point = None
            self._winding_direction = 1.0

        obs_kwargs: dict[str, Any] = {"type": self.observation_cfg.type}
        if self.observation_cfg.features is not None:
            obs_kwargs["features"] = self.observation_cfg.features
        self.observation_type = observation_factory(env=self, **obs_kwargs)
        self.observation_space = self.observation_type.space()
        self.render_obs_type = observation_factory(env=self, type=ObservationType.DIRECT)
        self.render_obs = None

        single_action_space = get_action_space(
            self.longitudinal_action_type,
            self.steer_action_type,
            self.vehicle_params,
        )
        self.action_space = from_single_to_multi_action_space(
            single_action_space, self.num_agents
        )

        self.reset_fn = make_reset_fn(
            track=self.track,
            num_agents=self.num_agents,
            type=self.reset_cfg.strategy,
            **self.reset_cfg.reset_kwargs(),
        )

        # metadata["render_fps"] is the RecordVideo *container* framerate. Frames are
        # captured once per env.render() call (== once per step), so this must be
        # round(1/timestep) for real-time video playback. It is deliberately NOT
        # render_config.render_fps, which is the (separate) display/emit cadence.
        base_fps = int(round(1.0 / self.timestep)) if self.timestep > 0 else 0
        self.metadata["render_fps"] = base_fps

        # render_mode picks the initial real-time factor; plain "human" uses the
        # configured value, "human_fast" is legacy 10x sugar, "unlimited" is free-run.
        if self.render_mode == "human_fast":
            initial_rtf = 10.0
        elif self.render_mode == "unlimited":
            initial_rtf = float("inf")
        else:
            initial_rtf = self.render_cfg.real_time_factor

        self._render_clock = RenderClock(
            render_fps=self.render_cfg.render_fps,
            real_time_factor=initial_rtf,
            timestep=self.timestep,
        )
        self._last_frame = None
        self._elapsed_steps = 0

        if self.render_enabled:
            self.renderer, self.render_config = make_renderer(
                params=self.vehicle_params,
                track=self.track,
                agent_ids=self.agent_ids,
                render_mode=self.render_mode,
                render_config=self.render_cfg,
            )
        else:
            self.renderer = None
            self.render_config = None

    def _check_done(self):
        """
        Check if the current rollout is done
        """
        if (
            self.loop_counter_mode is LoopCounterMode.FRENET_BASED
            and self.compute_frenet
            and self.track is not None
        ):
            s_frame_max = self.track.centerline.spline.s_frame_max
            for ind in range(self.num_agents):
                current_s = float(self.sim.state.frenet[ind, 0])
                delta_s = current_s - self.agents_prev_s[ind]
                # Correct for wraparound: a forward crossing makes delta_s
                # sharply negative; a backward crossing makes it sharply positive.
                if delta_s < -0.5 * s_frame_max:
                    delta_s += s_frame_max
                elif delta_s > 0.5 * s_frame_max:
                    delta_s -= s_frame_max
                self.cumulative_s[ind] += delta_s
                self.agents_prev_s[ind] = current_s
                new_lap_count = int(self.cumulative_s[ind] / s_frame_max)
                if new_lap_count > self.lap_counts[ind] and self.sim_time > self.timestep:
                    self.lap_counts[ind] = new_lap_count
                    self.lap_times[ind] = self.sim_time - self.lap_times_finish[ind]
                    self.lap_times_finish[ind] = self.sim_time

        elif (
            self.loop_counter_mode is LoopCounterMode.WINDING_ANGLE
            and self._winding_point is not None
        ):
            wp = self._winding_point
            wd = self._winding_direction
            for ind in range(self.num_agents):
                pose = self.sim.state.poses[ind]
                curr_vec = np.array([pose[0] - wp[0], pose[1] - wp[1]])
                prev_vec = self.agents_prev_angle[ind]  # stored as 2-vector
                # Signed angle from prev to curr, scaled by racing direction
                cross = float(prev_vec[0] * curr_vec[1] - prev_vec[1] * curr_vec[0])
                dot = float(prev_vec.dot(curr_vec))
                delta_angle = np.arctan2(cross, dot) * wd
                self.cumulative_angle[ind] += delta_angle
                self.agents_prev_angle[ind] = curr_vec
                new_lap_count = int(self.cumulative_angle[ind] / (2.0 * np.pi))
                if new_lap_count > self.lap_counts[ind] and self.sim_time > self.timestep:
                    self.lap_counts[ind] = new_lap_count
                    self.lap_times[ind] = self.sim_time - self.lap_times_finish[ind]
                    self.lap_times_finish[ind] = self.sim_time

        terminated = False
        if self.terminate_on_collision:
            if self.collision_agents == "any":
                terminated = bool(np.any(self.sim.collisions))
            else:  # "ego"
                terminated = bool(self.sim.collisions[self.ego_idx])
        if self.max_laps is not None:
            terminated = terminated or (self.lap_counts[self.ego_idx] >= self.max_laps)
        return terminated

    def _sample_vehicle_params(self):
        """Draw a randomized VehicleParameters from the DR ranges (env RNG)."""
        changes = {
            name: float(self.np_random.uniform(lo, hi))
            for name, (lo, hi) in self.dr_cfg.param_ranges.items()
        }
        return self.vehicle_params.with_updates(**changes)

    def _compute_progress(self) -> np.ndarray:
        """Per-agent forward Frenet arclength progress (metres) since the last
        step, wrap-corrected. Zeros when the Frenet frame is not computed."""
        progress = np.zeros(self.num_agents)
        if not (self.compute_frenet and self.track is not None):
            return progress
        s_max = self.track.centerline.spline.s_frame_max
        for ind in range(self.num_agents):
            s = float(self.sim.state.frenet[ind, 0])
            ds = s - self._reward_prev_s[ind]
            if ds < -0.5 * s_max:
                ds += s_max
            elif ds > 0.5 * s_max:
                ds -= s_max
            progress[ind] = ds
            self._reward_prev_s[ind] = s
        return progress

    def _compute_reward(self, obs, action, progress, info, terminated, truncated) -> float:
        cfg = self.reward_cfg
        if cfg.mode is RewardMode.CUSTOM:
            return float(cfg.reward_fn(obs, action, info, terminated, truncated))
        if cfg.mode is RewardMode.SURVIVAL:
            return self.timestep
        # PROGRESS
        ego = self.ego_idx
        reward = cfg.progress_weight * float(progress[ego])
        if cfg.velocity_weight:
            reward += cfg.velocity_weight * float(self.sim.state.standard_state[ego, 3])
        if cfg.timestep_weight:
            reward += cfg.timestep_weight * self.timestep
        if cfg.collision_penalty and self.sim.collisions[ego] > 0:
            reward -= cfg.collision_penalty
        return reward

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        # call simulation step
        self.sim.step(action)
        self._elapsed_steps += 1

        # advance the render clock's sim-time frame accumulator (drives fixed-fps
        # frame emission; decoupled from how often the user calls render()).
        self._render_clock.advance()

        # check done
        terminated = self._check_done()

        # per-agent forward progress this step (also feeds the progress reward)
        progress = self._compute_progress()

        # observation
        obs = self.observation_type.observe()
        if self.render_enabled and self.renderer is not None:
            if self.observation_cfg.type is ObservationType.DIRECT:
                self.render_obs = copy.deepcopy(obs)
            else:
                self.render_obs = copy.deepcopy(self.render_obs_type.observe())

        self.sim_time = self.sim.state.sim_time
        truncated = (
            self.max_episode_steps is not None
            and self._elapsed_steps >= self.max_episode_steps
        )
        # copy: these are the env's live arrays, mutated every step; without a
        # copy a stored info dict would change retroactively (breaks logging).
        info = {
            "lap_times": self.lap_times.copy(),
            "lap_counts": self.lap_counts.copy(),
            "sim_time": self.sim_time,
            "collisions": self.sim.collisions.copy(),  # per-agent collision flags
            "progress": progress,  # per-agent forward arclength this step (m)
        }
        # reward is computed last so a CUSTOM reward_fn sees the final info
        reward = self._compute_reward(obs, action, progress, info, terminated, truncated)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the gym environment by given poses

        Args:
            seed: random seed for the reset
            options: dictionary of options for the reset containing initial poses of the agents

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        super().reset(seed=seed)

        self.sim_time = 0.0
        self._elapsed_steps = 0
        self._render_clock.reset()
        self._last_frame = None
        if self.loop_counter_mode is LoopCounterMode.FRENET_BASED:
            self.cumulative_s.fill(0.0)
            self.agents_prev_s.fill(0.0)
        elif self.loop_counter_mode is LoopCounterMode.WINDING_ANGLE:
            self.cumulative_angle.fill(0.0)
            self.agents_prev_angle.fill(0.0)
        self.lap_counts.fill(0.0)
        self.lap_times.fill(0.0)
        self.lap_times_finish.fill(0.0)
        if options is not None and "poses" in options:
            poses = options["poses"]
            option = "pose"
        elif options is not None and "states" in options:
            poses = options["states"]
            option = "state"
        else:
            poses = self.reset_fn.sample(self.np_random)
            option = "pose"


        if option == "pose":
            assert isinstance(poses, np.ndarray) and poses.shape == (
                self.num_agents,
                3,
            ), "Initial poses must be a numpy array of shape (num_agents, 3)"
        elif option == "state":
            assert isinstance(poses, np.ndarray) and poses.shape == (
                self.num_agents,
                self.model.state_dim,
            ), f"Initial full state must be a numpy array of shape (num_agents, {self.model.state_dim})"
        else:
            raise ValueError("Invalid reset option.")

        # Domain randomization: sample vehicle params for this episode (from the
        # env RNG, so it is reproducible with reset(seed=...)) and push them to
        # the sim (rebuilds params array + scan/collision caches).
        if self.dr_cfg.enabled and self.dr_cfg.param_ranges:
            self.sim.update_params(self._sample_vehicle_params())

        # Derive the LiDAR-noise seed from the env RNG (which gymnasium seeds
        # from reset(seed=...)), so the noise stream is controlled by the reset
        # seed rather than being byte-identical every episode.
        noise_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.sim.reset(poses, option=option, noise_seed=noise_seed)

        # seed the progress-reward reference from the spawn arclength so the
        # first step's progress is ~0, not the whole spawn s
        if self.compute_frenet and self.track is not None:
            self._reward_prev_s[:] = self.sim.state.frenet[:, 0]

        if (
            self.loop_counter_mode is LoopCounterMode.FRENET_BASED
            and self.compute_frenet
            and self.track is not None
        ):
            # Seed the lap-progress reference from the spawn arclength so laps
            # are measured from where the car starts, not from the spline's s=0
            # datum. Without this a car spawned at s>0 completes lap 1 early
            # (its spawn s is counted as progress on the first step).
            self.agents_prev_s[:] = self.sim.state.frenet[:, 0]

        if (
            self.loop_counter_mode is LoopCounterMode.WINDING_ANGLE
            and self._winding_point is not None
        ):
            wp = self._winding_point
            for ind in range(self.num_agents):
                pose = self.sim.state.poses[ind]
                self.agents_prev_angle[ind] = np.array([pose[0] - wp[0], pose[1] - wp[1]])

        obs = self.observation_type.observe()
        if self.render_enabled and self.renderer is not None:
            if self.observation_cfg.type is ObservationType.DIRECT:
                self.render_obs = copy.deepcopy(obs)
            else:
                self.render_obs = copy.deepcopy(self.render_obs_type.observe())

        # copy: these are the env's live arrays, mutated every step; without a
        # copy a stored info dict would change retroactively (breaks logging).
        info = {"lap_times": self.lap_times.copy(), "lap_counts": self.lap_counts.copy(), "sim_time": self.sim_time}

        return obs, info

    def update_map(self, map_name: Track | str):
        """
        Updates the map used by simulation

        Args:
            map_name (Track | str): name of the map, path to map, or Track instance

        Returns:
            None
        """
        new_config = self.env_config.with_updates(map_name=map_name)
        self.configure(new_config)

    def update_params(self, params, index=-1):
        """
        Update the shared vehicle parameters used by the simulator and renderers.

        Args:
            params (VehicleParameters): new vehicle parameters.
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        if index >= 0:
            raise NotImplementedError(
                "Per-agent parameter updates are not supported in the simplified simulator"
            )
        if isinstance(params, VehicleParameters):
            vehicle_params = params
        else:
            raise TypeError("params must be a VehicleParameters instance")

        self.vehicle_params = vehicle_params
        self.env_config = self.env_config.with_updates(params=vehicle_params)

        self.sim.update_params(self.vehicle_params)
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.update_params(self.vehicle_params)
        if hasattr(self, "action_space"):
            single_action_space = get_action_space(
                self.longitudinal_action_type,
                self.steer_action_type,
                self.vehicle_params,
            )
            self.action_space = from_single_to_multi_action_space(
                single_action_space, self.num_agents
            )            

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """
        if self.render_enabled and self.renderer is not None:
            self.renderer.add_renderer_callback(callback_func)

    def set_real_time_factor(self, real_time_factor: float) -> None:
        """Change the real-time factor at runtime (mid-episode is fine).

        The real-time factor is sim-seconds simulated per wall-clock second in
        the human render modes: ``1.0`` = real time, ``5.0`` = 5x faster,
        ``float("inf")`` = no pacing (free-run). It has no effect on the
        physics or on rgb_array output. The pacer re-anchors on change so there
        is no catch-up sleep or fast-forward spike.

        Args:
            real_time_factor: positive float, or ``float("inf")`` for free-run.
        """
        if not (real_time_factor > 0):
            raise ValueError(
                f"real_time_factor must be > 0 (or float('inf')), got {real_time_factor}"
            )
        self._render_clock.set_rtf(real_time_factor)

    @property
    def real_time_factor(self) -> float:
        """Current real-time factor (see :meth:`set_real_time_factor`)."""
        return self._render_clock.rtf

    @property
    def render_fps(self) -> float:
        """Target fixed frame rate (display cap in human modes, emit cadence in rgb_array)."""
        return self._render_clock.render_fps

    @property
    def frame_is_new(self) -> bool:
        """True on steps that emit a distinct frame at the render_fps sim-time cadence.

        Use this for manual, exact-fps video capture (append a frame only when
        it is True) rather than relying on RecordVideo's one-frame-per-step.
        """
        return self._render_clock.frame_is_new

    def render(self, mode=None):
        """
        Render the current state. Rendering is decoupled from stepping by an
        internal render clock (see ``RenderClock``): the frame rate is fixed by
        ``render_config.render_fps`` and the pace is set by the real-time factor,
        both independent of how fast the user steps the simulation.

        Args:
            mode (str, optional): overrides the mode set at ``gym.make`` for this
                call. Defaults to the configured ``render_mode``. One of
                "human", "human_fast", "unlimited", "rgb_array".

        Returns:
            np.ndarray | None: an RGB frame (H, W, 3) in "rgb_array" mode; None in
            the human modes.
        """
        m = mode or self.render_mode
        if (
            m not in self.metadata["render_modes"]
            or not self.render_enabled
            or self.renderer is None
        ):
            return None

        clk = self._render_clock

        if m in ("human", "human_fast", "unlimited"):
            # Wall-clock gate: draw at most render_fps times/second regardless of
            # how fast the sim is stepped. Then pace the loop to the real-time factor.
            if clk.display_due():
                self.renderer.update(obs=self.render_obs)
                self.renderer.render()
            clk.pace(self.sim_time)
            return None

        # rgb_array / rgb_array_list: never sleep, always return a frame. Grab a
        # fresh one only on the sim-time cadence; return the cached frame in
        # between so RecordVideo (one capture per step) yields smooth video.
        if clk.frame_is_new or self._last_frame is None:
            self.renderer.update(obs=self.render_obs)
            self._last_frame = self.renderer.render()
        return self._last_frame

    def close(self):
        """
        Ensure renderer is closed upon deletion
        """
        if self.render_enabled and self.renderer is not None:
            self.renderer.close()
        super().close()
