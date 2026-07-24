"""Pure-pursuit waypoint follower AND a fully-spelled-out configuration reference.

Every field of ``EnvConfig`` and all of its nested config dataclasses is set
explicitly in ``main()`` -- mostly to its *default* value -- so this file
doubles as a live catalogue of every knob and what it defaults to. Where a value
must deviate from the default for the follower to work (e.g. the observation
type), the deviation is flagged with a ``# default: ...`` comment.

Rendering needs an X display (real or ``xvfb``); set ``render_enabled=False`` in
the config below to run fully headless.
"""
import time
from typing import Tuple

import gymnasium as gym
import numpy as np
from numba import njit

from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.dynamic_models import (
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
)
from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.reset import ResetStrategy
from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.env_config import (
    ControlConfig,
    DomainRandomizationConfig,
    EnvConfig,
    LoopCounterMode,
    ObservationConfig,
    RenderConfig,
    ResetConfig,
    RewardConfig,
    RewardMode,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.rendering import make_lidar_scan_callback

"""
Planner Helpers
"""


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start).astype(
            np.float32
        )  # NOTE: specify type or numba complains

        a = np.dot(V, V)
        b = np.float32(2.0) * np.dot(
            V, start - point
        )  # NOTE: specify type or numba complains
        c = (
            np.dot(start, start)
            + np.dot(point, point)
            - np.float32(2.0) * np.dot(start, point)
            - radius * radius
        )
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = (end - start).astype(np.float32)

            a = np.dot(V, V)
            b = np.float32(2.0) * np.dot(
                V, start - point
            )  # NOTE: specify type or numba complains
            c = (
                np.dot(start, start)
                + np.dot(point, point)
                - np.float32(2.0) * np.dot(start, point)
                - radius * radius
            )
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)], dtype=np.float32),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


class PurePursuitPlanner:
    """
    Example Planner
    """

    def __init__(self, track, wb):
        self.wheelbase = wb
        self.waypoints = np.stack(
            [track.raceline.xs, track.raceline.ys, track.raceline.vxs]
        ).T
        self.max_reacquire = 20.0

        self.drawn_waypoints = []
        self.lookahead_point = None
        self.current_index = None

        self.lookahead_point_render = None
        self.local_plan_render = None

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        # NOTE: specify type or numba complains
        self.waypoints = np.loadtxt(
            conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip
        ).astype(np.float32)

    def render_lookahead_point(self, e):
        """
        Callback to render the lookahead point.
        """
        if self.lookahead_point is not None:
            points = self.lookahead_point[:2][None]  # shape (1, 2)~
            if self.lookahead_point_render is None:
                self.lookahead_point_render = e.get_points_renderer(
                    points, color=(228, 26, 28), size=10
                )
            else:
                self.lookahead_point_render.update(points)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.current_index is not None:
            points = self.waypoints[self.current_index : self.current_index + 10, :2]
            if self.local_plan_render is None:
                self.local_plan_render = e.get_lines_renderer(
                    points, color=(255, 255, 51), size=5
                )
            else:
                self.local_plan_render.update(points)
                
    def get_render_callbacks(self):
        return [self.render_lookahead_point, self.render_local_plan]

    def _get_current_waypoint(
        self, waypoints, lookahead_distance, position, theta
    ) -> Tuple[np.ndarray, int]:
        """
        Returns the current waypoint to follow given the current pose.

        Args:
            waypoints: The waypoints to follow (Nx3 array)
            lookahead_distance: The lookahead distance [m]
            position: The current position (2D array)
            theta: The current heading [rad]

        Returns:
            waypoint: The current waypoint to follow (x, y, speed)
            i: The index of the current waypoint
        """
        wpts = waypoints[:, :2]
        lookahead_distance = np.float32(lookahead_distance)
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            t1 = np.float32(i + t)
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, t1, wrap=True
            )
            if i2 is None:
                return None, None
            current_waypoint = np.empty((3,), dtype=np.float32)
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, -1]
            return current_waypoint, i
        elif nearest_dist < self.max_reacquire:
            # return the full (x, y, speed) row as float32: get_actuation needs
            # index [2] (speed), and the njit dot needs a consistent dtype.
            return waypoints[i, :].astype(np.float32), i
        else:
            return None, None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point, i = self._get_current_waypoint(
            self.waypoints, lookahead_distance, position, pose_theta
        )

        if lookahead_point is None:
            return 4.0, 0.0

        # for rendering
        self.lookahead_point = lookahead_point
        self.current_index = i

        # actuation
        speed, steering_angle = get_actuation(
            pose_theta,
            self.lookahead_point,
            position,
            lookahead_distance,
            self.wheelbase,
        )
        speed = vgain * speed

        return speed, steering_angle

def build_config() -> EnvConfig:
    """Build a *fully-explicit* EnvConfig.

    Every field of every config dataclass is listed below, almost all set to
    their real default. This is the point of the file: a copy-paste template
    where you can see every knob and its default at a glance, and flip only what
    you need. ``EnvConfig()`` with no arguments produces the same thing (minus
    the observation-type deviation noted below).
    """
    return EnvConfig(
        # ---- top-level -----------------------------------------------------
        seed=12345,                 # base RNG seed (LiDAR/DR/actuator-noise streams)
        map_name="Spielberg",       # track name (downloaded on first use), a path,
                                    #   or a Track instance
        map_scale=1.0,              # >1 enlarges the track
        num_agents=1,               # rows in the sim; >1 is multi-agent
        ego_index=0,                # which agent is "ego" (0 <= ego_index < num_agents)
        collision_check=CollisionCheckMode.LIDAR_SCAN,  # LIDAR_SCAN | BOUNDING_BOX | NONE
        render_enabled=True,        # False -> no renderer built, runs fully headless

        # Vehicle physical parameters. F1TENTH_VEHICLE_PARAMETERS is the default
        # 1/10-scale preset; override individual fields with .with_updates(...).
        # Key F1TENTH defaults (SI units):
        #   m=3.74 (mass)   mu=1.0489 (friction)   I=0.04712 (yaw inertia)
        #   lf=0.15875  lr=0.17145 (CoG->axle)     h=0.074 (CoG height)
        #   s_min/s_max=-/+0.4189 (steer angle)    sv_min/sv_max=-/+3.2 (steer rate)
        #   v_min=-5.0  v_max=20.0                  a_max=9.51   v_switch=7.319
        #   width=0.31  length=0.58
        params=F1TENTH_VEHICLE_PARAMETERS,
        # e.g. params=F1TENTH_VEHICLE_PARAMETERS.with_updates(m=4.0, mu=0.9),

        # ---- control: how the two action columns are interpreted -----------
        control_config=ControlConfig(
            longitudinal_mode=LongitudinalActionType.SPEED,   # SPEED | ACCL
            steering_mode=SteerActionType.STEERING_ANGLE,     # STEERING_ANGLE | STEERING_SPEED
            steer_delay_steps=0,    # steering-command ring-buffer lag (steps)
            throttle_delay_steps=0, # longitudinal-command lag (steps)
            steer_noise_std=0.0,    # Gaussian noise on the steer command (sim2real)
            accl_noise_std=0.0,     # Gaussian noise on the longitudinal command
        ),

        # ---- physics -------------------------------------------------------
        simulation_config=SimulationConfig(
            timestep=0.01,                        # env step (s); must be a multiple
            integrator_timestep=0.01,             #   of integrator_timestep
            integrator=IntegratorType.RK4,        # RK4 | EULER
            dynamics_model=DynamicModel.ST,       # ST (single-track) | KS | MB(broken)
            loop_counter=LoopCounterMode.FRENET_BASED,  # FRENET_BASED | WINDING_ANGLE
            compute_frenet_frame=True,            # needed for FRENET_BASED laps + frenet obs
            max_laps=1,                           # None -> run forever
        ),

        # ---- observations --------------------------------------------------
        # DEFAULT is ObservationType.DIRECT (scan + std_state + state + lap info).
        # We use KINEMATIC_STATE so the follower can read pose_x/pose_y/pose_theta
        # directly (DIRECT does not expose those derived fields -- see CLAUDE.md).
        observation_config=ObservationConfig(
            type=ObservationType.KINEMATIC_STATE,  # default: DIRECT
            features=None,          # only used with type=FEATURES (a custom field tuple)
        ),

        # ---- reset / spawn -------------------------------------------------
        reset_config=ResetConfig(
            strategy=ResetStrategy.RL_GRID_STATIC,  # RL_{GRID,RANDOM}_{STATIC,RANDOM} | MAP_RANDOM_STATIC
            min_dist=None,          # None -> strategy default (agent spacing, m)
            max_dist=None,
            shuffle=None,           # None -> strategy default (spawn-order shuffle)
            move_laterally=None,    # None -> strategy default (lateral offset off line)
        ),

        # ---- LiDAR ---------------------------------------------------------
        lidar_config=LiDARConfig(
            enabled=True,
            num_beams=1080,                 # ~2000 is the internal angular-LUT ceiling
            field_of_view=4.712389,         # 270 deg. NOTE: only honoured on a FRESH
                                            #   LiDARConfig; with_updates(field_of_view=)
                                            #   does NOT re-derive the angles.
            angle_min=None,                 # None -> -field_of_view/2
            angle_max=None,                 # None -> +field_of_view/2
            range_min=0.0,
            range_max=30.0,
            noise_std=0.01,                 # per-beam Gaussian range noise (obs only)
            dropout_prob=0.0,               # per-beam no-return probability (sim2real)
            range_bias_std=0.0,             # per-beam systematic bias, drawn once/episode
            base_link_to_lidar_tf=(0.275, 0.0, 0.0),  # (x, y, yaw) sensor offset
        ),

        # ---- rendering (pacing + visuals) ----------------------------------
        render_config=RenderConfig(
            render_fps=60,          # max redraws / wall-second (human) or frame cadence
                                    #   in sim-time (rgb_array). Raise it to draw more often.
            real_time_factor=1.0,   # sim-seconds per wall-second in human modes;
                                    #   float("inf") = free-run. (render_mode="unlimited"
                                    #   forces inf; "human_fast" forces 10.)
            window_size=800,        # square pixel size (also the rgb_array/video size)
            focus_on="agent_0",     # camera-follow target; None = whole-map view
            vehicle_palette=(       # per-agent car colours, cycled by index
                "#FD3754", "#377eb8", "#984ea3", "#e41a1c", "#ff7f00",
                "#a65628", "#f781bf", "#888888", "#a6cee3", "#b2df8a",
            ),
            show_wheels=True,
            render_map_img=True,    # draw the occupancy map underneath
            car_thickness=1,
            bigger_car_when_map_centered=True,
            show_lap_info=True,     # overlay lap time / count
        ),

        # ---- termination / truncation --------------------------------------
        termination_config=TerminationConfig(
            max_episode_steps=None,      # None -> no step-based truncation
            terminate_on_collision=True,
            collision_agents="ego",      # "ego" | "any"
        ),

        # ---- reward --------------------------------------------------------
        reward_config=RewardConfig(
            mode=RewardMode.SURVIVAL,    # SURVIVAL (=timestep) | PROGRESS | CUSTOM
            progress_weight=1.0,         # the weights below apply to PROGRESS only
            velocity_weight=0.0,
            timestep_weight=0.0,
            collision_penalty=0.0,
            reward_fn=None,              # required iff mode=CUSTOM:
                                         #   (obs, action, info, terminated, truncated) -> float
        ),

        # ---- domain randomization (per-episode, at reset) ------------------
        domain_randomization_config=DomainRandomizationConfig(
            enabled=False,
            # ABSOLUTE (low, high) ranges keyed by VehicleParameters field name:
            param_ranges={},             # e.g. {"m": (3.0, 4.0), "mu": (0.9, 1.1)}
        ),
    )


def main():
    """Run the pure-pursuit follower on the fully-specified config above."""
    # Pure-pursuit tuning (planner-side, not part of the env config):
    #   tlad  = lookahead distance [m];  vgain = speed scaling of the raceline profile
    tlad = 0.82461887897713965 * 1.5
    vgain = 1.0

    cfg = build_config()

    # render_mode picks the initial pacing (overrides render_config.real_time_factor):
    #   "human"      -> real time (rtf from render_config, default 1.0)
    #   "human_fast" -> 10x real time
    #   "unlimited"  -> free-run: RTF as fast as the CPU allows (rtf = inf)
    #   "rgb_array"  -> returns frames, never sleeps (for RecordVideo)
    # Redraws are still capped at render_config.render_fps in every human mode.
    env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="human")

    track = env.unwrapped.track
    planner = PurePursuitPlanner(
        track=track,
        wb=F1TENTH_VEHICLE_PARAMETERS.lf + F1TENTH_VEHICLE_PARAMETERS.lr,
    )

    # Render callbacks are per-env-instance; only wire them up when a renderer
    # exists (render_enabled=True and a display is available).
    renderer = env.unwrapped.renderer
    if renderer is not None:
        track.raceline.render_waypoints(renderer)
        for r in planner.get_render_callbacks():
            env.unwrapped.add_render_callback(r)
        # LiDAR scan drawn as an rviz-style point cloud, to confirm scans work.
        env.unwrapped.add_render_callback(
            make_lidar_scan_callback("agent_0", cfg.lidar_config, color=(255, 0, 0), size=2)
        )

    obs, info = env.reset(seed=cfg.seed)  # or reset(options={"poses": (n,3)}) / {"states": ...}
    env.render()

    laptime = 0.0
    done = False
    start = time.time()
    while not done:
        action = np.zeros((cfg.num_agents, 2), dtype=np.float32)  # [[steer, speed], ...]
        for i, agent_id in enumerate(obs.keys()):
            speed, steer = planner.plan(
                obs[agent_id]["pose_x"],
                obs[agent_id]["pose_y"],
                obs[agent_id]["pose_theta"],
                tlad,
                vgain,
            )
            action[i] = (steer, speed)          # steering FIRST, then longitudinal
        obs, step_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        laptime += step_reward
        env.render()

    print("Sim elapsed time:", laptime, "Real elapsed time:", time.time() - start)
    env.close()


if __name__ == "__main__":
    main()
