# CLAUDE.md — f1tenth_gym (branch `dev-humble`)

## What this is

A multi-agent F1TENTH racing simulator behind a Gymnasium `Env`. **This fork has diverged sharply from upstream f1tenth_gym — do not rely on prior knowledge of it.** `origin` is the upstream org repo (`github.com/f1tenth/f1tenth_gym`); `dev-humble` is a long-running integration branch that is **476 commits ahead of / 0 behind `origin/main`**. `origin/main` is a legacy line still shipping `gym/f110_gym/` + `setup.py` — a *different package*. Branch off `dev-humble`, never `main`.

Key divergences from what you may remember:
- **There is no `RaceCar` class.** `grep -rn "RaceCar\|update_pose"` over `f1tenth_gym/ tests/ examples/` → zero hits. One [`F110Simulator`](f1tenth_gym/envs/simulator.py#L43) drives struct-of-arrays buffers ([`SimulationState`](f1tenth_gym/envs/state.py#L11)). **Agents are rows, not objects.**
- **There is no dict/YAML config.** [`F110Env.__init__`](f1tenth_gym/envs/f110_env.py#L52) raises `TypeError` unless it gets an `EnvConfig` instance. All config classes are **frozen dataclasses** mutated via `with_updates()`.
- "humble" refers to the ROS 2 Humble *platform* (Ubuntu 22.04, [Dockerfile:23](Dockerfile#L23)). **There is zero ROS code in the tree** — no `rclpy`, `package.xml`, `ament`, or `colcon` anywhere.

Live source is `f1tenth_gym/`, `tests/`, `examples/`. Everything else is noise (see *Ignore this cruft*).

---

## Quickstart (verified)

```python
import gymnasium as gym, numpy as np
from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig, SimulationConfig
from f1tenth_gym.envs.observation import ObservationType

cfg = EnvConfig(
    map_name="Spielberg",
    num_agents=1,
    simulation_config=SimulationConfig(max_laps=None),   # default is 1 -> ends after one lap!
    observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
    render_enabled=False,
)
env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)     # namespaced id, see gotchas
obs, info = env.reset()

for _ in range(100):
    action = np.array([[0.0, 2.0]], dtype=np.float32)    # [[steer_rad, speed_mps]] — steer FIRST
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(obs["agent_0"]["pose_x"], obs["agent_0"]["linear_vel_x"], info["sim_time"])
env.close()
```

Real output: `speed=1.984`, `reward=0.01`, `sim_time=1.0` — stable. **x/y vary between runs** (roughly x≈−1.6…−2.4, y≈−1.26…−1.46): `env.reset()` with no `seed` lets gymnasium seed `np_random` from OS entropy, and `RL_GRID_STATIC` → `GridResetFn` draws the spawn waypoint via `rng.choice` ([masked_reset.py:34](f1tenth_gym/envs/reset/masked_reset.py#L34)) — its `shuffle=False, move_laterally=False` suppress pose shuffling and lateral offset, *not* the waypoint draw. `EnvConfig.seed` does not cover this (it only reaches the LiDAR noise RNGs — see gotchas). Pass `env.reset(seed=42)` for a fixed start pose (`x=-1.567 y=-1.257`, reproducible).

**First run downloads the map** (`maps/` is gitignored; [track/utils.py:28-44](f1tenth_gym/envs/track/utils.py#L28) fetches `https://api.f1tenth.org/Spielberg.tar.xz`). Network required.

Repo history: `git log --oneline -20`; open work: `git branch -r`. Branch names are GitHub issue numbers — unmerged: `91-unexpected-collision-tracking-trajectory-in-monza`, `119-configure-misses-observation`, `125-rendering-is-flipped-ud`.

Dev commands:
```bash
uv sync    # installs dev group (pytest, black, isort, flake8, matplotlib)
           # ...AND uninstalls anything not in uv.lock — this removes moviepy, see gotchas

env -u PYTHONPATH uv run pytest   # 133 tests. The `env -u` is REQUIRED when ROS 2 Humble is on
                                  # PYTHONPATH: /opt/ros/humble's launch_testing registers a pytest11
                                  # plugin whose import chain needs `lark`, absent from the venv ->
                                  # ModuleNotFoundError before collection. Config: pyproject.toml
                                  # [tool.pytest.ini_options]

uv run flake8 f1tenth_gym tests examples --statistics   # advisory only; setup.cfg is the only lint
           # config, and its `exclude` does NOT cover .venv/ or gym_env/ — bare `flake8 .` returns
           # ~443k lines of venv noise. Scoped to live source: 382 findings.
```

---

## Architecture at a glance

```
import f1tenth_gym                       # side effect: gym.register("f1tenth-v0")  __init__.py:3-6
gym.make(config=EnvConfig(...))
  └─ F110Env.__init__                                     f110_env.py:46-64
       ├─ _apply_env_config()      flatten frozen tree -> ~20 attrs      :82-112
       └─ _initialize_components() track, sim, spaces, reset_fn, renderer :126-227

env.step(action)                                          f110_env.py:283-318
  ├─ sim.step(action)                                     simulator.py:222-282
  │    ├─ control_inputs[:,0]=steer, [:,1]=longitudinal            :232-233
  │    ├─ state.push_delay(steer)          FIFO ring buffer        :236-240
  │    ├─ for agent in range(N):           ← PYTHON LOOP, not vectorised :242
  │    │     steering_fn / longitudinal_fn  -> raw efforts         :244-245
  │    │     for _ in substeps: integrator_fn(dynamics_fn, x, u, dt, params_array) :248-255
  │    │     wrap yaw to [-pi,pi); write state / standard_state / poses  :256-263
  │    ├─ track.cartesian_to_frenet per agent (if compute_frenet)  :265-273
  │    ├─ _update_scans()      scans + WALL collision              :275-276  ← gated on scan_enabled
  │    └─ _update_agent_collisions()   agent-vs-agent              :281      ← NOT gated (crash, see gotchas)
  ├─ done = _check_done()   lap counter AND termination            :301   ← must precede observe()
  ├─ obs  = observation_type.observe()                             :304
  ├─ reward = self.timestep  (0.01, pure survival time)            :312
  └─ truncated = False  (hardcoded, no TimeLimit wrapper)          :315
```

---

## Subsystem map

| Dir / file | Responsibility | Entry point |
|---|---|---|
| [`f1tenth_gym/__init__.py`](f1tenth_gym/__init__.py) | The *entire* gym registration, no kwargs | `gym.register(id="f1tenth-v0")` |
| [`envs/f110_env.py`](f1tenth_gym/envs/f110_env.py) | gym.Env lifecycle, reward, termination, **all lap counting** | `F110Env` :29 |
| [`envs/simulator.py`](f1tenth_gym/envs/simulator.py) | SoA physics loop, substeps, LiDAR, collisions | `F110Simulator` :43 |
| [`envs/state.py`](f1tenth_gym/envs/state.py) | The SoA buffers + steer-delay ring buffer | `SimulationState` :11 |
| [`envs/env_config.py`](f1tenth_gym/envs/env_config.py) | The frozen config tree | `EnvConfig` :134 |
| [`envs/dynamic_models/`](f1tenth_gym/envs/dynamic_models/__init__.py) | KS/ST/MB models, `VehicleParameters`, presets | `DynamicModel` :280 |
| [`envs/integrators.py`](f1tenth_gym/envs/integrators.py) | Euler / RK4 (plain Python, **not** jitted) | `IntegratorType` :7 |
| [`envs/action.py`](f1tenth_gym/envs/action.py) | Action enums, PID/pass-through, action space | `get_action_space` :72 |
| [`envs/observation/`](f1tenth_gym/envs/observation/__init__.py) | Field vocabulary, presets, factory | `observation_factory` :111 |
| [`envs/lidar/`](f1tenth_gym/envs/lidar/laser_models.py) | `ScanSimulator2D`, EDT sphere-tracing, `ray_cast` | `ScanSimulator2D` :426 |
| [`envs/collision_models.py`](f1tenth_gym/envs/collision_models.py) | `CollisionCheckMode`, numba GJK, `get_vertices` | :35 |
| [`envs/track/`](f1tenth_gym/envs/track/track.py) | Map loading/download, splines, Frenet frame | `Track` :41 |
| [`envs/reset/`](f1tenth_gym/envs/reset/__init__.py) | Start-pose strategies (a real registry) | `make_reset_fn` :89 |
| [`envs/rendering/`](f1tenth_gym/envs/rendering/__init__.py) | Two PyQt6 backends + render callbacks | `make_renderer` :19 |

---

## The core loop in detail

**Construction order is load-bearing.** [`simulator.py:97`](f1tenth_gym/envs/simulator.py#L97) calls `model.get_initial_state(...)`, which assigns `state_dim`/`control_dim` **as a side effect onto the IntEnum member** ([dynamic_models/__init__.py:305-315](f1tenth_gym/envs/dynamic_models/__init__.py#L305)) — a process-global singleton. Line `:99` reads `model.control_dim` and works *only* because `:97` ran; [`f110_env.py:366`](f1tenth_gym/envs/f110_env.py#L366) reads `model.state_dim` only because the simulator was built first. Reorder → `AttributeError`.

**Params cross the numba boundary as a flat float32 array.** [`simulator.py:94`](f1tenth_gym/envs/simulator.py#L94): `params_array = vehicle_params.to_array(model)` = `np.asarray(astuple(self), float32)[: model.parameter_count()]` ([:136-138](f1tenth_gym/envs/dynamic_models/__init__.py#L136)). Every `@njit` kernel indexes it **positionally** (`mu = params[0]` … `v_max = params[15]`). You cannot pass a dict or dataclass into any jitted function. **Field declaration order is an ABI.**

**Integrator ↔ dynamics.** `rk4_integration` ([integrators.py:36-50](f1tenth_gym/envs/integrators.py#L36)) is **plain Python** — there is no `njit` import in that file. It crosses the Python↔numba boundary **4× per substep per agent per step**. Constraints (`steering_constraint`, `accl_constraints`, [utils.py:24-84](f1tenth_gym/envs/dynamic_models/utils.py#L24)) are applied **inside each derivative**, so they re-apply at every RK4 stage against *intermediate* states, not once per step.

**Precision.** Params are float32, `state` is float32 ([state.py:38](f1tenth_gym/envs/state.py#L38)), but njit kernels compute in float64 and the result is re-cast to float32 every step ([simulator.py:257](f1tenth_gym/envs/simulator.py#L257)). RK4's accuracy edge is partly thrown away at the step boundary.

**`reset()`** ([f110_env.py:320-390](f1tenth_gym/envs/f110_env.py#L320)): pose source is `options["poses"]` (n,3) → else `options["states"]` (n,state_dim) → else `reset_fn.sample(np_random)`. `sim.reset` ([simulator.py:179-221](f1tenth_gym/envs/simulator.py#L179)) does an **in-place `fill(0.0)`** on every buffer (held references stay valid), then writes `state`/`standard_state`/`poses`/`frenet` (`use_s_guess=False` → global search). It **never calls `_update_scans`**, so the first `obs["agent_0"]["scan"]` after reset is all zeros. Agents start at zero speed and zero steering **only on the `poses`/`reset_fn` path**: the state is allocated `np.zeros(state_dim)` and only `x, y` (`0:2`) and `theta` (`4`) are overwritten from the pose ([dynamic_models/__init__.py:316-320](f1tenth_gym/envs/dynamic_models/__init__.py#L316)), leaving `delta`/`v` at zero. That is the only branch that calls `get_initial_state` ([simulator.py:196](f1tenth_gym/envs/simulator.py#L196)); `options["states"]` writes the full state verbatim ([:199-202](f1tenth_gym/envs/simulator.py#L199)) and **can spawn at any speed/steering**.

---

## Dynamic models & state layouts

| Model | state_dim | Layout | Pose reference | Status |
|---|---|---|---|---|
| `KS`=1 | 5 | `[x, y, delta, v, yaw]` | **rear axle** ([kinematic.py:100](f1tenth_gym/envs/dynamic_models/kinematic.py#L100)) | ok, tested |
| `ST`=2 (default) | 7 | `[x, y, delta, v, yaw, yaw_rate, beta]` | **CoG** ([single_track.py:114](f1tenth_gym/envs/dynamic_models/single_track.py#L114)) | ok, tested |
| `MB`=3 | 29 | multi-body + Pacejka | CoG | **BROKEN — returns NaN** |

`standard_state` is always `(N, 7)`: `[X, Y, steering_angle, speed, yaw, yaw_rate, beta]`. **Docstrings claim index 6 is `V_Y`; it is the SLIP ANGLE beta** ([single_track.py:174](f1tenth_gym/envs/dynamic_models/single_track.py#L174)) — consumers correctly do `vy = speed*sin(beta)` ([full.py:139-141](f1tenth_gym/envs/observation/full.py#L139)). Under KS, `std_state[5]` and `[6]` are hardcoded 0.0, so `ang_vel_z`, `beta`, `linear_vel_y` are always exactly 0.

**Pose frame is not uniform.** `obs["pose_x"]` means the rear axle under KS and the CoG under ST — switching model silently shifts the reported pose by `lr` (0.17145 m for F1TENTH). The discrepancy is reconciled in exactly two places, [`_compute_collision_body_offset`](f1tenth_gym/envs/simulator.py#L373) and [`_build_scan_cache`](f1tenth_gym/envs/simulator.py#L303), both testing `if model != DynamicModel.KS` — i.e. *"is it KS"*, **not** *"is it rear-axle referenced"*. Any new rear-axle model gets silently shifted by `-lr`.

Presets: `F1TENTH_` / `F1FIFTH_` / `FULLSCALE_VEHICLE_PARAMETERS` ([dynamic_models/__init__.py:144](f1tenth_gym/envs/dynamic_models/__init__.py#L144)). Only FULLSCALE populates the MB/Pacejka fields; the two small-scale presets leave all 69 MB fields at `math.nan`.

Units: metres, radians, m/s, m/s², rad/m (curvature). Yaw wrapped to `[-π, π)` at [simulator.py:256](f1tenth_gym/envs/simulator.py#L256) — `(x + π) % 2π - π`, so `+π` maps to `-π`: **`-π` is attainable, `+π` is not.** (Same formula, with correct `# wrap to [-pi, pi]` comments, at [track.py:466,505](f1tenth_gym/envs/track/track.py#L466).)

---

## Observations, actions, resets, config

**Action space is always `Box(shape=(num_agents, 2))`**, columns `[steer, longitudinal]` — **steering FIRST** ([action.py:93-96](f1tenth_gym/envs/action.py#L93) vs [simulator.py:232-233](f1tenth_gym/envs/simulator.py#L232)). Both are float32, so a swap fails **silently**. Single-agent code must pass `np.array([[steer, speed]])`.

| Longitudinal mode | maps via | space bounds |
|---|---|---|
| `ACCL`=1 | identity | `[-a_max, +a_max]` |
| `SPEED`=2 (default) | `pid_accl` (real P controller, 4 gain quadrants) | `[v_min, v_max]` = `[-5, 20]` for F1TENTH |

| Steer mode | maps via | space bounds |
|---|---|---|
| `STEERING_ANGLE`=1 (default) | `pid_steer` — **bang-bang, NOT a PID**: `±max_sv` when \|err\|>1e-4 ([utils.py:86-95](f1tenth_gym/envs/dynamic_models/utils.py#L86)) | `[s_min, s_max]` |
| `STEERING_SPEED`=2 | identity | `[sv_min, sv_max]` |

**Observation types** ([observation/__init__.py:16](f1tenth_gym/envs/observation/__init__.py#L16)): `DIRECT`(default) / `ORIGINAL`(alias of DIRECT) / `FEATURES` / `KINEMATIC_STATE` / `DYNAMIC_STATE` / `FRENET_DYNAMIC_STATE`. All six resolve to `FullObservation` with a different field tuple. Result is `dict[agent_id -> dict[field -> ndarray]]`, keys `agent_0`, `agent_1`, …

| Field group | Names |
|---|---|
| Base (8, or 7 without Frenet) — the DIRECT set | `scan` (num_beams,), `std_state` (7,), `state` (state_dim,), `collision` (), `lap_time` (), `lap_count` (), `sim_time` (), `frenet_pose` (3,) = `(s, ey, ephi)` |
| Derived (9) | `pose_x`, `pose_y`, `pose_theta`, `linear_vel_x`, `linear_vel_y`, `linear_vel_magnitude`, `ang_vel_z`, `delta`, `beta` — all 0-d |

**Every scalar is a 0-d float32 ndarray, not a Python float** — built by `np.asarray(..., dtype=np.float32)` in `observe()` ([full.py:128-131](f1tenth_gym/envs/observation/full.py#L128) base, [:143-151](f1tenth_gym/envs/observation/full.py#L143) derived); the matching `shape=()` *spaces* come from `_scalar_box` ([full.py:38-39](f1tenth_gym/envs/observation/full.py#L38)). This is load-bearing: it keeps the numba `np.dot` in `examples/waypoint_follow.py` type-consistent.

> **Trap 1:** `DIRECT` does **not** contain `pose_x` — it is a *derived* field. `obs["agent_0"]["pose_x"]` under the default config raises `KeyError`. Use `KINEMATIC_STATE` or read `std_state`.
>
> **Trap 2:** `DIRECT` is **conditional**. With `compute_frenet_frame=False`, `_selected_fields` drops `frenet_pose` → `obs[...]["frenet_pose"]` raises `KeyError`; requesting it explicitly via `FEATURES` raises `ValueError: frenet_pose requested but environment does not compute the Frenet frame` ([full.py:85-92](f1tenth_gym/envs/observation/full.py#L85)). With `lidar_config.enabled=False`, `scan` has shape `(0,)`, not `(num_beams,)` ([full.py:118-122](f1tenth_gym/envs/observation/full.py#L118)).

**Reset strategies** ([reset/__init__.py:32](f1tenth_gym/envs/reset/__init__.py#L32)): `RL_GRID_STATIC`(default) / `RL_RANDOM_STATIC` / `RL_GRID_RANDOM` / `RL_RANDOM_RANDOM` / `MAP_RANDOM_STATIC`. All RL_* bind to `track.raceline` ([:57](f1tenth_gym/envs/reset/__init__.py#L57)), **never** the centerline, and all pass `move_laterally=False` — so multi-agent "grid" resets put every car **on** the raceline, separated only longitudinally.

**Config** — 14 top-level `EnvConfig` fields ([env_config.py:134](f1tenth_gym/envs/env_config.py#L134)). Defaults: `seed=12345, map_name="Spielberg", map_scale=1.0, params=F1TENTH, num_agents=1, ego_index=0, collision_check=LIDAR_SCAN, render_enabled=True`, plus `ControlConfig(SPEED, STEERING_ANGLE, steer_delay_steps=0)`, `SimulationConfig(timestep=0.01, integrator_timestep=0.01, RK4, ST, FRENET_BASED, compute_frenet_frame=True, max_laps=1)`, `ObservationConfig(DIRECT, None)`, `ResetConfig(RL_GRID_STATIC)`, `LiDARConfig(1080 beams, fov=4.712389, range 0–30, noise_std=0.01, tf=(0.275,0,0))`, `RenderConfig(render_fps=60, real_time_factor=1.0, frame_output_method="auto")`.

Nested mutation must nest: `cfg.with_updates(params=cfg.params.with_updates(mu=1.0))`, then `env.unwrapped.configure(cfg2)`.

---

## LiDAR & collision

The scan answers both *"what does the car see?"* and *"did it crash?"*.

[`_update_scans`](f1tenth_gym/envs/simulator.py#L506) per step:
1. Precompute all agents' collision-body vertices into `self._all_vertices` (`:508-516`).
2. Per agent: `_lidar_pose_from_base(pose)` → noise-free `ScanSimulator2D.scan` (`:523`) → sphere-trace beams through the map's **EDT** (`resolution * distance_transform_edt`, metres).
3. **WALL check**: `check_ttc_jit(scan_clean, ...)` (`:527-538`). On hit: `state[i, 3:] = 0.0`, `collisions[i] = 1.0`.
4. `adjusted_scan = scan_clean` — **an ALIAS, no copy** (`:542`) — then `ray_cast` shortens beams hitting each opponent (`:543-546`). **`ray_cast` mutates its scan argument in place** ([laser_models.py:421-422](f1tenth_gym/envs/lidar/laser_models.py#L421)) — verified.
5. Gaussian noise + clip → `state.scans[i]` (`:550-555`). **Noise is applied only to the observation, never to the collision path.**

**Beam angles are quantised to a 2000-entry LUT.** `trace_ray` never uses the beam angle directly — it indexes `sines`/`cosines` built from `np.linspace(0, 2π, theta_dis=2000)` with a truncated `int(theta_index)` ([laser_models.py:146-149](f1tenth_gym/envs/lidar/laser_models.py#L146), `:214-247`, `:488-490`), so every ray snaps to a **0.00314 rad grid** — up to ~72% of the default 0.00437 rad `angle_increment` (1080 beams / 270°). `theta_dis` is a hardcoded `ScanSimulator2D.__init__` default, **unreachable from `LiDARConfig`**, so raising `num_beams` past ~2000 buys no angular resolution. The collision path (`cache.angles`, `ray_cast`) uses *exact* float angles and therefore disagrees with the scan. This is why [test_scan_sim.py](tests/test_scan_sim.py) can only assert `mse < 2.0`.

**`check_ttc_jit` does not compute TTC.** The iTTC math is commented out ([laser_models.py:269-283](f1tenth_gym/envs/lidar/laser_models.py#L269)); the live body is `np.any(scan - side_distances <= ttc_thresh)`. So `F110Simulator.ttc_threshold = 0.005` is a **distance margin in metres, not a time**, collision is velocity-independent, and `vel`/`cosines` are dead parameters.

| `CollisionCheckMode` | Agent-vs-agent | Wall | Symmetry |
|---|---|---|---|
| `LIDAR_SCAN`=1 (default) | `check_ttc_jit` on opponent-shortened scan | scan check | **asymmetric** — A can flag while B doesn't |
| `BOUNDING_BOX`=2 | O(n²) GJK over all i<j pairs | **still the scan check** (misnomer) | symmetric — both bodies flagged |

`get_vertices` returns corners ordered `[rear-left, rear-right, front-right, front-left]` — `ray_cast` depends on this winding ([collision_models.py:278-280](f1tenth_gym/envs/collision_models.py#L278)).

---

## Track & racelines

`maps/` is **gitignored** (`.gitignore:173`, `**/maps/*`); only `maps/.gitkeep` is tracked. Tracks download from a **hardcoded** `https://api.f1tenth.org/<name>.tar.xz` into the repo-root `maps/` resolved by four `.parent` hops ([track/utils.py:28-44](f1tenth_gym/envs/track/utils.py#L28)) — works only for editable installs. No checksum verification; extraction is CVE-2007-4559-hardened via `filter="data"`.

Four constructors: `from_track_name` (tries `{stem}.yaml` then legacy `{stem}_map.yaml`), `from_track_path` (**only** the legacy name), `from_refline(x, y, velx)` (mapless synthetic), `from_raceline_file`.

**Fallback rules are asymmetric and crash.** `from_track_name` sets `centerline=None` if `{track}_centerline.csv` is missing and only does `raceline = centerline` ([track.py:169-184](f1tenth_gym/envs/track/track.py#L169)); it lacks the `if centerline is None: centerline = raceline` guard that `from_track_path` has ([:253-256](f1tenth_gym/envs/track/track.py#L253)). A dir with only a raceline CSV → `track.centerline is None` → `cartesian_to_frenet` raises `AttributeError: 'NoneType' object has no attribute 'spline'` on the first `reset()` (verified). Same trigger for any track dir whose stem contains spaces: `find_track_dir` matches on `stem.replace(" ","")` while the CSV lookup uses the raw string. Also: `from_centerline_file` hardcodes `fixed_speed=1.0` ([raceline.py:80](f1tenth_gym/envs/track/raceline.py#L80)), so `track.centerline.vxs` is all 1.0 m/s and is **not** a usable speed profile (Spielberg: centerline.vxs=[1,1,1] vs raceline.vxs=[8,8,8]).

**Frenet frame**: `(s, ey, ephi)` in metres/metres/radians. `+ey` is **LEFT** of the direction of travel (signed by the spline normal `[-sin(yaw), cos(yaw)]`, [track.py:500-505](f1tenth_gym/envs/track/track.py#L500)). **The frame is always the CENTERLINE, never the raceline** — `cartesian_to_frenet(..., use_raceline=False)` is the default ([track.py:469,482](f1tenth_gym/envs/track/track.py#L469)) and both sim call sites ([simulator.py:214](f1tenth_gym/envs/simulator.py#L214), [:269](f1tenth_gym/envs/simulator.py#L269)) pass no override; there is no config knob. But every RL_* reset binds `track.raceline` ([reset/__init__.py:57](f1tenth_gym/envs/reset/__init__.py#L57)), so **`obs[...]["frenet_pose"][1]` (ey) is non-zero at spawn** — verified 0.809 m on Spielberg. FRENET_BASED lap counting also divides `cumulative_s` by the *centerline* `s_frame_max` (343.32 m) while the car tracks the raceline (338.13 m) — a 1.5% lap-length error.

**Occupancy maps** are loaded `FLIP_TOP_BOTTOM` so row 0 = smallest world y, and binarised at a **hardcoded 128** — the yaml's `negate`/`occupied_thresh`/`free_thresh` are parsed and **never used**.

`CubicSplineND` interpolates a 7-channel `[x, y, cos ψ, sin ψ, k, vx, ax]` matrix over chordal arclength, `bc_type="periodic"`. Yaw goes through cos/sin channels so it wraps correctly.

**Every reference line is force-closed into a loop** ([cubic_spline.py:41-44](f1tenth_gym/envs/track/cubic_spline.py#L41)): if `points[-1][:2] != points[0][:2]` the first point is appended, else the last is overwritten with the first — `bc_type="periodic"` requires it. Consequence: `Track.from_refline(x=np.linspace(0,10), y=np.zeros(...))` (the [run_in_empty_track.py](examples/run_in_empty_track.py) pattern, whose docstring advertises "standard maneuvers") is **not a 10 m straight** — it is a closed ~20 m path with a phantom return leg, and `s`, curvature and lap counting all run over it. **There is no open-path mode.**

---

## Rendering

Fully decoupled: `F110Env` hands the renderer an immutable `render_obs` deepcopy once per step; the renderer never touches the sim. Two backends behind `EnvRenderer` ([renderer.py:40](f1tenth_gym/envs/rendering/renderer.py#L40), 7 abstract methods): `PyQtEnvRenderer` (2D pyqtgraph) and `PyQtEnvRendererGL` (GL, the default `render_type="pyqt6gl"`).

`render_mode` ∈ `{"human", "human_fast", "rgb_array", "unlimited"}` ([f110_env.py:44](f1tenth_gym/envs/f110_env.py#L44)). Extension hook: `env.unwrapped.add_render_callback(fn)` where `fn(env_renderer) -> None`; see [`make_lidar_scan_callback`](f1tenth_gym/envs/rendering/callbacks.py#L54) and `PurePursuitPlanner.get_render_callbacks()`. Callbacks read `env_renderer.obs` (now set by **both** backends).

**Rendering is decoupled from stepping by `RenderClock`** ([f110_env.py](f1tenth_gym/envs/f110_env.py) — class above `F110Env`), driven from `F110Env.render()`. Two independent clocks: a wall-clock **display cap** (human modes redraw at most `render_fps`/wall-second, so stepping faster than real time never forces more frames) and a sim-time **frame accumulator** (rgb_array grabs a distinct frame every `1/render_fps` sim-seconds; the cached frame is returned in between so `RecordVideo` yields smooth video). Human-mode **pacing** holds `sim/wall == real_time_factor` via an absolute anchor (`inf` = free-run). All configured via `EnvConfig.render_config` ([`RenderConfig`](f1tenth_gym/envs/env_config.py): `render_fps=60`, `real_time_factor=1.0`, `frame_output_method="auto"`). Runtime toggle: `env.unwrapped.set_real_time_factor(x)`; read-only `env.unwrapped.{real_time_factor, render_fps, frame_is_new}`. `human_fast`→rtf 10, `unlimited`→rtf ∞ (legacy sugar). `metadata["render_fps"]` stays `round(1/timestep)` (RecordVideo container fps for real-time playback) — deliberately *not* `render_config.render_fps`.

**Offscreen (rgb_array) dispatch is by `$DISPLAY`** ([rendering/__init__.py](f1tenth_gym/envs/rendering/__init__.py#L1), `_use_gl_for_offscreen`): with a display (real X or `xvfb-run`) → fast GL framebuffer grab under `QT_QPA_PLATFORM=xcb` (~1.4 ms GPU / ~2 ms xvfb); without one → 2D raster exporter under `offscreen` (~3 ms, works headless with **zero setup**, e.g. Colab before `apt-get install xvfb`). Override with `frame_output_method="gl"|"2d"`. Note the GL widget **cannot** render under the bare `offscreen` platform (no FBO) — that was the old silent-downgrade root cause. Both backends now return **contiguous RGB (H, W, 3) uint8** pinned to a square `window_size`. `RenderConfig.frame_output_method` reaches `make_renderer` via a `RenderSpec`; the rest of `RenderSpec` (palette, window size, car model) is still only settable by editing its defaults.

Caveat: `QT_QPA_PLATFORM` is process-global (locked at first `QApplication`), so one process cannot mix GL (`xcb`) and 2D (`offscreen`) renderers — the first renderer's platform wins.

---

## Testing & dev workflow

13 test modules, 133 test functions, all plain `unittest.TestCase`. **No `conftest.py`, no fixtures anywhere.** `pytest` config is [pyproject.toml:50-56](pyproject.toml#L50) (`addopts="-ra"`, `testpaths=["tests","integration"]`).

| File | Pins |
|---|---|
| `tests/test_env_config.py` | 40 tests — all config validation/coercion contracts |
| `tests/test_f110_env.py` | gymnasium `check_env`, `configure()` equivalence, vectorised envs |
| `tests/test_dynamics.py` | KS + ST derivatives vs CommonRoad ground truth. **No MB test exists** |
| `tests/test_scan_sim.py` | `ScanSimulator2D` vs `legacy_scan.npz`, `mse < 2.0` (very loose) |
| `tests/test_track.py`, `test_cubic_spline.py`, `test_observation.py`, `test_action.py`, `test_state.py`, `test_integrators.py`, `test_collision_checks.py`, `test_renderer.py`, `test_utils.py` | — |

**The suite needs network access** (map downloads) and **mutates the working tree** (`test_download_racetrack` renames/rmtrees `maps/Spielberg`).

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)): bare `pytest`, matrix py3.9–3.14, `fail-fast: false`. It triggers on `'dev*'` ([ci.yml:5](.github/workflows/ci.yml#L5)), which **does** match `dev-humble` — pytest runs on every push here. ci.yml installs flake8 but never invokes it.

Lint is a **separate** workflow ([lint.yml:44](.github/workflows/lint.yml#L44)): `flake8 . --statistics --exit-zero` — **it can never fail the build** — and it triggers on `'dev_*'` (underscore, [lint.yml:5](.github/workflows/lint.yml#L5)), which does **not** match `dev-humble`, so it never runs here. A third workflow, `docker.yml`, triggers on `'dev*'` and builds the image without pushing. black/isort/autoflake are declared dev deps with **zero config and zero invocations**.

---

## Gotchas & invariants

**Ordering contracts (none are documented in the code):**
1. `get_initial_state` must run before any `state_dim`/`control_dim` read — they're side-effect attrs on a global IntEnum ([simulator.py:97](f1tenth_gym/envs/simulator.py#L97) → `:99`).
2. **Wall `check_ttc_jit` MUST precede the `ray_cast` loop** (`:527` before `:543-546`). `ray_cast` mutates the aliased `scan_clean` in place — **verified**. Reordering silently turns every close overtake into a wall crash.
3. `_update_scans` must precede `_update_agent_collisions` — currently *broken* when LiDAR is disabled (below).
4. `_check_done` must precede `observe()` ([f110_env.py:301](f1tenth_gym/envs/f110_env.py#L301) before `:304`) because `observe` reads the env's live lap arrays.

**Bugs / footguns:**

| Issue | Evidence |
|---|---|
| **A collision ZEROES THE YAW.** `state[agent_idx, 3:] = 0.0` is meant to kill velocity, but index 4 is yaw. Verified: yaw −2.8798 → 0.0 on collision. Only `state[3]=0` (+ `5:`) can be intended | [simulator.py:535](f1tenth_gym/envs/simulator.py#L535), identical at `:573` |
| **After a collision `state`, `std_state` and `poses` DISAGREE** — only `state` is zeroed, the other three are left stale from `:257-273`. The `state` and `std_state` obs fields contradict each other on any colliding step | same |
| **`lidar_config.enabled=False` CRASHES `step()` in both modes.** `_update_scans` is gated at `:275` but `_update_agent_collisions` runs unconditionally at `:281`. LIDAR_SCAN → `IndexError`; BOUNDING_BOX → `AttributeError: no attribute '_all_vertices'`. **There is no working scan-free config** | `:275-281`, `:564`, `:578` |
| **`LoopCounterMode.TOGGLE` is declared and documented but never implemented.** No branch in `_check_done`, no buffer alloc, no reset zeroing. Silently counts zero laps forever | [env_config.py:39](f1tenth_gym/envs/env_config.py#L39) vs [f110_env.py:233-276](f1tenth_gym/envs/f110_env.py#L233) |
| **`DynamicModel.MB` is reachable from config but returns NaN.** `collision_body_center_x/y` were inserted at dataclass positions 18/19, shifting every MB param by 2 → `K_zt` reads 0.0 → divide-by-zero → `x0[16]=inf`. KS/ST unaffected (they slice the first 18 only). **Treat MB as dead code** | [dynamic_models/__init__.py:59](f1tenth_gym/envs/dynamic_models/__init__.py#L59) |
| **`obs[...]['scan']` is a VIEW of the live buffer, not a copy.** `scan.astype(np.float32, copy=False)` on a same-dtype slice returns `sim.state.scans[idx]` itself, while `state`/`std_state`/`frenet_pose` use `.astype(np.float32)` and **are** copies. Any stored scan mutates in place on the next `step()` — verified `np.shares_memory(...) is True`, delta up to 30.0 m. Fatal for replay buffers and trajectory logs; `.copy()` before storing | [full.py:118-125](f1tenth_gym/envs/observation/full.py#L118) vs `:127` |
| **`BOUNDING_BOX` REBINDS `state.collisions` every step.** `self.state.collisions = np.maximum(...)` allocates a new array instead of writing in place, so any handle captured from `sim.collisions` / `sim.state.collisions` goes stale after one step and reports zero forever (verified: `before is not after` → True; LIDAR_SCAN never rebinds). This is the exact inverse of the SoA in-place invariant every other buffer honours. Should be `self.state.collisions[:] = ...` | [simulator.py:580](f1tenth_gym/envs/simulator.py#L580) vs `:536`, `:574` |
| **`lidar_config.with_updates(field_of_view=X)` silently does nothing.** `__post_init__` derives `angle_min`/`angle_max` from `field_of_view` only when they are `None`; `replace()` carries the already-materialised angles over, and `ScanSimulator2D` computes `fov = angle_max - angle_min`, ignoring the `fov` arg entirely. Verified: `with_updates(field_of_view=2π)` → `field_of_view=6.283` but `angle_min/max` stay at ±2.356 (270°) — **the config object lies about its own FOV.** Pass `field_of_view` to a fresh `LiDARConfig(...)`, or set `angle_min`/`angle_max` explicitly | [lidar/config.py:35-40](f1tenth_gym/envs/lidar/config.py#L35), [laser_models.py:463-471](f1tenth_gym/envs/lidar/laser_models.py#L463) |
| **`Track.from_raceline_file` discards its own map origin.** `track_spec` is constructed twice; the second (`:385-393`) overwrites the first and hardcodes `origin=(0.0, 0.0, 0.0)`, throwing away the margin-based origin computed at `:373` from the raceline extents. The `occupancy_map` is still sized from those extents, so map and origin disagree unless the raceline starts at the world origin; `xy_2_rc` then returns `r=c=-1` and `dt[-1,-1]` silently wraps to the far corner instead of raising | [track.py:375-393](f1tenth_gym/envs/track/track.py#L375), [laser_models.py:77-89](f1tenth_gym/envs/lidar/laser_models.py#L77) |
| **`WINDING_ANGLE` breaks on non-convex tracks.** The winding centre is the shoelace centroid of the **centerline** but the direction sign comes from the first two **raceline** points; the code concedes it is only "guaranteed inside for convex tracks". On a U-shaped or figure-eight circuit the winding count is wrong | [f110_env.py:161-180](f1tenth_gym/envs/f110_env.py#L161) |
| **`Track.s_guess` is a SINGLE scalar shared by every agent.** `simulator.step` loops agents calling `cartesian_to_frenet` with no per-agent guess, forcing each into the previous agent's ±5 m window. Verified 2-agent: s=[20, 200] at reset became [194.73, 198.7] after one step | [track.py:96](f1tenth_gym/envs/track/track.py#L96), [:484-495](f1tenth_gym/envs/track/track.py#L484) |
| **FRENET_BASED never re-seeds `agents_prev_s` from the spawn s** — only zero-fills it, while WINDING_ANGLE *does* resync. So laps are measured from the spline's s=0 datum, not spawn. Verified: reset at s=161.87 → lap 1 fires ~half a lap early. Only the default RL_GRID_STATIC (spawns at s≈0) hides this | [f110_env.py:337-339](f1tenth_gym/envs/f110_env.py#L337) vs `:372-379` |
| **`timestep=0.03, integrator_timestep=0.01` is REJECTED** despite being an exact 3× multiple — `0.03 % 0.01 == 0.00999…` in IEEE754. Use 0.01/0.02/0.04/0.05/0.1 | [simulator.py:83-84](f1tenth_gym/envs/simulator.py#L83) |
| ~~`self.metadata` is the CLASS dict~~ **(FIXED)** — `__init__` now copies it per-instance (`self.metadata = dict(type(self).metadata)`), so `render_fps` no longer aliases across envs | [f110_env.py](f1tenth_gym/envs/f110_env.py) `__init__` |
| **`info` ALIASES the live lap arrays** — `info['lap_counts'] is env.lap_counts` → True. A stored info dict mutates retroactively; logging across steps records only the final state | [f110_env.py:316](f1tenth_gym/envs/f110_env.py#L316) |
| **`reset(seed=...)` does NOT reseed LiDAR noise.** Scan RNGs are re-seeded from the *config* seed (`default_rng(self.seed + idx)`), so the noise stream is byte-identical every episode | [simulator.py:191-193](f1tenth_gym/envs/simulator.py#L191) |
| **`truncated` is hardcoded False** and registration sets no `max_episode_steps` → no TimeLimit wrapper. With `max_laps=None` and a non-crashing agent, episodes run forever | [f110_env.py:315](f1tenth_gym/envs/f110_env.py#L315) |
| **Termination watches ONLY the ego.** Opponent crashes never end the episode; a crashed opponent keeps being integrated from a zeroed yaw | [f110_env.py:278-281](f1tenth_gym/envs/f110_env.py#L278) |
| **`ObservationConfig.features` is silently ignored** for every type except `FEATURES`, and is not validated at config-construction time | [f110_env.py:185-188](f1tenth_gym/envs/f110_env.py#L185) |
| **`ResetConfig` exposes only `strategy`** — `min_dist`/`max_dist`/`start_width`/`shuffle` are unreachable from config despite `make_reset_fn` accepting `**kwargs` | [f110_env.py:202-206](f1tenth_gym/envs/f110_env.py#L202) |
| `silent collision disable`: `_ray_to_rect_distance_vec` returns all-zero `side_distances` if the LiDAR origin falls **outside** the collision rect → check degenerates to `scan <= 0.005` → collisions never fire. The default ST config clears this by only **0.009 m** | [simulator.py:469-474](f1tenth_gym/envs/simulator.py#L469) |
| LiDAR angles are **radians**; validation actively rejects degrees with a "Did you pass degrees instead of radians?" hint | [lidar/config.py:62-71](f1tenth_gym/envs/lidar/config.py#L62) |
| `save_centerline` output **cannot be read back** by `from_centerline_file` (writer emits a uuid comment line before the header). It has zero callers | [track.py:435-436](f1tenth_gym/envs/track/track.py#L435) |
| Backwards laps don't decrement — `int()` truncation + a `>` guard make `lap_counts` monotonically non-decreasing | [f110_env.py:272-273](f1tenth_gym/envs/f110_env.py#L272) |
| `examples/random_trackgen.py` imports **shapely**, which is declared nowhere (not in pyproject, not in uv.lock) | [examples/random_trackgen.py:37-39](examples/random_trackgen.py#L37) |
| Same footgun, worse: `examples/video_recording.py` wraps in `gymnasium.wrappers.RecordVideo`, which hard-requires **moviepy** (`gym.error.DependencyNotInstalled` without it) — declared nowhere, **and `uv sync` actively uninstalls it**. Re-run `uv pip install moviepy` after any sync | [examples/video_recording.py:29](examples/video_recording.py#L29) |
| `PurePursuitPlanner`'s `max_reacquire` branch **crashes** (`numba TypingError: dot(float32, float64)` + a length-2 array indexed at `[2]`). Reachable whenever the car drifts >tlad off the raceline but <20 m | [examples/waypoint_follow.py:276-278](examples/waypoint_follow.py#L276) |

**Dead code / dead fields** (don't be fooled): `SimulationState.lap_counts/.lap_times/.lap_time_last_finish` are allocated, zeroed, and **never read or written** — the env keeps its own parallel float64 arrays. `F110Simulator._ray_to_rect_distance` (scalar, 60 lines) has no callers. `laser_models.py:575-690` is a broken in-module unittest block. `RenderSpec.frame_output_method` is never read.

---

## Navigation: if you're changing X, touch Y

Everything is `IntEnum` + hardcoded `if/elif` dispatch. **There is no registry** except for reset strategies.

| Change | Sites to edit |
|---|---|
| **New lap counter** | `LoopCounterMode` [env_config.py:31](f1tenth_gym/envs/env_config.py#L31) **AND three sites in f110_env.py**: buffer alloc `:150-183`, `_check_done` chain `:229-281`, reset zeroing `:337-345`. Missing one fails **silently** — that is exactly what happened to `TOGGLE` |
| **New dynamics model** | `DynamicModel` [:280](f1tenth_gym/envs/dynamic_models/__init__.py#L280) + `parameter_count()`, `from_string()`, `get_initial_state()`, `f_dynamics`, `get_standardized_state_fn()` (5 sites, same file) + a new `vehicle_dynamics_<x>` module + **check the `!= DynamicModel.KS` tests** at [simulator.py:379](f1tenth_gym/envs/simulator.py#L379) and `:330-336` |
| **New vehicle parameter** | **APPEND ONLY** to `VehicleParameters` — `to_array` slices `astuple()` positionally. Inserting before index 18 silently corrupts KS/ST; before 89 shifts MB |
| **New integrator** | `IntegratorType` [:7](f1tenth_gym/envs/integrators.py#L7) + `integration_fn()` + a module-level `f(f, x, u, dt, *args)` |
| **New action mode** | enum [action.py:7-26](f1tenth_gym/envs/action.py#L7) + transform fn + `*_action_from_type` + **`get_action_space` bounds** (else ValueError) |
| **New observation field** | `_BASE_FIELDS`/`_DERIVED_FIELDS` + `_FIELD_SPACE_BUILDERS` [full.py:42-66](f1tenth_gym/envs/observation/full.py#L42) + `base_values`/`derived_values` `:124-152` — **and the duplicated copies** in [observation/__init__.py:35-56](f1tenth_gym/envs/observation/__init__.py#L35). The vocabulary lives in two files and must stay in sync |
| **New obs preset** | `ObservationType` + `FEATURE_PRESETS` [:66-93](f1tenth_gym/envs/observation/__init__.py#L66) — the factory picks it up automatically |
| **New reset strategy** | `ResetStrategy` + a `ResetFn`/`MaskedResetFn`/`MapResetFn` subclass + `_RESET_BUILDERS` [reset/__init__.py:112-137](f1tenth_gym/envs/reset/__init__.py#L112). Imports live at the **bottom** of that module to break a circular import — follow the pattern |
| **New collision mode** | `CollisionCheckMode` [:35](f1tenth_gym/envs/collision_models.py#L35) + [simulator.py:559-580](f1tenth_gym/envs/simulator.py#L559). The `else` is a **catch-all** — a new member silently falls into the GJK path. Use `elif` + `raise` |
| **New config knob** | field + default on the frozen dataclass + rule in its `__post_init__` + read it in `_apply_env_config` [f110_env.py:82-112](f1tenth_gym/envs/f110_env.py#L82). New nested section also needs the isinstance check in `EnvConfig.__post_init__` `:187-217` |
| **New render backend** | 7 `EnvRenderer` abstractmethods + `update_params` (called unconditionally) + `close()` must free the GL/window resources + an `elif` in `make_renderer` |
| **New render_mode string** | `F110Env.metadata["render_modes"]` `:44`, the render_mode→rtf map in `_initialize_components`, `render()`'s mode branch, and `make_renderer`'s two mode lists |
| **Render pacing / fps / RTF** | `RenderClock` (above `F110Env`) owns it; `render()` calls `display_due`/`frame_is_new`/`pace`. Config in `RenderConfig`; don't reintroduce a sleep in the backends |

---

## Ignore this cruft

- **`docs/html/`** (268 *tracked* files, 105 mentioning the dead **`f110_gym`** package) and **`docs/xml/`** (67 tracked, 60 mentioning it) — 335 tracked generated Doxygen files, 165 documenting the dead package. **There is no `docs/latex/`**: `docs/Doxyfile` sets `GENERATE_LATEX = NO`, and its `INPUT = ../gym/f110_gym` does not exist on `dev-humble`, so these are never regenerated. Ignore entirely.
- **`gym_env/`** — a stale virtualenv. Not covered by the root `.gitignore`; hidden only by a `*` .gitignore uv wrote inside it.
- **`.venv/`**, `__pycache__/`, `video_*/` (untracked run output from `examples/video_recording.py`).
- **`tests/legacy_scan_gen.py`** — STALE and unrunnable: calls the retired `"f110_gym:f110-v0"` id and the pre-fork flat obs layout. The only surviving reference to the dead package name in live source.
- **`origin/main`** — a legacy `gym/f110_gym/` + setup.py tree. A different package. Do not read it for context.

Repo hygiene nits (no runtime effect): `uv.lock` and `video_*/` are untracked and unignored; `pyproject.toml` `testpaths` lists a non-existent `integration/` dir (pytest drops it silently); [waypoint_follow.py:352](examples/waypoint_follow.py#L352) has a pasted LLM prompt as a trailing comment.
