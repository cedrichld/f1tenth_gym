Configuration
=============

Every :class:`~f1tenth_gym.envs.f110_env.F110Env` is built from a single
configuration object: an :class:`~f1tenth_gym.envs.env_config.EnvConfig`. This
fork has **no dict/YAML config and no keyword soup** — you pass one typed,
validated object and get one environment. This page is the central reference for
every field, its default, and its allowed values.

.. code-block:: python

    import gymnasium as gym
    from f1tenth_gym.envs.env_config import EnvConfig

    env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig())
    obs, info = env.reset(seed=42)

Passing anything other than an ``EnvConfig`` instance (a dict, ``None``, keyword
args) raises ``TypeError`` — there is no fallback config path.

The frozen-dataclass philosophy
--------------------------------

``EnvConfig`` and all of its nested sections are **frozen dataclasses**. You
never mutate a config in place; instead you derive a new one with
``with_updates(**changes)``, which returns a fresh, re-validated copy:

.. code-block:: python

    from f1tenth_gym.envs.env_config import EnvConfig

    base = EnvConfig()
    two_agents = base.with_updates(num_agents=2)   # base is untouched

Two properties follow from this design:

**Validation runs in ``__post_init__``.** Constructing (or ``with_updates``-ing)
a config immediately coerces types (e.g. ``seed``/``num_agents`` to ``int``,
``map_scale`` to ``float``) and raises on invalid combinations — so a bad config
fails at construction time, not deep inside ``step()``. Examples of enforced
rules: ``num_agents >= 1``; ``0 <= ego_index < num_agents``; ``map_scale > 0``;
each nested section must be an instance of its own class (not a dict).

**Nested mutation must nest.** ``with_updates`` only replaces top-level fields.
To change a field inside a nested section, rebuild that section too:

.. code-block:: python

    from f1tenth_gym.envs.env_config import EnvConfig

    cfg = EnvConfig()
    cfg = cfg.with_updates(
        simulation_config=cfg.simulation_config.with_updates(max_laps=None),
    )

.. note::

   The default ``max_laps=1`` ends the episode after a **single lap**
   (``terminated=True``). For endless rollouts (RL training, long evaluations)
   set ``SimulationConfig(max_laps=None)`` as shown above.

Reconfiguring a live environment
--------------------------------

You can swap the entire config on an already-constructed env with
``env.unwrapped.configure(new_cfg)``. This rebuilds the affected components
(track, simulator, spaces, reset function, renderer) to match the new config:

.. code-block:: python

    import gymnasium as gym
    from f1tenth_gym.envs.env_config import EnvConfig

    env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig())
    env.reset(seed=0)

    new_cfg = env.unwrapped.env_config.with_updates(
        params=env.unwrapped.env_config.params.with_updates(mu=1.0),
    )
    env.unwrapped.configure(new_cfg)
    env.reset(seed=0)   # reset after reconfiguring

Top-level ``EnvConfig`` fields
------------------------------

``EnvConfig`` has 17 fields. The eight nested-config fields are documented in
their own subsections below.

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Field
     - Default
     - Meaning
   * - ``seed``
     - ``12345``
     - Base random seed (coerced to ``int``). See :doc:`reproducibility` — note
       this seed feeds the sim's noise RNGs; ``reset(seed=...)`` controls the
       spawn/gymnasium RNG.
   * - ``map_name``
     - ``"Spielberg"``
     - Track name (auto-downloaded), a path, or a ``Track`` instance. See
       :doc:`tracks`.
   * - ``map_scale``
     - ``1.0``
     - Scale factor for the map (must be ``> 0``).
   * - ``params``
     - ``F1TENTH_VEHICLE_PARAMETERS``
     - A ``VehicleParameters`` instance (must be one, or ``TypeError``). See
       :doc:`dynamics`.
   * - ``num_agents``
     - ``1``
     - Number of agents / rows (must be ``>= 1``).
   * - ``ego_index``
     - ``0``
     - Index of the ego agent (must satisfy ``0 <= ego_index < num_agents``).
   * - ``control_config``
     - ``ControlConfig()``
     - Action interpretation, delays, actuator noise. See below.
   * - ``simulation_config``
     - ``SimulationConfig()``
     - Physics timestep, integrator, dynamics model, lap counting. See below.
   * - ``observation_config``
     - ``ObservationConfig()``
     - Observation type / field subset. See below and :doc:`observations`.
   * - ``reset_config``
     - ``ResetConfig()``
     - Spawn strategy and spacing. See below.
   * - ``lidar_config``
     - ``LiDARConfig()``
     - LiDAR geometry and noise. See below.
   * - ``render_config``
     - ``RenderConfig()``
     - Rendering pacing / frame output. See below and :doc:`rendering`.
   * - ``termination_config``
     - ``TerminationConfig()``
     - Termination / truncation rules. See below.
   * - ``reward_config``
     - ``RewardConfig()``
     - Per-step reward. See below and :doc:`rewards_and_rl`.
   * - ``domain_randomization_config``
     - ``DomainRandomizationConfig()``
     - Per-episode vehicle-param randomization. See below and
       :doc:`rewards_and_rl`.
   * - ``collision_check``
     - ``CollisionCheckMode.LIDAR_SCAN``
     - Agent-vs-agent collision mode: ``LIDAR_SCAN`` (default), ``BOUNDING_BOX``
       (O(n²) GJK), or ``NONE`` (disable collision detection). See
       :doc:`observations`.
   * - ``render_enabled``
     - ``True``
     - Whether a renderer is constructed (coerced to ``bool``).

All the enum defaults come from ``f1tenth_gym.envs``; import them from the
modules referenced below (e.g. ``from f1tenth_gym.envs.collision_models import
CollisionCheckMode``).

``ControlConfig``
-----------------

How your action array is interpreted, plus actuation-realism knobs. Import from
``f1tenth_gym.envs.env_config``; the enums come from
``f1tenth_gym.envs.action``.

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - Field
     - Default
     - Allowed values / notes
   * - ``longitudinal_mode``
     - ``LongitudinalActionType.SPEED``
     - ``SPEED`` (action column 1 is a target speed, tracked by a PID) or
       ``ACCL`` (column 1 is a raw acceleration).
   * - ``steering_mode``
     - ``SteerActionType.STEERING_ANGLE``
     - ``STEERING_ANGLE`` (column 0 is a target steering angle) or
       ``STEERING_SPEED`` (column 0 is a steering angular velocity).
   * - ``steer_delay_steps``
     - ``0``
     - Ring-buffer lag (in steps) on the steering command; must be ``>= 0``.
   * - ``throttle_delay_steps``
     - ``0``
     - Ring-buffer lag on the longitudinal command; must be ``>= 0``.
   * - ``steer_noise_std``
     - ``0.0``
     - Std of Gaussian noise added to the steering command each step; ``>= 0``.
   * - ``accl_noise_std``
     - ``0.0``
     - Std of Gaussian noise added to the longitudinal command each step;
       ``>= 0``.

The mode fields set the **action-space bounds and columns**. See :doc:`actions`
for the exact bounds. All four realism knobs default to 0, so the default
control config is byte-identical to a noiseless, lag-free actuator; the sim2real
semantics are covered in :doc:`rewards_and_rl`.

.. warning::

   The action array is always ``shape=(num_agents, 2)`` with columns
   ``[steering, longitudinal]`` — **steering is column 0**. Both columns are
   ``float32``, so swapping them fails silently. Single agent:
   ``np.array([[steer, speed]], dtype=np.float32)``.

``SimulationConfig``
--------------------

The physics loop. Enums: ``IntegratorType`` from ``f1tenth_gym.envs.integrators``,
``DynamicModel`` from ``f1tenth_gym.envs.dynamic_models``, ``LoopCounterMode``
from ``f1tenth_gym.envs.env_config``.

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - Field
     - Default
     - Allowed values / notes
   * - ``timestep``
     - ``0.01``
     - Sim timestep in seconds; must be ``> 0``.
   * - ``integrator_timestep``
     - ``0.01``
     - Integration substep (can be smaller than ``timestep``); must be ``> 0``.
   * - ``integrator``
     - ``IntegratorType.RK4``
     - ``RK4`` or ``EULER``.
   * - ``dynamics_model``
     - ``DynamicModel.ST``
     - ``KS`` (kinematic single-track, 5-state) or ``ST`` (single-track,
       7-state). ``MB`` exists but is broken — see :doc:`dynamics`.
   * - ``loop_counter``
     - ``LoopCounterMode.FRENET_BASED``
     - ``FRENET_BASED`` (default) or ``WINDING_ANGLE``. ``TOGGLE`` is declared
       but **not implemented** (counts zero laps).
   * - ``compute_frenet_frame``
     - ``True``
     - Whether to compute Frenet ``(s, ey, ephi)`` coordinates each step.
   * - ``max_laps``
     - ``1``
     - Laps before the episode terminates; ``None`` = infinite. If set, must be
       ``>= 1``.

.. note::

   ``FRENET_BASED`` lap counting requires the Frenet frame. ``with_updates``
   and ``EnvConfig.__post_init__`` **auto-enable** ``compute_frenet_frame=True``
   when ``loop_counter is FRENET_BASED`` — you cannot accidentally disable it
   while keeping Frenet lap counting.

.. warning::

   ``timestep`` must be an exact multiple of ``integrator_timestep`` and the
   check uses IEEE-754 modulo. ``timestep=0.03, integrator_timestep=0.01`` is
   **rejected** (``0.03 % 0.01`` is not exactly 0). Use values like
   ``0.01 / 0.02 / 0.04 / 0.05 / 0.1``.

``ObservationConfig``
---------------------

What ``obs`` looks like. ``ObservationType`` from
``f1tenth_gym.envs.observation``.

.. list-table::
   :header-rows: 1
   :widths: 24 26 50

   * - Field
     - Default
     - Allowed values / notes
   * - ``type``
     - ``ObservationType.DIRECT``
     - ``DIRECT`` / ``ORIGINAL`` (alias of DIRECT) / ``FEATURES`` /
       ``KINEMATIC_STATE`` / ``DYNAMIC_STATE`` / ``FRENET_DYNAMIC_STATE``.
   * - ``features``
     - ``None``
     - A tuple of field names — **only valid with** ``type=FEATURES``.

.. warning::

   Setting ``features`` with any ``type`` other than ``FEATURES`` raises
   ``ValueError`` (it used to be silently ignored). If you want a custom field
   subset, use ``ObservationConfig(type=ObservationType.FEATURES,
   features=(...))``.

See :doc:`observations` for the exact field vocabulary of each type (and the
trap that ``DIRECT`` does **not** contain ``pose_x`` — use ``KINEMATIC_STATE``).

``ResetConfig``
---------------

Where agents spawn at ``reset()``. ``ResetStrategy`` from
``f1tenth_gym.envs.reset``.

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Field
     - Default
     - Allowed values / notes
   * - ``strategy``
     - ``ResetStrategy.RL_GRID_STATIC``
     - ``RL_GRID_STATIC`` / ``RL_RANDOM_STATIC`` / ``RL_GRID_RANDOM`` /
       ``RL_RANDOM_RANDOM`` / ``MAP_RANDOM_STATIC``.
   * - ``min_dist``
     - ``None``
     - Minimum spawn spacing (m). ``None`` = strategy default (1.5 for RL, 0.5
       for map). Must be ``>= 0``.
   * - ``max_dist``
     - ``None``
     - Maximum spawn spacing (m). ``None`` = strategy default (2.5 for RL, 1.0
       for map). Must be ``> 0``.
   * - ``shuffle``
     - ``None``
     - Override whether spawn order is shuffled. ``None`` = strategy default
       (STATIC strategies don't shuffle).
   * - ``move_laterally``
     - ``None``
     - Override whether spawns are offset laterally off the reference line.
       ``None`` = strategy default.

Validation: if both ``min_dist`` and ``max_dist`` are set, ``min_dist`` must be
``< max_dist``. Non-``None`` fields are forwarded to the reset-function
builder via ``reset_kwargs()``.

.. note::

   All ``RL_*`` strategies bind to the **raceline** (never the centerline) and
   default to ``move_laterally=False``, so a multi-agent "grid" reset places
   every car *on* the raceline, separated only longitudinally. Because the
   Frenet frame is measured against the centerline, ``obs[...]["frenet_pose"]``
   ``ey`` is generally non-zero at spawn.

``LiDARConfig``
---------------

The simulated LiDAR. Import from ``f1tenth_gym.envs.lidar`` (or
``f1tenth_gym.envs.lidar.config``). All angular fields are in **radians**.

.. list-table::
   :header-rows: 1
   :widths: 26 24 50

   * - Field
     - Default
     - Allowed values / notes
   * - ``enabled``
     - ``True``
     - When ``False``, ``scan`` has shape ``(0,)`` and collision detection
       adapts (see :doc:`observations`).
   * - ``num_beams``
     - ``1080``
     - Number of beams; must be ``>= 1``. (Angular resolution is internally
       capped by a 2000-entry LUT — raising beams past ~2000 buys no
       resolution.)
   * - ``field_of_view``
     - ``4.712389`` (270°)
     - Total FOV in radians; must be ``> 0``. Only used to derive
       ``angle_min``/``angle_max`` **when those are ``None``** — see warning.
   * - ``angle_min``
     - ``None`` → ``-field_of_view/2``
     - Scan start angle. If given, must be ``>= -π``.
   * - ``angle_max``
     - ``None`` → ``+field_of_view/2``
     - Scan end angle. If given, must be ``<= π`` and ``> angle_min``.
   * - ``range_min``
     - ``0.0``
     - Minimum range (m); readings below are clipped. Must be ``>= 0`` and
       ``< range_max``.
   * - ``range_max``
     - ``30.0``
     - Maximum range (m); readings above are clipped. Must be ``> 0``.
   * - ``noise_std``
     - ``0.01``
     - Std of per-reading Gaussian range noise; ``>= 0``.
   * - ``dropout_prob``
     - ``0.0``
     - Per-beam, per-step no-return probability (clamped to ``range_max``);
       must be in ``[0, 1]``.
   * - ``range_bias_std``
     - ``0.0``
     - Std of a per-beam systematic bias drawn **once per episode**
       (reproducible with ``reset(seed=...)``); ``>= 0``.
   * - ``base_link_to_lidar_tf``
     - ``(0.275, 0.0, 0.0)``
     - ``(x, y, yaw)`` offset of the LiDAR from ``base_link``, in
       metres/radians.

``noise_std``, ``dropout_prob`` and ``range_bias_std`` affect only the
**observed** scan — collision detection always uses the clean scan. The noise
semantics are covered in :doc:`rewards_and_rl`. Convenience read-only
properties: ``angle_increment`` and ``maximum_range``.

.. warning::

   ``lidar_config.with_updates(field_of_view=X)`` is a **silent no-op** on the
   actual FOV. ``__post_init__`` only derives ``angle_min``/``angle_max`` from
   ``field_of_view`` when they are ``None``; a ``with_updates`` carries the
   already-materialised angles over, and the scanner computes the effective FOV
   as ``angle_max - angle_min``. So ``with_updates(field_of_view=2*np.pi)``
   leaves ``angle_min``/``angle_max`` at ±2.356 (270°). To change the FOV,
   build a **fresh** ``LiDARConfig(field_of_view=...)``, or set ``angle_min`` and
   ``angle_max`` explicitly:

   .. code-block:: python

       import numpy as np
       from f1tenth_gym.envs.lidar import LiDARConfig

       # Correct: fresh config derives the angles
       lidar = LiDARConfig(field_of_view=np.deg2rad(180))
       # Or set the angles directly
       lidar = LiDARConfig(angle_min=-np.pi / 2, angle_max=np.pi / 2)

``RenderConfig``
----------------

Rendering pacing and frame output. See :doc:`rendering` for the full model.

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - Field
     - Default
     - Notes
   * - ``render_fps``
     - ``60``
     - Target fixed frame rate; caps human-mode redraws and sets the rgb_array
       distinct-frame cadence in sim time. Coerced to ``int``, must be ``> 0``.
   * - ``real_time_factor``
     - ``1.0``
     - Sim-seconds per wall-second in human modes; ``float("inf")`` = free-run.
       Coerced to ``float``, must be ``> 0``. Togglable at runtime via
       ``env.unwrapped.set_real_time_factor(x)``.
   * - ``window_size``
     - ``800``
     - Square render / rgb_array / video size in pixels. Coerced to ``int``,
       ``> 0``.
   * - ``focus_on``
     - ``"agent_0"``
     - Agent id the camera follows; ``None`` = whole-map view.
   * - ``vehicle_palette``
     - 10 hex colours
     - Per-agent car colours, cycled by index.
   * - ``show_wheels``
     - ``True``
     - Draw wheels on each car.
   * - ``render_map_img``
     - ``True``
     - Draw the occupancy-map image under the scene.
   * - ``car_thickness``
     - ``1``
     - Car outline thickness in pixels (coerced to ``int``).
   * - ``bigger_car_when_map_centered``
     - ``True``
     - Scale cars up in the zoomed-out map view.
   * - ``show_lap_info``
     - ``True``
     - Overlay the lap-time / lap-count label.

``TerminationConfig``
---------------------

When the episode ends. Both ``terminated`` and ``truncated`` are driven here.

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Field
     - Default
     - Allowed values / notes
   * - ``max_episode_steps``
     - ``None``
     - If set, ``truncated=True`` once this many steps elapse since reset
       (a config-level TimeLimit). ``None`` = no limit; must be ``>= 1`` if set.
   * - ``terminate_on_collision``
     - ``True``
     - Whether a collision sets ``terminated=True``.
   * - ``collision_agents``
     - ``"ego"``
     - Whose collision counts: ``"ego"`` (only the ego agent) or ``"any"``
       (any agent). Any other value raises ``ValueError``.

.. note::

   ``truncated`` is no longer hardcoded ``False``: with
   ``max_episode_steps=None`` and ``max_laps=None`` an episode can still run
   forever, but setting ``max_episode_steps`` gives you a truncation limit
   without a gymnasium ``TimeLimit`` wrapper. Per-agent collision flags are
   exposed in ``info["collisions"]``.

``RewardConfig``
----------------

The per-step scalar reward. ``RewardMode`` from ``f1tenth_gym.envs.env_config``.
Full semantics in :doc:`rewards_and_rl`; the fields:

.. list-table::
   :header-rows: 1
   :widths: 26 24 50

   * - Field
     - Default
     - Allowed values / notes
   * - ``mode``
     - ``RewardMode.SURVIVAL``
     - ``SURVIVAL`` (``reward = timestep``, the historical default), ``PROGRESS``
       (weighted Frenet progress + speed + survival − collision), or
       ``CUSTOM``.
   * - ``progress_weight``
     - ``1.0``
     - Reward per metre of forward arclength (PROGRESS).
   * - ``velocity_weight``
     - ``0.0``
     - Reward per m/s of ego speed (PROGRESS).
   * - ``timestep_weight``
     - ``0.0``
     - Survival bonus per step, scaled by timestep (PROGRESS).
   * - ``collision_penalty``
     - ``0.0``
     - Subtracted when the ego is colliding (PROGRESS); must be ``>= 0``.
   * - ``reward_fn``
     - ``None``
     - For ``CUSTOM``: a callable
       ``(obs, action, info, terminated, truncated) -> float``.

Validation: ``RewardMode.CUSTOM`` **requires** ``reward_fn`` (else
``ValueError``); a non-callable ``reward_fn`` raises ``TypeError``.
``RewardMode.PROGRESS`` requires the Frenet frame — ``EnvConfig.__post_init__``
raises if ``compute_frenet_frame`` is ``False``. Regardless of mode, ``info``
always carries the raw ``progress`` and ``collisions`` signals.

``DomainRandomizationConfig``
-----------------------------

Per-episode randomization of vehicle parameters, sampled at each ``reset()`` from
the env RNG (reproducible with ``reset(seed=...)``). Full semantics in
:doc:`rewards_and_rl`.

.. list-table::
   :header-rows: 1
   :widths: 26 24 50

   * - Field
     - Default
     - Allowed values / notes
   * - ``enabled``
     - ``False``
     - Whether to randomize at each reset.
   * - ``param_ranges``
     - ``{}``
     - ``{param_name: (low, high)}`` in **absolute physical units**. Each name
       must be an actual ``VehicleParameters`` field, and ``low <= high``, or
       ``ValueError``.

.. code-block:: python

    from f1tenth_gym.envs.env_config import EnvConfig, DomainRandomizationConfig

    cfg = EnvConfig().with_updates(
        domain_randomization_config=DomainRandomizationConfig(
            enabled=True,
            param_ranges={"m": (3.0, 4.0), "mu": (0.9, 1.1)},
        ),
    )

.. note::

   Field names are the raw ``VehicleParameters`` fields — e.g. ``m`` (mass) and
   ``h`` (CoG height), **not** ``mass``/``h_cg``. Prefer randomizing dynamics
   params (``m``, ``mu``, ``lf``, ``lr``, ``I``, ``h``); randomizing actuation
   limits (``v_max``, ``s_max``, …) desyncs the fixed action/observation spaces.

A worked example
----------------

Putting the nested pattern together — a headless, endless, two-agent training
config with a progress reward and mild actuator noise:

.. code-block:: python

    import gymnasium as gym
    from f1tenth_gym.envs.env_config import (
        EnvConfig, SimulationConfig, ObservationConfig,
        ControlConfig, RewardConfig, RewardMode,
    )
    from f1tenth_gym.envs.observation import ObservationType

    cfg = EnvConfig(
        num_agents=2,
        render_enabled=False,
        simulation_config=SimulationConfig(max_laps=None),
        observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
        control_config=ControlConfig(steer_noise_std=0.01, accl_noise_std=0.05),
        reward_config=RewardConfig(mode=RewardMode.PROGRESS, velocity_weight=0.1),
    )

    env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
    obs, info = env.reset(seed=0)
    env.close()

See also
--------

- :doc:`quickstart` — the minimal loop.
- :doc:`actions` and :doc:`observations` — how ``ControlConfig`` /
  ``ObservationConfig`` shape the spaces.
- :doc:`dynamics` — ``params`` and the dynamics models.
- :doc:`tracks` — ``map_name`` loading and racelines.
- :doc:`rendering` — the ``RenderConfig`` pacing model.
- :doc:`rewards_and_rl` — reward, domain randomization, and sim2real noise.
- :doc:`reproducibility` — how ``seed`` and ``reset(seed=...)`` interact.
