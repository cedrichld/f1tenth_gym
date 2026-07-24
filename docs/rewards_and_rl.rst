Rewards, RL & sim2real
======================

This page covers everything you need to drive the F1TENTH gym from a
reinforcement-learning training loop: the pluggable reward surface
(:class:`~f1tenth_gym.envs.env_config.RewardConfig`), per-episode domain
randomization, the sim2real actuation- and sensor-noise knobs, and the thin
Gymnasium wrappers that expose a flat single-agent interface.

Scope boundary
--------------

The simulator is **simulation-only, with clean interfaces**. It deliberately
ships *no* planners and *no* training loops:

- **Planners** (pure pursuit, MPC, gap-follow, ...) live in a separate
  ``f1tenth_planning`` repository. The one in ``examples/waypoint_follow.py``
  is a demo, not a supported API.
- **RL training loops** (PPO, SAC, replay buffers, ...) live in a separate
  ``f1tenth_learning`` repository.

What lives *here* is the reward/observation surface and the sim2real knobs an
RL user needs, plus a couple of thin Gymnasium adapters. Everything below is
configured through the frozen :class:`~f1tenth_gym.envs.env_config.EnvConfig`
tree (see :doc:`configuration`) — there is no dict or YAML config.

Rewards
-------

By default the environment returns a pure survival reward
(``reward = timestep = 0.01`` per step). You select and shape the reward with
:class:`~f1tenth_gym.envs.env_config.RewardConfig`, whose ``mode`` is a
:class:`~f1tenth_gym.envs.env_config.RewardMode`.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``RewardMode``
     - Reward returned by ``step``
   * - ``SURVIVAL`` (=0, default)
     - ``timestep`` — the historical time-alive reward (0.01 per step at the
       default ``timestep=0.01``).
   * - ``PROGRESS`` (=1)
     - Weighted sum of forward Frenet arclength progress, ego speed, a survival
       bonus, and a collision penalty. Requires the Frenet frame.
   * - ``CUSTOM`` (=2)
     - ``reward_fn(obs, action, info, terminated, truncated) -> float`` — you
       supply the callable.

Raw signals always in ``info``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regardless of the mode, every ``step`` puts the raw per-agent reward signals in
the ``info`` dict, so external code can compute any reward it likes:

- ``info["progress"]`` — per-agent forward Frenet arclength this step, in
  metres (wrap-corrected; zeros when the Frenet frame is not computed).
- ``info["collisions"]`` — a per-agent copy of the collision flags
  (``1.0`` = colliding).

``info`` also carries ``lap_times``, ``lap_counts``, and ``sim_time`` (all
copies of the env's live arrays, so a stored ``info`` will not mutate
retroactively).

SURVIVAL (default)
~~~~~~~~~~~~~~~~~~~

No configuration needed — the default
:class:`~f1tenth_gym.envs.env_config.EnvConfig` already uses
``RewardMode.SURVIVAL``. Each surviving step returns
``simulation_config.timestep``.

PROGRESS
~~~~~~~~

``PROGRESS`` builds the ego reward from four terms:

.. code-block:: text

   reward =  progress_weight  * (forward Frenet Δs this step, metres)
           + velocity_weight  * (ego speed, m/s)
           + timestep_weight  * timestep
           - collision_penalty        (only on steps where the ego is colliding)

The weights are fields on
:class:`~f1tenth_gym.envs.env_config.RewardConfig`
(``progress_weight=1.0``, ``velocity_weight=0.0``, ``timestep_weight=0.0``,
``collision_penalty=0.0`` by default; ``collision_penalty`` must be ``>= 0``).
The progress term comes from the same per-agent, wrap-corrected Frenet Δs that
is exposed as ``info["progress"]`` — it is seeded from the spawn arclength at
reset (so the first step's progress is ~0, not the whole spawn ``s``) and is
**independent of the lap counter**, so it works with ``max_laps=None`` and any
lap-counting mode.

.. warning::

   ``PROGRESS`` needs the Frenet frame.
   :meth:`EnvConfig.__post_init__ <f1tenth_gym.envs.env_config.EnvConfig>`
   raises ``ValueError`` if ``reward_config.mode`` is ``PROGRESS`` while
   ``simulation_config.compute_frenet_frame`` is ``False``. The default config
   already computes the Frenet frame, so this only bites if you turned it off.

.. code-block:: python

   from f1tenth_gym.envs.env_config import EnvConfig, RewardConfig, RewardMode

   cfg = EnvConfig(
       reward_config=RewardConfig(
           mode=RewardMode.PROGRESS,
           progress_weight=1.0,
           velocity_weight=0.1,
           timestep_weight=0.0,
           collision_penalty=10.0,
       ),
   )

The Frenet frame here is always the **centerline** frame (there is no raceline
option), so ``progress`` measures arclength along the centerline; see
:doc:`tracks` for the frame details.

CUSTOM
~~~~~~

Supply your own scalar reward. The callable is invoked *after* ``info`` is
fully populated for the step, so it can read ``info["progress"]``,
``info["collisions"]``, ``info["lap_counts"]``, etc.

.. code-block:: python

   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, RewardConfig, RewardMode

   def my_reward(obs, action, info, terminated, truncated):
       # obs is the multi-agent dict; each scalar field is a 0-d float32 ndarray
       speed = float(obs["agent_0"]["linear_vel_x"])
       progress = float(info["progress"][0])
       crashed = bool(info["collisions"][0])
       return progress + 0.05 * speed - (100.0 if crashed else 0.0)

   cfg = EnvConfig(
       reward_config=RewardConfig(mode=RewardMode.CUSTOM, reward_fn=my_reward),
   )

.. note::

   ``RewardMode.CUSTOM`` requires ``reward_fn`` to be set (else ``ValueError``
   at construction), and ``reward_fn`` must be callable. Note that
   ``linear_vel_x`` is a *derived* observation field — it is only present under
   an observation type such as ``KINEMATIC_STATE`` (see :doc:`observations`);
   under the default ``DIRECT`` type, read the ``std_state`` field instead.

Domain randomization
--------------------

:class:`~f1tenth_gym.envs.env_config.DomainRandomizationConfig` randomizes
vehicle parameters **once per episode**, at ``reset()``. It is disabled by
default (``enabled=False``).

``param_ranges`` maps a :class:`~f1tenth_gym.envs.dynamic_models.VehicleParameters`
**field name** to an **absolute** ``(low, high)`` range in physical units. Each
listed parameter is drawn uniformly from that range at every reset using the
env RNG, so the draws are **reproducible with** ``reset(seed=...)``. Only the
parameters you list are randomized; everything else keeps its configured value.

.. code-block:: python

   from f1tenth_gym.envs.env_config import EnvConfig, DomainRandomizationConfig

   cfg = EnvConfig(
       domain_randomization_config=DomainRandomizationConfig(
           enabled=True,
           param_ranges={
               "m":  (3.0, 4.0),    # mass, kg
               "mu": (0.9, 1.1),    # tyre friction coefficient
               "lf": (0.14, 0.18),  # CoG-to-front-axle distance, m
           },
       ),
   )

.. note::

   Field names are the **actual** ``VehicleParameters`` fields — e.g. ``m``
   (mass) and ``h`` (CoG height), not ``mass``/``h_cg``. Unknown names, or a
   range with ``low > high``, raise ``ValueError`` at construction. Prefer
   randomizing *dynamics* parameters (``m``, ``mu``, ``lf``, ``lr``, ``I``,
   ``h``).

.. warning::

   Randomizing the **actuation limits** (``v_min``/``v_max``/``s_min``/
   ``s_max``/...) is allowed but desyncs the fixed action and observation
   spaces from the live vehicle, so it is not recommended.

Under the hood, ``reset()`` samples a new
:class:`~f1tenth_gym.envs.dynamic_models.VehicleParameters` from the env RNG and
applies it to the simulator *before* re-initialising the agent states. The
ground truth the physics kernels use is the flat ``sim.params_array`` — that is
what changes each episode.

Sim2real: actuation realism
---------------------------

:class:`~f1tenth_gym.envs.env_config.ControlConfig` carries three actuation
imperfection knobs, applied inside ``sim.step``. **All default to 0, so the
simulation is byte-identical to before unless you set them.**

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Effect
   * - ``steer_noise_std``
     - Std of Gaussian noise added to the commanded steering input each step
       (servo/actuator noise).
   * - ``accl_noise_std``
     - Std of Gaussian noise added to the commanded longitudinal input each
       step.
   * - ``throttle_delay_steps``
     - Ring-buffer lag (in steps) on the longitudinal command, modelling
       drivetrain/ESC lag. Mirrors the existing ``steer_delay_steps``.

.. code-block:: python

   from f1tenth_gym.envs.env_config import EnvConfig, ControlConfig

   cfg = EnvConfig(
       control_config=ControlConfig(
           steer_noise_std=0.02,     # rad
           accl_noise_std=0.1,       # command units
           throttle_delay_steps=2,   # steps of longitudinal lag
       ),
   )

The command noise is drawn from a dedicated control RNG that is reseeded off the
reset seed, so it is reproducible per seed (see :doc:`reproducibility`). Noise
is applied *before* the delay buffers.

Sim2real: richer LiDAR noise
----------------------------

Beyond the plain ``noise_std``,
:class:`~f1tenth_gym.envs.lidar.LiDARConfig` adds two sensor-realism knobs
(both default 0):

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Effect
   * - ``dropout_prob``
     - Per-beam, per-step probability that a beam returns a no-return (clamped
       to ``range_max``), modelling missed detections. Must be in ``[0, 1]``.
   * - ``range_bias_std``
     - Std of a per-beam **systematic** range bias, drawn **once per episode**
       (reproducible with ``reset(seed=...)``) and held constant across the
       rollout — models calibration error, unlike the per-step ``noise_std``.

.. code-block:: python

   from f1tenth_gym.envs.env_config import EnvConfig
   from f1tenth_gym.envs.lidar import LiDARConfig

   cfg = EnvConfig(
       lidar_config=LiDARConfig(
           num_beams=1080,
           noise_std=0.01,
           dropout_prob=0.02,     # 2% of beams drop each step
           range_bias_std=0.03,   # per-beam calibration bias, fixed per episode
       ),
   )

.. warning::

   All LiDAR noise knobs (``noise_std``, ``dropout_prob``, ``range_bias_std``)
   affect the **observed** scan only. Collision detection uses the clean,
   noise-free scan, so adding sensor noise never changes when a crash fires.

.. note::

   Setting ``field_of_view`` on an *existing* config via ``with_updates`` does
   **not** change the actual FOV (``angle_min``/``angle_max`` are already
   materialised). Pass ``field_of_view`` to a fresh ``LiDARConfig(...)``, or set
   ``angle_min``/``angle_max`` explicitly. See :doc:`observations`.

Wrappers
--------

Two thin Gymnasium adapters in ``f1tenth_gym.envs.wrappers`` bridge the native
multi-agent ``dict[agent -> dict[field -> ndarray]]`` observation to
single-agent RL code. Both record their constructor args, so they pickle
cleanly (usable with ``VecEnv`` / subprocess workers).

SingleAgentWrapper
~~~~~~~~~~~~~~~~~~~

:class:`~f1tenth_gym.envs.wrappers.SingleAgentWrapper` unwraps
``obs["agent_0"]`` to the observation and reshapes the action from ``(1, 2)`` to
``(2,)``, so a 1-agent env is directly consumable by single-agent RL libraries.

- **Requires** ``num_agents == 1`` (else ``ValueError``).
- Reward is already scalar; ``info`` passes through unchanged (its per-agent
  arrays have length 1).
- Compose with ``gymnasium.wrappers.FlattenObservation`` to get a flat, finite
  ``Box`` observation that passes ``gymnasium.utils.env_checker.check_env`` —
  the observation-space bounds are physical and finite (see
  :doc:`observations`), so the flattened space normalises correctly.

ObservationDelayWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~f1tenth_gym.envs.wrappers.ObservationDelayWrapper` returns the
observation as it was ``delay_steps`` steps ago (sensor/perception lag) while
reward, termination, and ``info`` still reflect the true current state. Before
that much history exists, it repeats the reset observation. It deep-copies
frames (no aliasing), works on both the native nested dict and a flattened
``Box``, and ``delay_steps=0`` is a pure passthrough.

.. code-block:: python

   import gymnasium as gym
   from f1tenth_gym.envs.wrappers import ObservationDelayWrapper

   env = ObservationDelayWrapper(env, delay_steps=3)  # obs is 3 steps stale

Full example: single-agent PROGRESS + DR
----------------------------------------

A complete, runnable rollout combining a flat single-agent interface, the
``PROGRESS`` reward, and a small domain-randomization dict. The seed makes both
the vehicle-parameter draws and the reset pose reproducible (see
:doc:`reproducibility`).

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from gymnasium.wrappers import FlattenObservation

   import f1tenth_gym  # registers f1tenth_gym:f1tenth-v0
   from f1tenth_gym.envs.env_config import (
       EnvConfig, SimulationConfig, ObservationConfig,
       RewardConfig, RewardMode, DomainRandomizationConfig,
   )
   from f1tenth_gym.envs.observation import ObservationType
   from f1tenth_gym.envs.wrappers import SingleAgentWrapper

   cfg = EnvConfig(
       num_agents=1,
       observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
       simulation_config=SimulationConfig(
           max_laps=None,             # endless episode (default ends after 1 lap)
           compute_frenet_frame=True, # required by PROGRESS
       ),
       reward_config=RewardConfig(
           mode=RewardMode.PROGRESS,
           progress_weight=1.0,
           velocity_weight=0.1,
           collision_penalty=10.0,
       ),
       domain_randomization_config=DomainRandomizationConfig(
           enabled=True,
           param_ranges={"m": (3.0, 4.0), "mu": (0.9, 1.1)},
       ),
       render_enabled=False,
   )

   # Native multi-agent env -> flat single-agent Box, ready for SB3/CleanRL.
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   env = SingleAgentWrapper(env)
   env = FlattenObservation(env)

   obs, info = env.reset(seed=42)     # reproducible DR draw + spawn pose
   total_reward = 0.0
   for _ in range(200):
       action = np.array([0.0, 2.0], dtype=np.float32)  # [steer, speed], flat (2,)
       obs, reward, terminated, truncated, info = env.step(action)
       total_reward += reward
       if terminated or truncated:
           break

   print("return:", total_reward, "progress:", float(info["progress"][0]))
   env.close()

.. note::

   After ``SingleAgentWrapper``, the action is a flat ``(2,)`` array with
   ``[steering, longitudinal]`` — **steering is still column 0**. The wrapper
   reshapes it back to ``(1, 2)`` internally. See :doc:`actions`.

Reproducibility
---------------

Every stochastic sim2real feature on this page is driven by the environment RNG
and is reproducible by passing a seed to ``reset``:

- Domain-randomization parameter draws (per episode).
- Actuation command noise (``steer_noise_std`` / ``accl_noise_std``).
- LiDAR ``dropout_prob`` (per step) and ``range_bias_std`` (per episode).

Call ``env.reset(seed=...)`` to fix the whole stream. For the full picture of
which RNGs each seed reaches — and why ``EnvConfig.seed`` alone is not enough —
see :doc:`reproducibility`.
