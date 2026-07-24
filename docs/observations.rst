Observations
============

The environment returns observations as a **nested dictionary**, one entry per
agent, whose contents depend on the :class:`~f1tenth_gym.envs.observation.ObservationType`
preset you select in the config. This page documents the six presets, the
field vocabulary they draw from, and the (finite) observation space they
declare.

For how to *reconfigure* the observation type on a live env, see
:doc:`configuration`; for flattening the observation into a normalisable
``Box`` suitable for RL, see :doc:`rewards_and_rl`.

Structure
---------

An observation is a ``dict[agent_id -> dict[field -> ndarray]]``. The agent
keys are the strings ``"agent_0"``, ``"agent_1"``, ... one per agent
(``num_agents`` of them). Each agent's value is itself a dict mapping field
names to numpy arrays.

**Every scalar field is a 0-d ``float32`` ndarray, not a Python ``float``.**
Vector fields (``scan``, ``std_state``, ``state``, ``frenet_pose``) are 1-d
``float32`` arrays. This is deliberate and load-bearing — it keeps numba
kernels in downstream code (e.g. the pure-pursuit example) type-consistent.

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig
   from f1tenth_gym.envs.observation import ObservationType

   cfg = EnvConfig(
       observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
       render_enabled=False,
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   obs, info = env.reset(seed=42)

   for _ in range(50):
       action = np.array([[0.0, 2.0]], dtype=np.float32)   # [[steer, speed]]
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break

   agent = obs["agent_0"]
   print("pose_x =", float(agent["pose_x"]))          # 0-d float32 ndarray
   print("linear_vel_x =", float(agent["linear_vel_x"]))
   env.close()

Observation types
-----------------

All six :class:`~f1tenth_gym.envs.observation.ObservationType` presets resolve
to the same underlying ``FullObservation`` provider, differing only in the
tuple of fields they expose:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Type
     - Fields exposed
   * - ``DIRECT`` (default)
     - The **base** fields: ``scan``, ``std_state``, ``state``, ``collision``,
       ``lap_time``, ``lap_count``, ``sim_time``, and ``frenet_pose``
       (the last only when the Frenet frame is computed — see the warning
       below).
   * - ``ORIGINAL``
     - Alias of ``DIRECT`` (kept for backwards compatibility).
   * - ``FEATURES``
     - A **custom** subset you specify via
       ``ObservationConfig(type=FEATURES, features=(...))``.
   * - ``KINEMATIC_STATE``
     - ``pose_x``, ``pose_y``, ``delta``, ``linear_vel_x``, ``pose_theta``.
   * - ``DYNAMIC_STATE``
     - ``pose_x``, ``pose_y``, ``delta``, ``linear_vel_magnitude``,
       ``pose_theta``, ``ang_vel_z``, ``beta``.
   * - ``FRENET_DYNAMIC_STATE``
     - ``pose_x``, ``pose_y``, ``delta``, ``linear_vel_x``, ``linear_vel_y``,
       ``pose_theta``, ``ang_vel_z``, ``beta``.

The default config uses ``DIRECT``.

.. warning::

   ``DIRECT`` does **not** contain ``pose_x`` — it is a *derived* field, not a
   base field. Under the default config ``obs["agent_0"]["pose_x"]`` raises
   ``KeyError``. To read the pose from ``DIRECT``, use ``std_state`` (indices
   0, 1, 4 are X, Y, yaw). To get named pose fields directly, use
   ``KINEMATIC_STATE`` / ``DYNAMIC_STATE`` / ``FRENET_DYNAMIC_STATE`` or a
   ``FEATURES`` tuple that includes them.

Field vocabulary
-----------------

Fields split into **base** fields (read straight from the simulator's
struct-of-arrays buffers) and **derived** fields (computed from
``std_state``).

Base fields
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Field
     - Shape
     - Meaning
   * - ``scan``
     - ``(num_beams,)``
     - LiDAR ranges in metres. Shape is ``(1080,)`` under the default LiDAR
       config. See the disabled-LiDAR warning below.
   * - ``std_state``
     - ``(7,)``
     - Standardised state ``[X, Y, steering_angle, speed, yaw, yaw_rate,
       beta]`` — always 7 wide regardless of dynamics model. Index 6 is the
       **slip angle** ``beta`` (not a lateral velocity). Under KS dynamics
       indices 5 and 6 (``yaw_rate``, ``beta``) are hardcoded ``0.0``.
   * - ``state``
     - ``(state_dim,)``
     - The raw model state. ``state_dim`` depends on the dynamics model:
       KS ``[x, y, delta, v, yaw]`` (5); ST (default)
       ``[x, y, delta, v, yaw, yaw_rate, beta]`` (7). See :doc:`dynamics`.
   * - ``collision``
     - ``()``
     - ``1.0`` if this agent is in collision this step, else ``0.0``.
   * - ``lap_time``
     - ``()``
     - Elapsed time on the current lap, in seconds.
   * - ``lap_count``
     - ``()``
     - Number of completed laps (monotonically non-decreasing).
   * - ``sim_time``
     - ``()``
     - Total simulated time since reset, in seconds.
   * - ``frenet_pose``
     - ``(3,)``
     - ``[s, ey, ephi]`` — arclength along the **centerline**, signed lateral
       deviation (``+ey`` is LEFT of travel), and heading error, all in
       metres / radians. Present only when the Frenet frame is computed.

Derived fields
~~~~~~~~~~~~~~

All derived fields are 0-d ``float32`` scalars, computed from ``std_state``
(with ``vx = speed·cos(beta)``, ``vy = speed·sin(beta)``):

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Meaning
   * - ``pose_x`` / ``pose_y``
     - Position X / Y (``std_state[0]`` / ``[1]``).
   * - ``pose_theta``
     - Yaw (``std_state[4]``), wrapped to ``[-pi, pi)``.
   * - ``linear_vel_x``
     - Longitudinal velocity ``speed·cos(beta)``.
   * - ``linear_vel_y``
     - Lateral velocity ``speed·sin(beta)``.
   * - ``linear_vel_magnitude``
     - Speed magnitude ``hypot(vx, vy)`` (non-negative, unlike the signed
       ``std_state[3]``).
   * - ``ang_vel_z``
     - Yaw rate (``std_state[5]``). Always ``0.0`` under KS.
   * - ``delta``
     - Steering angle (``std_state[2]``).
   * - ``beta``
     - Slip angle (``std_state[6]``). Always ``0.0`` under KS.

.. note::

   The pose frame is not uniform across dynamics models: under KS the pose
   refers to the **rear axle**, under ST to the **centre of gravity**.
   Switching model silently shifts the reported pose by ``lr``. See
   :doc:`dynamics`.

Custom feature subsets (FEATURES)
---------------------------------

Use ``ObservationType.FEATURES`` with an explicit ``features`` tuple to select
exactly the fields you want. Names must come from the base or derived
vocabulary above; an unknown name raises ``ValueError``.

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig
   from f1tenth_gym.envs.observation import ObservationType

   cfg = EnvConfig(
       observation_config=ObservationConfig(
           type=ObservationType.FEATURES,
           features=("pose_x", "pose_y", "pose_theta", "linear_vel_x", "scan"),
       ),
       render_enabled=False,
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   obs, info = env.reset(seed=42)
   print(sorted(obs["agent_0"].keys()))
   env.close()

.. note::

   ``features`` may only be set when ``type=FEATURES``; setting it alongside
   any other type raises at config construction. See :doc:`configuration`.

The observation space
----------------------

``env.observation_space`` is a nested ``gym.spaces.Dict`` — one ``Dict`` per
agent, whose entries are ``Box`` spaces for each selected field. **All bounds
are finite and roughly physical**, derived from the vehicle parameters and
track extents: velocities from ``v_min``/``v_max``, steering from
``s_min``/``s_max``, pose from the centerline bounding box plus a 5 m margin,
angles ``±pi``, yaw-rate from a kinematic estimate, and Frenet ``s`` from the
frame length.

Because the space is finite (not the old blanket ``±1e30``), it passes
gymnasium's ``check_env`` and can be flattened and normalised. Compose
``gymnasium.wrappers.FlattenObservation`` (typically after
``SingleAgentWrapper`` for a single-agent env) to obtain a flat, finite
``Box`` — see :doc:`rewards_and_rl`.

.. code-block:: python

   import gymnasium as gym
   from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig
   from f1tenth_gym.envs.observation import ObservationType

   cfg = EnvConfig(
       observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
       render_enabled=False,
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   print(env.observation_space["agent_0"])   # Dict of finite Box spaces
   env.close()

Gotchas
-------

.. warning::

   ``frenet_pose`` exists only when the environment computes the Frenet frame
   (``SimulationConfig(compute_frenet_frame=True)``, the default). With
   ``compute_frenet_frame=False``, ``DIRECT`` silently drops ``frenet_pose``,
   and requesting it explicitly via a ``FEATURES`` tuple raises
   ``ValueError: frenet_pose requested but environment does not compute the
   Frenet frame``.

.. warning::

   At spawn, ``frenet_pose[1]`` (``ey``) is generally **non-zero**: the Frenet
   frame tracks the **centerline**, but the RL reset strategies place the car
   on the **raceline**. Do not assume a fresh reset yields ``ey == 0``.

.. warning::

   With the LiDAR disabled (``lidar_config.enabled=False``), ``scan`` has shape
   ``(0,)``, not ``(num_beams,)``. Guard any code that indexes into the scan.

.. warning::

   After ``reset()``, the first observation's ``scan`` is all zeros — ``reset``
   does not run a LiDAR update. The scan is populated on the first ``step()``.
