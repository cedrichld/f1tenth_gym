Quickstart
==========

This page walks through a minimal, fully runnable episode: build an environment,
reset it, step a constant action in a loop, and read the ego observation. Every
snippet below runs as written against the real API on branch ``dev-humble``.

.. note::

   The first time you use a map, the simulator downloads it from
   ``https://api.f1tenth.org`` into a gitignored ``maps/`` directory, so the
   first ``gym.make`` / ``reset`` needs network access. See :doc:`tracks`.

A complete episode
------------------

.. code-block:: python

   import gymnasium as gym
   import numpy as np

   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig
   import f1tenth_gym  # noqa: F401  (import registers the "f1tenth-v0" id)

   cfg = EnvConfig(
       map_name="Spielberg",
       num_agents=1,
       # Default max_laps=1 ends the episode after ONE lap. Use None to run
       # until you decide to stop (or set a step limit, see configuration).
       simulation_config=SimulationConfig(max_laps=None),
       render_enabled=False,
   )

   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)

   obs, info = env.reset(seed=0)

   for _ in range(200):
       # action columns are [steering, longitudinal] -- STEERING FIRST.
       action = np.array([[0.0, 2.0]], dtype=np.float32)  # go straight at 2 m/s
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break

   ego = obs["agent_0"]
   print("speed:", float(ego["std_state"][3]), "sim_time:", float(ego["sim_time"]))
   env.close()

Creating the environment
------------------------

The environment is configured with a single :class:`~f1tenth_gym.envs.env_config.EnvConfig`
instance -- a **frozen dataclass tree**, not a dict or YAML file. Pass it as the
``config`` keyword to ``gym.make``:

.. code-block:: python

   env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(...))

The id is namespaced (``f1tenth_gym:f1tenth-v0``); the ``f1tenth_gym:`` prefix
triggers the package import whose side effect registers the id. The defaults are
a single agent on the ``Spielberg`` track with single-track (ST) dynamics,
speed + steering-angle control, and the ``DIRECT`` observation set. To change any
of these, see :doc:`configuration`.

Nested configuration mutates by nesting, since every config object is frozen:

.. code-block:: python

   cfg2 = cfg.with_updates(
       simulation_config=cfg.simulation_config.with_updates(max_laps=None)
   )

To reconfigure a live environment in place, call ``env.unwrapped.configure(cfg2)``.

reset() -> (obs, info)
----------------------

``reset`` follows the Gymnasium API and returns a **2-tuple** ``(obs, info)``.
Pass ``seed=`` for a reproducible start pose:

.. code-block:: python

   obs, info = env.reset(seed=0)

Without a seed, the spawn waypoint is drawn from OS entropy and the start pose
varies between runs. See :doc:`reproducibility` for exactly which random streams
a seed controls.

.. note::

   Immediately after ``reset`` the ``scan`` field is all zeros -- the reset path
   does not run a LiDAR sweep. The first real scan appears after the first
   ``step``.

step(action) -> the 5-tuple
---------------------------

``step`` returns the Gymnasium **5-tuple**
``(obs, reward, terminated, truncated, info)`` -- never a 4-tuple and never a
lone ``done``:

- ``obs`` -- the observation dict (see below).
- ``reward`` -- a float. With the default reward mode it is the physics
  timestep (``0.01``), i.e. pure survival time. See :doc:`rewards_and_rl`.
- ``terminated`` -- ``True`` on a collision (default) or when the lap target is
  reached.
- ``truncated`` -- driven by an optional step limit; ``False`` by default (no
  limit is imposed unless you configure one).
- ``info`` -- an auxiliary dict carrying per-agent quantities such as
  ``"collisions"``, ``"progress"``, and ``"sim_time"``.

Always end an episode on ``terminated or truncated``:

.. code-block:: python

   obs, reward, terminated, truncated, info = env.step(action)
   if terminated or truncated:
       obs, info = env.reset(seed=0)

The action array
----------------

An action is an ``np.ndarray`` of shape ``(num_agents, 2)`` and dtype
``float32``. The two columns are ``[steering, longitudinal]`` -- **steering is
column 0**. For a single agent:

.. code-block:: python

   action = np.array([[steer_rad, speed_mps]], dtype=np.float32)

With the default control config, column 0 is a steering angle in radians and
column 1 is a target speed in m/s (a PID converts it to acceleration).
The bounds and alternative control modes (acceleration, steering-speed) are
documented in :doc:`actions`.

.. warning::

   Both columns are ``float32``, so swapping them fails **silently** -- the car
   just drives wrong. If your car spins or refuses to move, check that steering
   is in column 0.

Reading the ego observation
---------------------------

The observation is a nested dict: ``obs[agent_id][field]``, with keys
``"agent_0"``, ``"agent_1"``, ... Every scalar field is a **0-d float32
ndarray**, so wrap reads in ``float(...)`` when you want a Python number.

Under the default ``DIRECT`` observation set, the ego pose is available through
the 7-element ``std_state`` array, laid out as
``[X, Y, steering_angle, speed, yaw, yaw_rate, beta]``:

.. code-block:: python

   ego = obs["agent_0"]
   x, y   = float(ego["std_state"][0]), float(ego["std_state"][1])
   yaw    = float(ego["std_state"][4])
   speed  = float(ego["std_state"][3])

If you would rather read named pose fields, switch to the ``KINEMATIC_STATE``
observation set, which adds ``pose_x``, ``pose_y``, ``pose_theta`` (and the
velocity/steering derived fields):

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
   obs, info = env.reset(seed=0)
   obs, *_ = env.step(np.array([[0.0, 2.0]], dtype=np.float32))
   ego = obs["agent_0"]
   print(float(ego["pose_x"]), float(ego["pose_y"]), float(ego["pose_theta"]))

The full field vocabulary of each observation set is documented in
:doc:`observations`.

The three classic traps
------------------------

.. warning::

   Three defaults trip up nearly every newcomer:

   1. **The episode ends after one lap.** ``max_laps`` defaults to ``1``, so a
      long run stops early with ``terminated=True``. Set
      ``SimulationConfig(max_laps=None)`` for an endless episode.

   2. **Steering is action column 0.** The layout is ``[steering, longitudinal]``.
      Because both are ``float32``, a swap fails silently.

   3. **``DIRECT`` has no ``pose_x``.** ``pose_x`` / ``pose_y`` / ``pose_theta``
      are *derived* fields absent from the default ``DIRECT`` set --
      ``obs["agent_0"]["pose_x"]`` raises ``KeyError``. Either read
      ``std_state`` (indices 0, 1, 4) or use
      ``ObservationConfig(type=ObservationType.KINEMATIC_STATE)``.

Where to go next
----------------

- :doc:`configuration` -- the full frozen-config tree and every knob.
- :doc:`examples` -- runnable scripts, including a pure-pursuit follower.
