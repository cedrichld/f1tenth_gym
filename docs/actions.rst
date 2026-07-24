Actions
=======

Every :class:`~f1tenth_gym.envs.f110_env.F110Env` consumes a single action array
per :meth:`step`. This page explains the fixed shape of that array, what each
column means, the two longitudinal and two steering interpretation modes, how to
select them, and how to read the environment's :attr:`action_space`.

For where these controls sit in the wider configuration tree, see
:doc:`configuration`. For what the simulator gives back, see :doc:`observations`.

The action array
----------------

The action space is **always** a ``gymnasium.spaces.Box`` of shape
``(num_agents, 2)`` with ``dtype=np.float32``. There is one row per agent and
exactly two columns:

* **Column 0 — steering** (``[steer, ...]``)
* **Column 1 — longitudinal** (``[..., longitudinal]``)

Single-agent code must still pass a 2-D array with a leading agent axis:

.. code-block:: python

   import numpy as np

   action = np.array([[0.0, 2.0]], dtype=np.float32)   # [[steer, speed]]

.. warning::

   **Steering is column 0, longitudinal is column 1.** Both columns are
   ``float32``, so swapping them fails **silently** — the environment happily
   interprets your speed command as a steering command and vice versa, and you
   just get nonsense trajectories with no error. Always order the pair
   ``[steer, longitudinal]``.

Longitudinal modes
------------------

The meaning of column 1 is set by
:class:`~f1tenth_gym.envs.action.LongitudinalActionType`:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Meaning of column 1
     - Box bounds (per row)
   * - ``SPEED`` *(default)*
     - Target speed in m/s, tracked by a proportional controller
     - ``[v_min, v_max]``
   * - ``ACCL``
     - Direct longitudinal acceleration in m/s², passed through unchanged
     - ``[-a_max, a_max]``

Under the default ``SPEED`` mode the commanded speed is turned into an
acceleration by :func:`~f1tenth_gym.envs.dynamic_models.utils.pid_accl`, a real
proportional controller with four gain quadrants (forward vs. reverse, and
accelerating vs. braking). It is a P controller only — there is no I or D term.

Under ``ACCL`` mode the command is used as the acceleration directly
(:func:`~f1tenth_gym.envs.action.accl_action` is the identity).

.. note::

   For the default ``F1TENTH_VEHICLE_PARAMETERS`` the speed bounds are
   ``v_min = -5.0`` m/s and ``v_max = 20.0`` m/s, and the acceleration bound is
   ``a_max = 9.51`` m/s². These come from the vehicle parameters, so they change
   if you swap :attr:`EnvConfig.params` to another preset (``F1FIFTH_`` or
   ``FULLSCALE_VEHICLE_PARAMETERS``).

Steering modes
--------------

The meaning of column 0 is set by
:class:`~f1tenth_gym.envs.action.SteerActionType`:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Meaning of column 0
     - Box bounds (per row)
   * - ``STEERING_ANGLE`` *(default)*
     - Target steering angle in radians
     - ``[s_min, s_max]``
   * - ``STEERING_SPEED``
     - Direct steering angular velocity in rad/s, passed through unchanged
     - ``[sv_min, sv_max]``

.. warning::

   Under ``STEERING_ANGLE`` the target angle is realised by
   :func:`~f1tenth_gym.envs.dynamic_models.utils.pid_steer`, which despite its
   name is **not a PID controller — it is bang-bang**. Whenever the angle error
   exceeds ``1e-4`` rad it commands the full ``±sv_max`` steering velocity toward
   the target, and zero otherwise. Expect the steering rate to slam to its limit
   rather than ramp smoothly.

Under ``STEERING_SPEED`` the command is the steering angular velocity directly
(:func:`~f1tenth_gym.envs.action.steering_speed_action` is the identity).

.. note::

   For the default ``F1TENTH_VEHICLE_PARAMETERS`` the steering-angle bounds are
   ``s_min = -0.4189`` rad and ``s_max = 0.4189`` rad, and the steering-velocity
   bounds are ``sv_min = -3.2`` rad/s and ``sv_max = 3.2`` rad/s.

Selecting the modes
-------------------

Both modes live on :class:`~f1tenth_gym.envs.env_config.ControlConfig`, nested
inside :class:`~f1tenth_gym.envs.env_config.EnvConfig` as
``control_config``. The defaults are ``longitudinal_mode = SPEED`` and
``steering_mode = STEERING_ANGLE``. Because the config tree is a tree of frozen
dataclasses, mutate it by nesting ``with_updates`` calls:

.. code-block:: python

   from f1tenth_gym.envs.env_config import EnvConfig
   from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType

   cfg = EnvConfig()
   cfg = cfg.with_updates(
       control_config=cfg.control_config.with_updates(
           longitudinal_mode=LongitudinalActionType.ACCL,
           steering_mode=SteerActionType.STEERING_SPEED,
       ),
   )

``ControlConfig`` also carries the sim2real actuation knobs
``steer_delay_steps``, ``throttle_delay_steps``, ``steer_noise_std`` and
``accl_noise_std`` (all default ``0`` / no-op); see :doc:`configuration` and
:doc:`rewards_and_rl`.

Reading the action space
------------------------

The environment builds its action space from the selected modes and the vehicle
parameters, so you never need to hardcode the bounds — read them off
``env.action_space``:

.. code-block:: python

   import gymnasium as gym
   from f1tenth_gym.envs.env_config import EnvConfig

   env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
   env.reset(seed=0)

   print(env.action_space.shape)   # (1, 2)  -> (num_agents, 2)
   print(env.action_space.low)     # [[s_min, v_min]]  e.g. [[-0.4189, -5.0]]
   print(env.action_space.high)    # [[s_max, v_max]]  e.g. [[ 0.4189, 20.0]]

   # A valid random action always has the right shape and dtype:
   action = env.action_space.sample()
   obs, reward, terminated, truncated, info = env.step(action)
   env.close()

.. note::

   The single-agent Box is repeated once per agent to form the
   ``(num_agents, 2)`` space, so ``env.action_space.low[i]`` gives the
   ``[steer, longitudinal]`` lower bounds for agent ``i`` — identical across
   agents.

Single-agent example
--------------------

A minimal control loop under the default ``SPEED`` + ``STEERING_ANGLE`` modes:
drive straight at 2 m/s.

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig

   cfg = EnvConfig(
       num_agents=1,
       simulation_config=SimulationConfig(max_laps=None),   # otherwise ends after 1 lap
       render_enabled=False,
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   obs, info = env.reset(seed=42)

   for _ in range(200):
       action = np.array([[0.0, 2.0]], dtype=np.float32)    # [steer_rad, speed_mps]
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break

   env.close()

Multi-agent example
-------------------

With ``num_agents=2`` the action array has two rows — one command pair per agent.
Here agent 0 goes straight and agent 1 steers left while both hold 2 m/s:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig

   cfg = EnvConfig(
       num_agents=2,
       simulation_config=SimulationConfig(max_laps=None),
       render_enabled=False,
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   obs, info = env.reset(seed=42)

   for _ in range(200):
       action = np.array(
           [
               [0.0, 2.0],   # agent_0: steer straight, 2 m/s
               [0.2, 2.0],   # agent_1: steer left,     2 m/s
           ],
           dtype=np.float32,
       )
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break

   env.close()

.. note::

   The single-agent :class:`~f1tenth_gym.envs.wrappers.SingleAgentWrapper`
   reshapes a flat ``(2,)`` action into the ``(1, 2)`` array the env expects, so
   wrapped single-agent code can pass ``np.array([steer, speed])`` directly. See
   :doc:`rewards_and_rl`.

See also
--------

* :doc:`configuration` — the full ``EnvConfig`` / ``ControlConfig`` tree.
* :doc:`observations` — what ``step`` returns.
* :doc:`dynamics` — how the realised steering velocity and acceleration feed the
  vehicle models, and where the ``s_min``/``v_max`` bounds come from.
* :doc:`quickstart` — an end-to-end first program.
