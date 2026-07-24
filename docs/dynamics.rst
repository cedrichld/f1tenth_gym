Vehicle dynamics & parameters
==============================

The simulator advances every agent through a chosen **dynamics model** using a
numerical **integrator**, driven by a flat array of **vehicle parameters**.
This page explains the two usable models (KS and ST), how their state vectors
are laid out, what the pose reported in observations actually refers to, the
built-in parameter presets, and how to override individual parameters or switch
integrators.

All quantities are in SI units: metres (m), radians (rad), metres per second
(m/s), metres per second squared (m/s²), and curvature in rad/m.

.. note::

   There is no ``RaceCar`` class in this fork. Agents are **rows** in
   struct-of-arrays buffers driven by a single ``F110Simulator``; a model is
   just a choice of derivative function and a state-vector layout. See
   :doc:`configuration` for how the pieces fit together.

The models
----------

The model is selected through
:class:`~f1tenth_gym.envs.env_config.SimulationConfig` via its
``dynamics_model`` field, whose values come from
:class:`~f1tenth_gym.envs.dynamic_models.DynamicModel`.

.. list-table::
   :header-rows: 1
   :widths: 12 10 34 22 12

   * - Model
     - state_dim
     - State layout
     - Pose reference
     - Status
   * - ``KS`` (= 1)
     - 5
     - ``[x, y, delta, v, yaw]``
     - rear axle
     - ok, tested
   * - ``ST`` (= 2, **default**)
     - 7
     - ``[x, y, delta, v, yaw, yaw_rate, beta]``
     - centre of gravity (CoG)
     - ok, tested
   * - ``MB`` (= 3)
     - 29
     - multi-body + Pacejka tyres
     - CoG
     - **BROKEN — returns NaN**

Where ``x, y`` are the planar position (m), ``delta`` is the steering angle
(rad), ``v`` is the longitudinal speed (m/s), ``yaw`` is the heading (rad),
``yaw_rate`` is the yaw rate (rad/s), and ``beta`` is the body slip angle (rad).

Kinematic single-track (KS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``KS`` is a 5-state kinematic bicycle model. It has no tyre forces or slip, so
it is fast and well-behaved at low speed, but it does not capture sliding. Its
pose is referenced to the **rear axle**. Under ``KS`` there is no lateral
velocity, yaw rate, or slip angle, so the corresponding derived observation
fields (``ang_vel_z``, ``beta``, ``linear_vel_y``) are always exactly ``0``.

Single-track (ST, default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ST`` is the 7-state dynamic single-track model with linear tyre stiffness
(``C_Sf``/``C_Sr``). It adds yaw-rate and slip-angle states, so it reproduces
understeer/oversteer behaviour that ``KS`` cannot. Its pose is referenced to
the **centre of gravity**. ``ST`` is the default and is what all tuned presets
target.

.. warning::

   The multi-body model (``DynamicModel.MB``) is reachable from configuration
   but is **broken and returns NaN** — a parameter-ordering defect shifts the
   multi-body parameters so that a stiffness term divides by zero. Treat ``MB``
   as unavailable. Use ``KS`` or ``ST`` only. There is no ``MB`` test in the
   suite, and only the ``FULLSCALE`` preset even populates the multi-body
   fields.

Standardized state
~~~~~~~~~~~~~~~~~~~

Regardless of the model, the simulator also maintains a **standardized state**
of shape ``(N, 7)`` for the ``N`` agents, with columns::

   [X, Y, steering_angle, speed, yaw, yaw_rate, beta]

This is exposed as the ``std_state`` observation field (see
:doc:`observations`). Column 6 is the **slip angle** ``beta`` — not a lateral
velocity, despite some docstrings. Consumers that want lateral velocity compute
it as ``vy = speed * sin(beta)``. Under ``KS``, columns 5 and 6 (``yaw_rate``
and ``beta``) are hardcoded to ``0.0``.

The pose-frame subtlety
-----------------------

.. warning::

   The pose reported in observations is **not in the same physical frame** for
   ``KS`` and ``ST``. Under ``KS`` the reported pose is the **rear axle**;
   under ``ST`` it is the **centre of gravity**. Switching model therefore
   silently shifts the reported ``pose_x``/``pose_y`` (and ``std_state``
   position) by ``lr`` — about ``0.17145 m`` for the F1TENTH preset. If you
   compare trajectories, log positions, or feed poses to a downstream planner
   across different models, account for this offset.

VehicleParameters and presets
------------------------------

Vehicle parameters live in the frozen dataclass
:class:`~f1tenth_gym.envs.dynamic_models.VehicleParameters`. Three presets ship
with the package, importable from ``f1tenth_gym.envs.dynamic_models``:

- ``F1TENTH_VEHICLE_PARAMETERS`` — the 1/10-scale F1TENTH car (the default,
  ``EnvConfig.params``).
- ``F1FIFTH_VEHICLE_PARAMETERS`` — a 1/5-scale car.
- ``FULLSCALE_VEHICLE_PARAMETERS`` — a full-size vehicle; the only preset that
  populates the multi-body/Pacejka fields.

Key F1TENTH defaults
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Field
     - Value
     - Meaning
   * - ``m``
     - ``3.74``
     - mass (kg)
   * - ``mu``
     - ``1.0489``
     - tyre–ground friction coefficient
   * - ``I``
     - ``0.04712``
     - yaw moment of inertia (kg·m²)
   * - ``lf``
     - ``0.15875``
     - CoG-to-front-axle distance (m)
   * - ``lr``
     - ``0.17145``
     - CoG-to-rear-axle distance (m)
   * - ``h``
     - ``0.074``
     - CoG height (m)
   * - ``s_min`` / ``s_max``
     - ``-0.4189`` / ``0.4189``
     - steering-angle limits (rad)
   * - ``sv_min`` / ``sv_max``
     - ``-3.2`` / ``3.2``
     - steering-velocity limits (rad/s)
   * - ``v_min`` / ``v_max``
     - ``-5.0`` / ``20.0``
     - speed limits (m/s)
   * - ``v_switch``
     - ``7.319``
     - speed above which max accel is derated (m/s)
   * - ``a_max``
     - ``9.51``
     - maximum longitudinal acceleration (m/s²)
   * - ``width``
     - ``0.31``
     - vehicle width (m)
   * - ``length``
     - ``0.58``
     - vehicle length (m)
   * - ``C_Sf`` / ``C_Sr``
     - ``4.718`` / ``5.4562``
     - front/rear cornering stiffness (used by ``ST``)

The wheelbase is ``lf + lr`` (``0.3302 m`` for F1TENTH). These parameter values
also feed the action-space and observation-space bounds — see :doc:`actions`
and :doc:`observations`.

.. note::

   Field names are terse and **not** what you might guess: mass is ``m`` (not
   ``mass``), CoG height is ``h`` (not ``h_cg``), yaw inertia is ``I``. The
   steering and speed limits are ``s_min/s_max`` and ``v_min/v_max``.

.. warning::

   ``VehicleParameters`` is **append-only**. The parameters are marshalled to
   the numba dynamics kernels as a flat ``float32`` array that is indexed
   *positionally*, so field order is an ABI. Never insert or reorder fields —
   only ``with_updates()`` existing ones (safe) or append new fields at the end.

Overriding parameters
~~~~~~~~~~~~~~~~~~~~~~~

Because the dataclass is frozen, mutate it with ``with_updates()``, which
returns a new instance:

.. code-block:: python

   from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS

   # Heavier car on a lower-grip surface
   params = F1TENTH_VEHICLE_PARAMETERS.with_updates(m=4.0, mu=0.9)

Then place the modified parameters on the environment config via
``params=...``. For per-episode randomization of parameters instead, see the
domain-randomization interface in :doc:`rewards_and_rl`.

Integrators & timestep
-----------------------

The numerical integration scheme comes from
:class:`~f1tenth_gym.envs.integrators.IntegratorType`, set through
``SimulationConfig.integrator``:

- ``IntegratorType.RK4`` (= 2, **default**) — classic 4th-order Runge–Kutta.
- ``IntegratorType.EULER`` (= 1) — explicit forward Euler; cheaper, less
  accurate.

Each ``step()`` advances the physics over ``timestep`` seconds, subdivided into
substeps of ``integrator_timestep`` seconds. The integrator is invoked once per
substep per agent (RK4 evaluates the derivative four times per substep).

.. warning::

   ``timestep`` must be an **integer multiple** of ``integrator_timestep``, and
   the check is exact in IEEE-754 floating point. A seemingly valid pairing like
   ``timestep=0.03, integrator_timestep=0.01`` is **rejected**, because
   ``0.03 % 0.01`` is not ``0`` in float. Stick to values such as ``0.01``,
   ``0.02``, ``0.04``, ``0.05``, or ``0.1``.

Runnable example: KS model with an overridden parameter
-------------------------------------------------------

This selects the kinematic model, drops the friction coefficient, and switches
to the Euler integrator, then rolls the environment forward. Note that ``KS``
reports the **rear-axle** pose and keeps ``ang_vel_z``/``beta`` at zero.

.. code-block:: python

   import gymnasium as gym
   import numpy as np

   from f1tenth_gym.envs.env_config import (
       EnvConfig,
       SimulationConfig,
       ObservationConfig,
   )
   from f1tenth_gym.envs.observation import ObservationType
   from f1tenth_gym.envs.dynamic_models import (
       DynamicModel,
       F1TENTH_VEHICLE_PARAMETERS,
   )
   from f1tenth_gym.envs.integrators import IntegratorType

   cfg = EnvConfig(
       num_agents=1,
       params=F1TENTH_VEHICLE_PARAMETERS.with_updates(mu=0.9),
       simulation_config=SimulationConfig(
           dynamics_model=DynamicModel.KS,      # 5-state kinematic model
           integrator=IntegratorType.EULER,     # instead of the default RK4
           max_laps=None,                        # otherwise ends after one lap
       ),
       observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
       render_enabled=False,
   )

   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   obs, info = env.reset(seed=42)

   for _ in range(100):
       action = np.array([[0.1, 3.0]], dtype=np.float32)  # [[steer_rad, speed_mps]]
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break

   agent = obs["agent_0"]
   print("rear-axle pose:", float(agent["pose_x"]), float(agent["pose_y"]))
   print("speed:", float(agent["linear_vel_x"]))
   env.close()

To change the model or parameters on a **live** environment instead of at
construction time, build an updated config and call
``env.unwrapped.configure(new_cfg)``:

.. code-block:: python

   new_cfg = cfg.with_updates(
       simulation_config=cfg.simulation_config.with_updates(
           dynamics_model=DynamicModel.ST,
       ),
   )
   env.unwrapped.configure(new_cfg)

See also
--------

- :doc:`configuration` — the full frozen config tree and ``with_updates()``.
- :doc:`observations` — how ``state``, ``std_state``, and derived pose/velocity
  fields are produced from the model state.
- :doc:`actions` — how commanded steering/speed map through the vehicle limits.
- :doc:`rewards_and_rl` — per-episode vehicle-parameter randomization.
