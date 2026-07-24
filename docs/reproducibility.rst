Reproducibility & determinism
=============================

The simulator is fully deterministic: given the same seed and the same
sequence of actions, an episode replays bit-for-bit. This page explains
what "the seed" actually controls, how :func:`reset` seeding follows the
gymnasium contract, and how to write experiments whose runs you can
reproduce exactly.

If you have not yet built an environment, start with :doc:`quickstart`.
For the configuration surface referenced below, see
:doc:`configuration`.

The determinism model
----------------------

Every physics step advances **all agents together** through the
struct-of-arrays simulator. There is no per-agent thread, no
asynchronous work, and no wall-clock coupling in the physics: stepping
faster or slower than real time changes nothing about the trajectory
(only the renderer watches the wall clock). Concretely, the state
transition is a pure function of

* the current simulation state,
* the action you pass to :func:`step`, and
* the random draws seeded at ``reset`` time (LiDAR noise, actuator
  noise, domain randomization, and the spawn pose).

Fix those inputs and the observations are identical across runs, across
processes, and across machines with the same NumPy build.

.. note::

   The physics kernels compute in float64 and the result is re-cast to
   float32 in the state buffers every step. This is deterministic — the
   same inputs always produce the same float32 result — but it does mean
   trajectories are stored at float32 precision. See :doc:`dynamics`.

What ``reset(seed=...)`` seeds
------------------------------

``reset(seed=...)`` is the gymnasium way to seed, and it is the seed that
matters for reproducibility. Passing a seed calls
``gymnasium.Env.reset(seed=seed)``, which seeds the env's ``np_random``
generator. From that single generator the environment then derives
**every** stochastic input for the episode:

* **Spawn pose** — the reset strategy samples from ``np_random``. Under
  the default ``RL_GRID_STATIC`` reset this draws the spawn waypoint via
  ``rng.choice`` on the raceline, so the start ``x``/``y`` depend on the
  seed.
* **Per-agent LiDAR-noise streams** — ``reset`` draws an integer from
  ``np_random`` and passes it to the simulator as ``noise_seed``; each
  agent ``idx`` gets its own generator seeded at ``noise_seed + idx``.
  This drives the Gaussian scan ``noise_std``, per-beam ``dropout_prob``,
  and the once-per-episode per-beam ``range_bias_std`` (see
  :doc:`observations`).
* **Actuator-noise stream** — the simulator's ``control_rng`` is seeded
  at ``noise_seed + 2**20`` (offset well away from the per-agent scan
  seeds so the streams never collide). This drives ``steer_noise_std``
  and ``accl_noise_std`` from :class:`~f1tenth_gym.envs.env_config.ControlConfig`.
* **Domain-randomization draw** — when
  :class:`~f1tenth_gym.envs.env_config.DomainRandomizationConfig` is
  enabled, the per-episode vehicle-parameter sample is drawn from
  ``np_random`` before the sim resets, so randomized dynamics are
  reproducible with the reset seed too (see :doc:`rewards_and_rl`).

Because all of these flow from the one generator seeded by
``reset(seed=...)``, a fixed seed pins the entire episode's randomness.

``EnvConfig.seed`` vs ``reset(seed=...)``
-----------------------------------------

:class:`~f1tenth_gym.envs.env_config.EnvConfig` has a ``seed`` field
(default ``12345``). It is the **base/default** seed the simulator falls
back to for its noise generators when ``reset`` is called **without** a
seed. It does **not**, by itself, fix the spawn pose across no-seed
episodes.

.. warning::

   ``EnvConfig.seed`` alone does **not** make the start pose
   reproducible. When you call ``env.reset()`` with no ``seed`` argument,
   gymnasium seeds ``np_random`` from OS entropy, so the spawn-waypoint
   draw — and therefore the reported ``x``/``y`` — varies run to run,
   regardless of ``EnvConfig.seed``. To fix the start pose, pass
   ``env.reset(seed=...)`` explicitly.

For a fully controlled experiment, pass the same integer to
``reset(seed=...)`` every episode you want to reproduce. If you want a
different-but-reproducible pose per episode, seed with a deterministic
sequence (e.g. ``reset(seed=episode_index)``).

Reproducing two identical rollouts
----------------------------------

The following script runs the same seeded reset and the same action
sequence twice, and asserts the observation streams are identical.

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig

   cfg = EnvConfig(
       map_name="Spielberg",
       num_agents=1,
       simulation_config=SimulationConfig(max_laps=None),  # don't end after one lap
       render_enabled=False,
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)

   def rollout(seed):
       obs, info = env.reset(seed=seed)
       # a fixed, deterministic action sequence
       xs = []
       for t in range(200):
           steer = 0.1 * np.sin(t * 0.05)
           action = np.array([[steer, 3.0]], dtype=np.float32)  # [[steer, speed]]
           obs, reward, terminated, truncated, info = env.step(action)
           xs.append(float(obs["agent_0"]["std_state"][0]))  # X of the standard state
           if terminated or truncated:
               break
       return np.array(xs)

   a = rollout(seed=42)
   b = rollout(seed=42)

   assert np.array_equal(a, b), "seeded rollouts diverged!"
   print("identical:", np.array_equal(a, b), "steps:", len(a))

   env.close()

Both rollouts start from the same spawn pose (seed ``42``) and receive
the same actions, so every recorded value matches exactly. Change the
seed passed to either ``rollout`` call and the trajectories differ.

.. note::

   The default ``DIRECT`` observation does not expose ``pose_x`` — it is a
   derived field. The example reads ``std_state[0]`` (the X coordinate)
   instead. Use the ``KINEMATIC_STATE`` observation type if you want
   named ``pose_x``/``pose_y`` fields. See :doc:`observations`.

.. warning::

   The action layout is ``[steering, longitudinal]`` — **steering is
   column 0**. Both columns are float32, so swapping them fails silently
   rather than raising. For a single agent, pass
   ``np.array([[steer, speed]], dtype=np.float32)``.

.. note::

   ``reset`` writes the spawn state but never runs a LiDAR scan, so the
   ``scan`` field in the observation returned by ``reset`` is all zeros;
   the first real scan appears after the first ``step``. This is
   deterministic and identical across seeded runs.

Unseeded episodes vary
----------------------

If you omit the seed, runs are **not** reproducible by design:

.. code-block:: python

   import gymnasium as gym
   from f1tenth_gym.envs.env_config import EnvConfig

   env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
   obs1, _ = env.reset()   # np_random seeded from OS entropy
   obs2, _ = env.reset()   # different entropy -> different spawn
   # obs1 and obs2 generally differ in x/y and in LiDAR noise
   env.close()

This is standard gymnasium behaviour: without a seed, ``np_random`` is
drawn from OS entropy, so the spawn pose (and every downstream noise
stream) changes each episode. On Spielberg the start ``x`` roughly spans
``-1.6 … -2.4`` and ``y`` roughly ``-1.26 … -1.46`` across unseeded
resets. Seed the reset to eliminate this variation.

Checklist for a reproducible experiment
----------------------------------------

* Pass an explicit ``seed`` to **every** ``reset`` you want to reproduce
  (do not rely on ``EnvConfig.seed`` for the spawn pose).
* Use a **fixed, deterministic** action sequence (or a seeded policy).
* Keep the configuration identical between runs — reconfiguring a live
  env with ``env.unwrapped.configure(new_cfg)`` changes the physics and
  the observation surface. See :doc:`configuration`.
* Domain randomization and all sim2real noise knobs are already folded
  into the reset seed, so you do not need to seed them separately — just
  seed the reset. See :doc:`rewards_and_rl`.

See also :doc:`quickstart`, :doc:`observations`, and :doc:`actions`.
