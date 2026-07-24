.. image:: assets/f1_stickers_01.png
  :width: 60
  :align: left

F1TENTH Gym Documentation
=========================

``f1tenth_gym`` is a multi-agent, deterministic 1/10th-scale autonomous-racing
simulator that implements the `Gymnasium <https://gymnasium.farama.org/>`_
``Env`` API. It provides realistic single-track / kinematic vehicle dynamics,
a ray-cast 2D LiDAR, agent-and-wall collision detection, a Frenet-frame track
representation, and an OpenGL renderer — everything needed to develop and
evaluate planning and reinforcement-learning controllers.

.. note::

   This documentation covers the ``dev-humble`` line of ``f1tenth_gym``, which
   is configured entirely through a typed, frozen
   :class:`~f1tenth_gym.envs.env_config.EnvConfig` dataclass. It has **diverged
   substantially** from the older ``f110_gym`` package (dict-config,
   ``gym.make("f110_gym:f110-v0")``, 4-tuple ``step``); older tutorials will not
   work here. Start with :doc:`quickstart`.

Highlights
----------

- **Gymnasium-native** — ``reset() -> (obs, info)`` and
  ``step() -> (obs, reward, terminated, truncated, info)``.
- **Typed configuration** — one frozen
  :class:`~f1tenth_gym.envs.env_config.EnvConfig` tree; no YAML, no dicts. See
  :doc:`configuration`.
- **Multi-agent** — every agent is a row in struct-of-arrays physics buffers,
  stepped together and deterministically seedable.
- **Faster than real time** — steady-state stepping runs ~40–55× real time.
- **RL & sim2real ready** — pluggable rewards, domain randomization, actuator
  and sensor noise, and thin single-agent / observation-delay wrappers. See
  :doc:`rewards_and_rl`.

Quick example
-------------

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig

   env = gym.make(
       "f1tenth_gym:f1tenth-v0",
       config=EnvConfig(simulation_config=SimulationConfig(max_laps=None)),
   )
   obs, info = env.reset(seed=0)
   for _ in range(100):
       action = np.array([[0.0, 2.0]], dtype=np.float32)   # [[steer_rad, speed_mps]]
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break
   env.close()

Citing
------

If you use this environment, please cite:

.. code-block:: bibtex

   @inproceedings{okelly2020f1tenth,
     title={{F1TENTH}: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
     author={O'Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
     booktitle={NeurIPS 2019 Competition and Demonstration Track},
     pages={77--89},
     year={2020},
     organization={PMLR}
   }

To build a physical 1/10th-scale car, follow the guide at https://f1tenth.org/build.html.

.. toctree::
   :caption: Getting started
   :maxdepth: 2

   installation
   quickstart

.. toctree::
   :caption: User guide
   :maxdepth: 2

   configuration
   observations
   actions
   dynamics
   tracks
   rendering
   rewards_and_rl
   reproducibility
   examples

.. toctree::
   :caption: Reference
   :maxdepth: 2

   api/index
