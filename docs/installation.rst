Installation
============

This page covers installing ``f1tenth_gym`` (branch ``dev-humble``) and verifying
that the environment works end to end. Once installed, head to
:doc:`quickstart` for your first driving loop.

.. note::

   This fork has diverged sharply from the upstream ``f110_gym`` package. The
   installable package is ``f1tenth_gym`` and the gym id is
   ``f1tenth_gym:f1tenth-v0``. The legacy ``f110_gym`` / ``f110-v0`` API does
   not exist here.

Requirements
------------

* **Python 3.9 or newer.** The project declares ``requires-python = ">=3.9"``
  and is classified for CPython 3.9 through 3.14.
* A working C toolchain is not required for normal use — ``numba`` ships wheels
  and JIT-compiles the physics kernels at runtime.
* **Network access on first use**: the requested track map is downloaded from
  ``api.f1tenth.org`` the first time you use it (see
  `First run downloads the map`_).
* Rendering requires an X display (real or virtual). See `Rendering needs a
  display`_.

Core dependencies
~~~~~~~~~~~~~~~~~~

Installing the package pulls in its runtime dependencies automatically. The
core stack is:

* ``numpy`` — struct-of-arrays state buffers and math
* ``scipy`` — spline / interpolation for tracks
* ``numba`` — JIT-compiled dynamics, LiDAR, and collision kernels
* ``gymnasium`` (``>=0.29.1,<0.30``) — the ``Env`` API
* ``pillow`` — occupancy-map image loading
* ``pyyaml`` — map metadata parsing
* ``pyqtgraph`` and ``pyqt6`` — the OpenGL renderer
* ``pyopengl`` / ``pyopengl-accelerate`` — GL backend
* ``requests`` — map download
* plus ``opencv-python``, ``pandas``, and ``yamldataclassconfig``

The development group (``pytest``, ``black``, ``isort``, ``autoflake``,
``flake8``, ``matplotlib``, ``ipykernel``) is installed automatically by
``uv sync`` and is optional for pip users.

Install with uv (recommended)
------------------------------

`uv <https://docs.astral.sh/uv/>`_ manages the virtual environment and locks
dependencies from ``uv.lock``. Clone the repository and sync:

.. code-block:: bash

   git clone https://github.com/f1tenth/f1tenth_gym.git
   cd f1tenth_gym
   git checkout dev-humble
   uv sync

``uv sync`` creates ``.venv/`` and installs the package (editable) together with
its runtime and dev dependencies. Run subsequent commands through ``uv run``,
for example ``uv run python your_script.py``.

.. warning::

   ``uv sync`` also **uninstalls** anything not present in ``uv.lock``. A couple
   of examples need extra packages that are not declared
   (``moviepy`` for :doc:`examples` video recording, ``shapely`` for track
   generation). Re-install them after a sync with, e.g.,
   ``uv pip install moviepy``.

Install with pip
----------------

From a clone (editable install), which is the recommended layout because the
map downloader writes into the repo-root ``maps/`` directory:

.. code-block:: bash

   git clone https://github.com/f1tenth/f1tenth_gym.git
   cd f1tenth_gym
   git checkout dev-humble
   pip install -e .

Or install directly from GitHub, pinned to this branch:

.. code-block:: bash

   pip install "git+https://github.com/f1tenth/f1tenth_gym.git@dev-humble"

.. note::

   Branch off ``dev-humble``, never ``main``. ``origin/main`` is a legacy line
   that still ships the different, incompatible ``gym/f110_gym/`` package.

First run downloads the map
---------------------------

The ``maps/`` directory is gitignored, so it starts empty. The first time you
create an environment for a given track, the simulator downloads it from a
hardcoded URL, ``https://api.f1tenth.org/<map_name>.tar.xz``, and extracts it
into the repo-root ``maps/`` directory. The default map is ``Spielberg``.

.. warning::

   The map directory is resolved relative to the installed source tree
   (four ``.parent`` hops from ``track/utils.py``), so the download path works
   for an **editable / cloned install**. Network access is required on that
   first run; afterwards the cached map under ``maps/`` is reused offline.

Verify your install
--------------------

This snippet creates the default single-agent Spielberg environment, resets it
with a fixed seed for a reproducible start pose, steps it a few times, prints a
value, and closes. It requires network access on the very first run to fetch the
map.

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import (
       EnvConfig,
       ObservationConfig,
       SimulationConfig,
   )
   from f1tenth_gym.envs.observation import ObservationType

   cfg = EnvConfig(
       map_name="Spielberg",
       num_agents=1,
       simulation_config=SimulationConfig(max_laps=None),   # default is 1 -> ends after one lap!
       observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
       render_enabled=False,
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   obs, info = env.reset(seed=42)

   for _ in range(100):
       action = np.array([[0.0, 2.0]], dtype=np.float32)   # [[steer_rad, speed_mps]] — steer FIRST
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break

   print(obs["agent_0"]["pose_x"], obs["agent_0"]["linear_vel_x"], info["sim_time"])
   env.close()

A healthy install prints a speed near the commanded ``2.0`` m/s and a
``sim_time`` around ``1.0`` after 100 steps of ``timestep=0.01``. With
``seed=42`` the start pose is reproducible (``x=-1.567``, ``y=-1.257``).

.. note::

   ``step()`` returns the gymnasium 5-tuple
   ``(obs, reward, terminated, truncated, info)`` — never a 4-tuple and never a
   bare ``done``. The action is an array of shape ``(num_agents, 2)`` with
   columns ``[steering, longitudinal]`` — **steering is column 0**. The default
   ``DIRECT`` observation does not expose ``pose_x``; the snippet above requests
   ``KINEMATIC_STATE`` so ``obs["agent_0"]["pose_x"]`` exists. See
   :doc:`observations` and :doc:`actions` for details.

Rendering needs a display
-------------------------

The renderer is an OpenGL backend (``pyqtgraph.opengl`` / PyQt6) and needs an X
display — a real one, or a virtual one via ``xvfb``. It cannot render under the
headless ``offscreen`` Qt platform. If you request a display render mode with no
``$DISPLAY``, the environment raises a ``RuntimeError`` with setup guidance
rather than failing deep inside Qt.

For headless servers, CI, and Google Colab (``xvfb`` + ``rgb_array`` video),
see :doc:`rendering`.

Running the tests
-----------------

The test suite (all plain ``unittest`` cases run through pytest) needs network
access for map downloads and mutates the working tree during track-download
tests:

.. code-block:: bash

   env -u PYTHONPATH uv run pytest

.. note::

   The ``env -u PYTHONPATH`` prefix is required only when ROS 2 Humble is on
   your ``PYTHONPATH`` — ``/opt/ros/humble`` registers a pytest plugin whose
   import chain is not available in the venv, which otherwise breaks collection.
   Without ROS on the path, plain ``uv run pytest`` works.

Next steps
----------

* :doc:`quickstart` — your first driving loop, explained.
* :doc:`configuration` — the frozen :class:`~f1tenth_gym.envs.env_config.EnvConfig`
  tree and how to mutate it.
* :doc:`observations` and :doc:`actions` — the observation dict and action layout.
* :doc:`rendering` — display setup, headless, and Colab.
