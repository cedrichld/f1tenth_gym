Examples
========

The ``examples/`` directory contains runnable scripts that exercise the
simulator through the real Gymnasium API. This page is an annotated catalogue:
what each script demonstrates and how to run it. All commands assume you are at
the repository root with the package installed (see :doc:`installation`).

.. note::

   Every example creates the env the same way::

       import gymnasium as gym
       from f1tenth_gym.envs.env_config import EnvConfig
       env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(...))

   and drives the gymnasium 5-tuple ``obs, reward, terminated, truncated, info =
   env.step(action)`` with ``action`` of shape ``(num_agents, 2)`` and columns
   ``[steering, longitudinal]`` (steering first). See :doc:`quickstart`,
   :doc:`configuration`, and :doc:`actions`.

waypoint_follow.py
------------------

The flagship example. It is two things at once:

1. **A pure-pursuit waypoint follower.** ``PurePursuitPlanner`` binds to
   ``track.raceline`` (the optimized racing line, not the centerline) and, each
   step, reads the ego pose and returns a ``(speed, steering)`` command that
   chases a lookahead point on the line. It uses numba-jitted geometry helpers
   (``nearest_point_on_trajectory``, ``first_point_on_trajectory_intersecting_circle``,
   ``get_actuation``). The planner also publishes render callbacks
   (``render_lookahead_point``, ``render_local_plan``) so the lookahead point and
   local plan draw on top of the map.

2. **A fully-spelled-out configuration reference.** ``build_config()`` constructs
   an :class:`~f1tenth_gym.envs.env_config.EnvConfig` with *every* field of
   *every* nested config dataclass written out explicitly -- almost all set to
   their real default. It is the canonical copy-paste template for seeing every
   knob and its default at a glance: the top-level fields (``seed``,
   ``map_name``, ``map_scale``, ``num_agents``, ``ego_index``,
   ``collision_check``, ``render_enabled``, ``params``) plus ``ControlConfig``,
   ``SimulationConfig``, ``ObservationConfig``, ``ResetConfig``, ``LiDARConfig``,
   ``RenderConfig``, ``TerminationConfig``, ``RewardConfig``, and
   ``DomainRandomizationConfig``. If you want to understand the configuration
   tree, read this function alongside :doc:`configuration`.

The one deliberate deviation from the defaults is the observation type: it uses
``ObservationType.KINEMATIC_STATE`` instead of the default ``DIRECT`` so the
follower can read ``obs["agent_0"]["pose_x"]`` / ``pose_y`` / ``pose_theta``
directly. ``DIRECT`` does not expose those derived fields (see
:doc:`observations`).

Run it (needs an X display for the ``"human"`` render mode)::

.. code-block:: bash

   python examples/waypoint_follow.py

To run fully headless, set ``render_enabled=False`` in ``build_config()`` and
change ``render_mode`` in ``main()``; render callbacks are only wired up when a
renderer actually exists.

.. warning::

   The planner's ``max_reacquire`` branch is a known footgun: it can raise a
   numba ``TypingError`` if the car drifts more than the lookahead distance but
   less than 20 m off the raceline. Keep the follower near the line.

telemetry_plot.py
-----------------

A standalone real-time telemetry dashboard. While a simple plain-NumPy
pure-pursuit follower drives a lap, it live-plots four channels in a
`pyqtgraph <https://www.pyqtgraph.org/>`_ window -- **speed**, **steering
angle**, **yaw rate**, and **slip angle (beta)** -- all read out of
``obs["agent_0"]["std_state"]`` (the ``(7,)`` standardized state
``[X, Y, steering, speed, yaw, yaw_rate, beta]``).

This example is intentionally **not** wired into the gym's OpenGL renderer
(``render_enabled=False``); it is a self-contained pattern for low-latency
plotting you can lift into your own training or evaluation scripts. Because it
never imports the GL backend it runs without a GPU, though a display/X server is
still needed for the Qt window (use ``xvfb-run`` for a virtual one).

The GUI refreshes at a fixed rate (``--fps``) while the dynamics advance by a
configurable real-time factor (``--rtf``): the sim can run several physics steps
per plotted frame, decoupling dynamics speed from plot refresh. It builds the
env with ``SimulationConfig(max_laps=None)`` so the rollout does not end after
one lap.

Run it::

.. code-block:: bash

   python examples/telemetry_plot.py --map Spielberg --rtf 1.0

Flags: ``--map`` (track name, default ``Spielberg``), ``--rtf`` (real-time
factor, default ``1.0``), ``--fps`` (plot refresh rate, default ``50``),
``--window`` (seconds of history shown, default ``10``). Requires ``pyqtgraph``
(a core dependency) and a Qt binding.

run_in_empty_track.py
---------------------

Demonstrates building a **synthetic, mapless track** from a custom reference
line with ``Track.from_refline(x, y, velx)`` instead of downloading a real map.
The script lays down a straight-ish reference line and passes the resulting
``Track`` instance directly as ``map_name`` in the
:class:`~f1tenth_gym.envs.env_config.EnvConfig`, then drives it with the same
``PurePursuitPlanner`` from ``waypoint_follow.py``. Useful for testing control
logic without a full circuit. See :doc:`tracks` for the details of synthetic
tracks.

.. warning::

   Every reference line is force-closed into a periodic loop. A line from
   ``(0,0)`` to ``(10,0)`` is therefore **not** a 10 m open straight -- it
   becomes a closed ~20 m path with a phantom return leg, and ``s``, curvature,
   and lap counting all run over that loop. There is no open-path mode. See
   :doc:`tracks`.

Run it::

.. code-block:: bash

   python examples/run_in_empty_track.py

video_recording.py
------------------

Records an ``rgb_array`` rollout to an MP4 using gymnasium's ``RecordVideo``
wrapper. It builds the env with ``render_mode="rgb_array"`` (a fast OpenGL
framebuffer grab), wraps it, resets from an explicit spawn pose via
``env.reset(options={"poses": poses})``, and drives a few seconds of
pure-pursuit before ``env.close()`` flushes the video. See :doc:`rendering` for
how ``rgb_array`` and the render clock work.

Run it (on a desktop with a display, or under ``xvfb-run`` on a headless
server)::

.. code-block:: bash

   xvfb-run -a python examples/video_recording.py

.. note::

   ``video_recording.py`` needs **moviepy**, which is *not* a declared
   dependency -- and ``uv sync`` actively uninstalls it. Install it after every
   sync::

       pip install moviepy

   Without it, ``RecordVideo`` raises ``gymnasium.error.DependencyNotInstalled``.
   The output directory (``video_<timestamp>/``) is created next to where you run
   the script.

random_trackgen.py
------------------

Procedurally generates random closed tracks. Included for completeness.

.. note::

   ``random_trackgen.py`` imports **shapely**, which is declared nowhere (not in
   ``pyproject.toml``, not in ``uv.lock``). Install it manually before running::

       pip install shapely

A note on running the examples
------------------------------

``run_in_empty_track.py`` and ``video_recording.py`` import ``PurePursuitPlanner``
from ``waypoint_follow`` as a sibling module, so run them from inside the
``examples/`` directory (or with ``examples/`` on ``PYTHONPATH``) so the import
resolves. The first run of any example that uses a real map downloads the track
from the F1TENTH map server, so network access is required once. See
:doc:`installation` and :doc:`reproducibility`.
