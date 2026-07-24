Rendering
=========

f1tenth_gym ships a single OpenGL rendering backend (``PyQtEnvRendererGL``,
built on ``pyqtgraph.opengl``). It draws a top-down, north-up view of the
track and cars, follows a chosen agent with the camera, and can either drive a
live GUI window or hand you raw RGB frames for video recording.

The renderer is **fully decoupled from the physics step**: you own the step
loop, and the environment only draws when you call ``env.render()``. A small
internal ``RenderClock`` governs pacing and frame emission, so how fast you
step the simulation and how often the screen redraws are two separate things.

.. note::

   Rendering is OpenGL-only and needs an **X display** (real or virtual). There
   is no 2D raster fallback and no headless ``offscreen`` Qt path -- GL cannot
   allocate a framebuffer without a display. See `Headless and Colab`_ below for
   running under ``xvfb``.

Enabling rendering
------------------

Two things turn rendering on: ``render_enabled=True`` in the
:class:`~f1tenth_gym.envs.env_config.EnvConfig` (the default), and a
``render_mode`` passed to ``gym.make``.

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig

   cfg = EnvConfig(
       map_name="Spielberg",
       num_agents=1,
       simulation_config=SimulationConfig(max_laps=None),  # don't end after one lap
       render_enabled=True,   # default
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="human")
   obs, info = env.reset(seed=42)

   for _ in range(1000):
       action = np.array([[0.0, 4.0]], dtype=np.float32)  # [[steer, speed]]
       obs, reward, terminated, truncated, info = env.step(action)
       env.render()
       if terminated or truncated:
           obs, info = env.reset(seed=42)
   env.close()

If ``render_enabled=False``, no renderer is built and ``env.render()`` returns
``None`` regardless of ``render_mode``.

Render modes
------------

``render_mode`` is one of ``"human"``, ``"human_fast"``, ``"unlimited"``, or
``"rgb_array"`` (the env's ``metadata["render_modes"]``). The three human modes
draw to a live window and pace the loop; ``rgb_array`` never sleeps and returns
frames instead.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - ``render_mode``
     - Real-time factor
     - Behaviour
   * - ``"human"``
     - ``render_config.real_time_factor`` (default ``1.0``)
     - Live window, paced to real time. Redraws capped at ``render_fps``/wall-second.
   * - ``"human_fast"``
     - ``10.0`` (legacy sugar)
     - Live window, runs the sim 10x faster than real time.
   * - ``"unlimited"``
     - ``float("inf")`` (free-run)
     - Live window, **no pacing** -- steps as fast as the CPU allows.
   * - ``"rgb_array"``
     - n/a (never paces)
     - No window. ``env.render()`` returns an ``(H, W, 3)`` ``uint8`` frame.

.. warning::

   Requesting any display render mode (``"human"``, ``"human_fast"``,
   ``"unlimited"``, or ``"rgb_array"``) when ``$DISPLAY`` is unset raises a
   clear ``RuntimeError`` at construction with ``xvfb``/Colab setup steps,
   instead of failing cryptically deep inside Qt. Run under a virtual display
   (see `Headless and Colab`_).

Real-time factor vs. redraw rate
--------------------------------

This is the key mental model for the render clock, and the two knobs are
**independent**:

* **Real-time factor (RTF)** -- how many sim-seconds are simulated per
  wall-clock second in the human modes. ``1.0`` is real time, ``5.0`` is 5x
  faster, ``float("inf")`` disables pacing. RTF only affects *pacing*; it never
  changes the physics or the rgb_array output.
* **Redraw rate (** ``render_fps`` **)** -- how often the screen is actually
  redrawn. In the human modes this is a wall-clock cap: the window redraws at
  **most** ``render_fps`` times per real second, no matter how fast you step.
  In ``rgb_array`` mode it instead sets the distinct-frame cadence in *sim*
  time (a fresh frame is grabbed every ``1/render_fps`` sim-seconds; the cached
  frame is returned in between).

.. note::

   ``"unlimited"`` uncaps the **sim speed** (RTF = ``inf``) but redraws are
   still capped at ``render_config.render_fps``. Running free does not force the
   GPU to draw more frames -- it just stops the loop from sleeping between steps.

You can change the RTF at runtime without rebuilding the env:

.. code-block:: python

   env.unwrapped.set_real_time_factor(5.0)          # 5x real time
   env.unwrapped.set_real_time_factor(float("inf")) # free-run
   print(env.unwrapped.real_time_factor)            # read it back
   print(env.unwrapped.render_fps)                  # target fixed frame rate
   print(env.unwrapped.frame_is_new)                # True on distinct-frame steps

``set_real_time_factor`` rejects non-positive values (``inf`` is allowed) and
re-anchors the pacer so there is no timing jump on change.

.. note::

   The env's ``metadata["render_fps"]`` is **not** ``render_config.render_fps``.
   It is set to ``round(1/timestep)`` (100 for the default 0.01 s timestep) so
   that ``gymnasium.wrappers.RecordVideo`` writes a container that plays back at
   real time -- one frame is captured per step. ``render_config.render_fps`` is
   the separate display/emit cadence.

RenderConfig
------------

Visual and pacing options live on
:class:`~f1tenth_gym.envs.env_config.RenderConfig`, the single rendering config
(the old ``RenderSpec`` was folded into it). It is a frozen dataclass; mutate it
with ``with_updates`` and nest it under
:class:`~f1tenth_gym.envs.env_config.EnvConfig`.

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Field
     - Default
     - Meaning
   * - ``render_fps``
     - ``60``
     - Display cap (human) / distinct-frame cadence in sim time (rgb_array).
   * - ``real_time_factor``
     - ``1.0``
     - Sim-seconds per wall-second in human modes (``inf`` = free-run).
   * - ``window_size``
     - ``800``
     - Square window size in pixels; also the rgb_array / video resolution.
   * - ``focus_on``
     - ``"agent_0"``
     - Agent id the camera follows; ``None`` = zoomed-out map view.
   * - ``vehicle_palette``
     - 10 hex colours
     - Per-agent car colours, cycled by agent index.
   * - ``show_wheels``
     - ``True``
     - Draw the wheels on each car.
   * - ``render_map_img``
     - ``True``
     - Draw the occupancy-map image under the scene.
   * - ``car_thickness``
     - ``1``
     - Car outline thickness in pixels.
   * - ``bigger_car_when_map_centered``
     - ``True``
     - Scale cars up in the zoomed-out map view.
   * - ``show_lap_info``
     - ``True``
     - Overlay the lap-time / lap-count label on the frame.

``render_fps``, ``real_time_factor``, and ``window_size`` are validated in
``__post_init__`` (must be positive; ``real_time_factor`` may be ``inf``).

.. code-block:: python

   from f1tenth_gym.envs.env_config import EnvConfig, RenderConfig

   cfg = EnvConfig(
       render_config=RenderConfig(
           window_size=1000,
           real_time_factor=2.0,
           focus_on=None,        # full-map camera instead of chase cam
           show_wheels=False,
       ),
   )

To change the render config on a live env, build a new config and call
``configure``:

.. code-block:: python

   new_cfg = cfg.with_updates(
       render_config=cfg.render_config.with_updates(real_time_factor=3.0),
   )
   env.unwrapped.configure(new_cfg)

The orientation is north-up: world +x maps to screen right and +y to up, so the
view matches ``plt.plot(xs, ys)`` of the raceline. Cars turn red while colliding
and revert to their palette colour once clear.

rgb_array and video recording
------------------------------

In ``rgb_array`` mode, ``env.render()`` returns a contiguous ``(H, W, 3)``
``uint8`` array pinned to the square ``window_size``. It never sleeps, so it is
the mode to use for offline rollouts and video.

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig, RenderConfig

   cfg = EnvConfig(
       simulation_config=SimulationConfig(max_laps=None),
       render_config=RenderConfig(window_size=600),
       render_enabled=True,
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="rgb_array")
   obs, info = env.reset(seed=42)

   frame = None
   for _ in range(50):
       action = np.array([[0.0, 4.0]], dtype=np.float32)
       obs, reward, terminated, truncated, info = env.step(action)
       frame = env.render()          # (600, 600, 3) uint8
       if terminated or truncated:
           break
   env.close()

   print(frame.shape, frame.dtype)   # (600, 600, 3) uint8

Pair ``rgb_array`` with ``gymnasium.wrappers.RecordVideo`` to write an MP4. The
recorded resolution is ``window_size`` and the container fps is the env's
``metadata["render_fps"]`` (real-time playback).

.. code-block:: python

   import gymnasium as gym
   from gymnasium.wrappers import RecordVideo
   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig

   cfg = EnvConfig(simulation_config=SimulationConfig(max_laps=None))
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="rgb_array")
   env = RecordVideo(env, video_folder="videos")
   # ... reset / step / render loop ...

.. warning::

   ``RecordVideo`` requires **moviepy**, which is not declared as a dependency
   of this project, and ``uv sync`` actively uninstalls it. Run
   ``uv pip install moviepy`` after syncing, or you will get
   ``gym.error.DependencyNotInstalled``.

Render callbacks
----------------

You can draw extra geometry on top of the scene by registering a callback with
``env.unwrapped.add_render_callback(fn)``, where ``fn(env_renderer) -> None`` is
called once per ``render()``. The callback reads the last observation from
``env_renderer.obs`` and uses the renderer's drawing helpers (e.g.
``env_renderer.get_points_renderer(points, color=..., size=...)``).

A ready-made one, ``make_lidar_scan_callback``, renders an agent's LiDAR scan as
a point cloud. It reads the agent's ``scan`` and ``std_state`` fields, converts
the polar ranges into world-frame points, and updates a persistent point-cloud
renderer each frame:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig
   from f1tenth_gym.envs.rendering import make_lidar_scan_callback

   cfg = EnvConfig(simulation_config=SimulationConfig(max_laps=None))
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg, render_mode="human")

   scan_cb = make_lidar_scan_callback(
       "agent_0", cfg.lidar_config, color=(255, 0, 0), size=3, subsample=4
   )
   env.unwrapped.add_render_callback(scan_cb)

   obs, info = env.reset(seed=42)
   for _ in range(1000):
       obs, reward, terminated, truncated, info = env.step(
           np.array([[0.0, 4.0]], dtype=np.float32)
       )
       env.render()
       if terminated or truncated:
           break
   env.close()

.. note::

   ``make_lidar_scan_callback`` takes the environment's
   :class:`~f1tenth_gym.envs.lidar.LiDARConfig` (``cfg.lidar_config``) because
   it precomputes the beam angles from ``angle_min``/``angle_max``/``num_beams``
   and the LiDAR mounting offset. ``subsample`` renders every Nth beam for
   performance. Callbacks are only invoked when a renderer exists
   (``render_enabled=True`` and a display render mode).

Headless and Colab
-------------------

Because the backend is OpenGL, a headless server needs a virtual X display. The
simplest route is ``xvfb-run``:

.. code-block:: bash

   xvfb-run -a python your_script.py

``rgb_array`` grabs run under ``QT_QPA_PLATFORM=xcb`` against the (virtual or
real) X server. On Google Colab, start a virtual display before creating the
env, then use ``rgb_array`` and embed the frames/video inline (there is no live
GUI window in Colab):

.. code-block:: bash

   !apt-get -qq install -y xvfb
   !pip -q install pyvirtualdisplay

.. code-block:: python

   from pyvirtualdisplay import Display
   Display(visible=0, size=(800, 800)).start()
   # ... then create the env with render_mode="rgb_array" and record/embed frames

CI runs the test suite under ``xvfb-run``; renderer tests skip automatically
when there is no ``$DISPLAY``.

A standalone dashboard
----------------------

For live telemetry outside the gym window, ``examples/telemetry_plot.py`` shows
a standalone ``pyqtgraph`` dashboard live-plotting speed, steering, yaw-rate and
slip while a plain-numpy pure-pursuit follower drives. It is **not** wired into
the gym renderer -- it is a lift-into-your-own-code pattern where the plot
refreshes at ``--fps`` while the dynamics advance by ``--rtf`` (multiple physics
steps per frame). ``examples/waypoint_follow.py`` is the canonical example of
driving the ``"human"`` renderer with a planner's own render callbacks.

See also
--------

* :doc:`configuration` -- the full :class:`~f1tenth_gym.envs.env_config.EnvConfig` tree.
* :doc:`observations` -- the ``scan`` and ``std_state`` fields the LiDAR callback reads.
* :doc:`examples` -- runnable scripts including video recording and telemetry.
* :doc:`reproducibility` -- seeding ``reset(seed=...)`` for deterministic rollouts.
