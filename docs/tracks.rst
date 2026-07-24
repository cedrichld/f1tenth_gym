Tracks, maps & the Frenet frame
================================

Every :class:`~f1tenth_gym.envs.f110_env.F110Env` runs on a :class:`~f1tenth_gym.envs.track.track.Track`: an occupancy map for LiDAR/wall-collision plus one or two reference lines (a *centerline* and a *raceline*). This page explains how a track is resolved from ``map_name``, the map file convention, how to build a synthetic track from scratch, the difference between the centerline and the raceline, and the Frenet ``(s, ey, ephi)`` frame that the simulator reports.

The track is selected by :class:`~f1tenth_gym.envs.env_config.EnvConfig`'s ``map_name`` field (default ``"Spielberg"``). See :doc:`configuration` for the full config tree and :doc:`quickstart` for a first run.

How a map is resolved
---------------------

``map_name`` may be a built-in name, a path, or a live ``Track`` instance. ``F110Env._resolve_track`` dispatches on the value:

- **A bare name** (no path separators, no file suffix), e.g. ``"Spielberg"`` — resolved with :meth:`Track.from_track_name <f1tenth_gym.envs.track.track.Track.from_track_name>`, which looks the map up under the repo-root ``maps/`` directory and **downloads it if missing**.
- **A path** (contains ``/`` or ``\`` or a file suffix) — resolved with :meth:`Track.from_track_path <f1tenth_gym.envs.track.track.Track.from_track_path>` against a local track directory you provide.
- **A ``Track`` instance** — used as-is, no I/O. This is how you feed a synthetic or programmatically built track (see `Synthetic tracks`_ below).

Built-in tracks and the download
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bundled ``maps/`` directory is **git-ignored** (only ``maps/.gitkeep`` is tracked). The first time a built-in track is requested, ``find_track_dir`` fetches it from a hard-coded endpoint::

    https://api.f1tenth.org/<name>.tar.xz

and extracts it into the repo-root ``maps/`` folder (resolved by walking up from the source file, so this only works for an **editable / from-source install**). Extraction uses ``filter="data"`` (tar hardening); there is no checksum verification.

.. code-block:: python

    import gymnasium as gym
    from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig

    cfg = EnvConfig(
        map_name="Spielberg",                                # downloaded on first use
        simulation_config=SimulationConfig(max_laps=None),   # else the episode ends after 1 lap
        render_enabled=False,
    )
    env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
    obs, info = env.reset(seed=42)
    env.close()

.. note::

   The download needs network access and a writable, source-checkout ``maps/`` directory. A missing map raises ``FileNotFoundError("No maps exists for <name>.")`` on a 404. Other F1TENTH track names on the endpoint (for example ``"Monza"``, ``"Levine"``) resolve the same way.

The map file convention
-----------------------

A track directory holds a **ROS-style occupancy map**: a YAML metadata file next to a single-channel image, plus optional reference-line CSVs. ``from_track_name`` tries ``{stem}.yaml`` first, then falls back to the legacy ``{stem}_map.yaml``. The YAML (:class:`~f1tenth_gym.envs.track.track.TrackSpec`) carries ``image``, ``resolution``, ``origin``, ``negate``, ``occupied_thresh`` and ``free_thresh``.

Only a subset of that metadata actually drives the simulator:

- **resolution** — metres per pixel (multiplied by ``EnvConfig.map_scale``).
- **origin** — the world coordinate of the map's **bottom-left corner**; only ``origin[0]`` and ``origin[1]`` are used (the ``origin[2]`` rotation is ignored). ``origin[:2]`` is scaled by ``map_scale``.
- **image** — the occupancy image, loaded ``FLIP_TOP_BOTTOM`` so that grid row 0 is the smallest world ``y``.

.. warning::

   The occupancy image is binarised at a **hard-coded threshold of 128** (``<=128`` → free/0, ``>128`` → occupied/255). The YAML's ``negate``, ``occupied_thresh`` and ``free_thresh`` fields are parsed into the :class:`~f1tenth_gym.envs.track.track.TrackSpec` but **never used** — do not rely on them to change how the map is thresholded.

Reference-line CSVs are optional and looked up by name in the track directory: ``{name}_centerline.csv`` (columns ``x_m, y_m, ...``, comma-delimited) and ``{name}_raceline.csv`` (columns ``s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2``, semicolon-delimited).

.. warning::

   ``from_track_name`` sets ``centerline=None`` when ``{name}_centerline.csv`` is absent, and — unlike ``from_track_path`` — it does **not** fall back to the raceline. A track directory that ships only a raceline CSV leaves ``track.centerline is None``, and the first ``reset()`` then raises ``AttributeError: 'NoneType' object has no attribute 'spline'`` because the Frenet transform reads the centerline. Provide a centerline CSV (or supply the ``Track`` yourself with ``centerline`` set) for custom tracks.

Custom track directories
~~~~~~~~~~~~~~~~~~~~~~~~~~

Point ``map_name`` at a path (with a separator or a file suffix) to load your own map with the same ROS convention:

.. code-block:: python

    from f1tenth_gym.envs.env_config import EnvConfig

    cfg = EnvConfig(map_name="/abs/path/to/mytrack/mytrack.yaml", render_enabled=False)
    # F110Env resolves this via Track.from_track_path

Centerline vs. raceline
-----------------------

A :class:`~f1tenth_gym.envs.track.track.Track` carries two :class:`~f1tenth_gym.envs.track.raceline.Raceline` objects, each backed by a periodic cubic spline over ``[x, y, cos ψ, sin ψ, k, vx, ax]``:

- ``track.centerline`` — the geometric middle of the track. When built from a bare centerline CSV via ``from_centerline_file``, its speed profile is a **constant ``1.0`` m/s** (``fixed_speed=1.0`` is hard-coded), so ``centerline.vxs`` is *not* a usable racing speed profile.
- ``track.raceline`` — the optimized racing line, carrying a real speed profile ``raceline.vxs``.

Which one matters depends on the subsystem:

- **Reset / spawning** — every ``RL_*`` reset strategy binds to ``track.raceline`` (never the centerline). See :doc:`reproducibility` and the reset section of :doc:`configuration`. Cars spawn *on the raceline*.
- **Frenet frame** — the simulator's reported ``(s, ey, ephi)`` is always computed against the **centerline** (see below).

.. warning::

   The centerline and raceline are **different-length loops**. On Spielberg the centerline frame is ~343.3 m while the raceline is ~338.1 m. Because cars spawn on the raceline but Frenet is measured on the centerline, ``ey`` is **non-zero at spawn** (about 0.8 m on Spielberg), and FRENET_BASED lap counting carries a ~1.5% lap-length error. This is expected, not a bug in your code.

The Frenet frame ``(s, ey, ephi)``
----------------------------------

When ``SimulationConfig.compute_frenet_frame`` is ``True`` (the default), each step reports a per-agent Frenet pose in the observation as ``frenet_pose`` (a 3-vector ``(s, ey, ephi)``; see :doc:`observations`). Units are metres, metres, radians:

- **s** — arclength progress along the **centerline** spline, wrapped into ``[0, s_frame_max)``.
- **ey** — signed lateral deviation from the centerline. **``+ey`` is to the LEFT** of the direction of travel (the sign comes from the spline normal ``[-sin ψ, cos ψ]``).
- **ephi** — heading deviation ``psi - yaw_of_centerline``, wrapped to ``[-π, π)``.

.. note::

   The Frenet frame is **always centerline-based** — ``cartesian_to_frenet`` and ``frenet_to_cartesian`` default to ``use_raceline=False`` and both simulator call sites use that default. There is no config knob to measure Frenet against the raceline.

Converting coordinates directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can transform points yourself on any ``Track``:

.. code-block:: python

    from f1tenth_gym.envs.track.track import Track

    track = Track.from_track_name("Spielberg")

    # Cartesian -> Frenet: returns (s, ey, ephi)
    s, ey, ephi = track.cartesian_to_frenet(x=0.0, y=0.0, psi=0.0)

    # Frenet -> Cartesian: returns (x, y, psi)
    x, y, psi = track.frenet_to_cartesian(s=s, ey=ey, ephi=ephi)

Pass ``use_raceline=True`` to either method to measure against the raceline instead of the centerline.

Synthetic tracks
----------------

To race on a shape with no map file, build a :class:`~f1tenth_gym.envs.track.track.Track` from a reference line and pass the **instance** as ``map_name``. :meth:`Track.from_refline <f1tenth_gym.envs.track.track.Track.from_refline>` takes ``x``, ``y`` and per-waypoint velocity ``velx`` arrays, fits a periodic cubic spline, and synthesizes a free-space occupancy map sized to the extents plus a 5 m margin (both centerline and raceline are set to this same line).

.. code-block:: python

    import numpy as np
    import gymnasium as gym
    from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig
    from f1tenth_gym.envs.track.track import Track

    # A closed circle of radius 10 m, target speed 5 m/s everywhere
    theta = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
    x = 10.0 * np.cos(theta)
    y = 10.0 * np.sin(theta)
    velx = np.full_like(x, 5.0)

    track = Track.from_refline(x=x, y=y, velx=velx)

    cfg = EnvConfig(
        map_name=track,                                      # pass the Track instance directly
        simulation_config=SimulationConfig(max_laps=None),
        render_enabled=False,
    )
    env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
    obs, info = env.reset(seed=42)

    for _ in range(50):
        action = np.array([[0.0, 3.0]], dtype=np.float32)    # [[steer, speed]] — steering first
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()

.. warning::

   ``from_refline`` **force-closes the reference line into a loop** — there is no open-path mode. If the last point does not coincide with the first, an extra point is appended to close the loop (required by the periodic spline). A "straight line" input such as ``x=np.linspace(0, 10, N), y=np.zeros(N)`` therefore becomes a closed ~20 m path with a phantom return leg, and ``s``, curvature and lap counting all run over that closed loop. Design your reference line as an intentionally closed circuit.

You can also swap the map on a live env with ``env.unwrapped.update_map(track_or_name)``, or reconfigure via ``env.unwrapped.configure(cfg.with_updates(map_name=...))`` (see :doc:`configuration`).

See also
--------

- :doc:`observations` — where ``frenet_pose`` and pose fields appear in the observation dict.
- :doc:`rewards_and_rl` — PROGRESS reward uses the Frenet ``s`` and requires ``compute_frenet_frame=True``.
- :doc:`reproducibility` — how reset strategies spawn agents on the raceline.
- :doc:`api/index` — full ``Track`` / ``Raceline`` reference.
