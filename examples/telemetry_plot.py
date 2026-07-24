"""Standalone real-time telemetry dashboard for f1tenth_gym.

Live-plots vehicle state (speed, steering, yaw rate, slip angle) in a
`pyqtgraph <https://www.pyqtgraph.org/>`_ window while a simple pure-pursuit
follower drives a lap. This is intentionally *not* wired into the gym's own
renderer -- it is a self-contained example of low-latency plotting you can
lift into your own training/eval scripts. Nothing here imports the OpenGL
renderer, so it runs without a GPU (a display/X server is still needed for the
Qt window; use ``xvfb-run`` for a virtual one).

The GUI refreshes at a fixed rate (``--fps``) while the dynamics advance by a
configurable real-time factor (``--rtf``) -- i.e. the sim can run several
physics steps per plotted frame, decoupling dynamics speed from plot refresh.

Run::

    python examples/telemetry_plot.py --map Spielberg --rtf 1.0

Dependencies: ``pyqtgraph`` and a Qt binding (PyQt5/PySide2/PyQt6/PySide6).
"""
from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import gymnasium as gym
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig


# std_state layout: [X, Y, steering, speed, yaw, yaw_rate, beta]
_SPEED, _STEER, _YAW_RATE, _BETA = 3, 2, 5, 6


class PurePursuitFollower:
    """Minimal pure-pursuit path follower over the track raceline.

    Plain NumPy (no numba) so the example stays dependency-light and robust.
    """

    def __init__(self, raceline, wheelbase: float, lookahead: float = 1.5,
                 speed_gain: float = 0.6):
        self.xs = np.asarray(raceline.xs, dtype=np.float64)
        self.ys = np.asarray(raceline.ys, dtype=np.float64)
        self.vxs = np.asarray(raceline.vxs, dtype=np.float64)
        self.wheelbase = wheelbase
        self.lookahead = lookahead
        self.speed_gain = speed_gain
        self._pts = np.stack([self.xs, self.ys], axis=1)

    def plan(self, x: float, y: float, theta: float) -> tuple[float, float]:
        # nearest waypoint
        d = np.hypot(self.xs - x, self.ys - y)
        i = int(np.argmin(d))
        n = len(self.xs)
        # walk forward until we're at least `lookahead` metres ahead
        j = i
        acc = 0.0
        while acc < self.lookahead:
            nxt = (j + 1) % n
            acc += float(np.hypot(self.xs[nxt] - self.xs[j], self.ys[nxt] - self.ys[j]))
            j = nxt
            if j == i:  # wrapped the whole loop
                break
        # lookahead point in the vehicle frame
        dx, dy = self.xs[j] - x, self.ys[j] - y
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)
        y_local = sin_t * dx + cos_t * dy
        ld = max(self.lookahead, 1e-3)
        steer = np.arctan2(2.0 * self.wheelbase * y_local, ld * ld)
        speed = self.speed_gain * float(self.vxs[i])
        return float(steer), speed


class TelemetryDashboard:
    def __init__(self, env, follower, window_s: float, fps: float, rtf: float):
        self.env = env
        self.follower = follower
        self.dt = env.unwrapped.timestep
        self.window = int(window_s / self.dt)
        self.steps_per_tick = max(1, round(rtf * (1.0 / fps) / self.dt))

        self.t = deque(maxlen=self.window)
        self.speed = deque(maxlen=self.window)
        self.steer = deque(maxlen=self.window)
        self.yaw_rate = deque(maxlen=self.window)
        self.slip = deque(maxlen=self.window)

        self.win = pg.GraphicsLayoutWidget(show=True, title="F1TENTH Telemetry")
        self.win.resize(920, 720)
        self.win.setWindowTitle("F1TENTH Telemetry")

        def _plot(row, title, ylabel, color):
            p = self.win.addPlot(row=row, col=0, title=title)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("left", ylabel)
            p.setLabel("bottom", "time", units="s")
            return p.plot(pen=pg.mkPen(color, width=2))

        self.c_speed = _plot(0, "Speed", "m/s", "#4CAF50")
        self.c_steer = _plot(1, "Steering angle", "deg", "#2196F3")
        self.c_yaw = _plot(2, "Yaw rate", "deg/s", "#FF9800")
        self.c_slip = _plot(3, "Slip angle (beta)", "deg", "#E91E63")

        self.obs, _ = env.reset()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000.0 / fps))

    def update(self):
        for _ in range(self.steps_per_tick):
            s = self.obs["agent_0"]["std_state"]
            steer_cmd, speed_cmd = self.follower.plan(float(s[0]), float(s[1]), float(s[4]))
            self.obs, _, terminated, truncated, _ = self.env.step(
                np.array([[steer_cmd, speed_cmd]], dtype=np.float32)
            )
            st = self.obs["agent_0"]["std_state"]
            self.t.append(float(self.obs["agent_0"]["sim_time"]))
            self.speed.append(float(st[_SPEED]))
            self.steer.append(np.rad2deg(float(st[_STEER])))
            self.yaw_rate.append(np.rad2deg(float(st[_YAW_RATE])))
            self.slip.append(np.rad2deg(float(st[_BETA])))
            if terminated or truncated:
                self.obs, _ = self.env.reset()

        t = list(self.t)
        self.c_speed.setData(t, list(self.speed))
        self.c_steer.setData(t, list(self.steer))
        self.c_yaw.setData(t, list(self.yaw_rate))
        self.c_slip.setData(t, list(self.slip))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="Spielberg", help="track name")
    ap.add_argument("--rtf", type=float, default=1.0,
                    help="real-time factor: dynamics steps run this much faster "
                         "than the fixed plot refresh (>1 = faster than real time)")
    ap.add_argument("--fps", type=float, default=50.0, help="plot refresh rate")
    ap.add_argument("--window", type=float, default=10.0,
                    help="seconds of history shown")
    args = ap.parse_args()

    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            map_name=args.map,
            simulation_config=SimulationConfig(max_laps=None),
            render_enabled=False,
        ),
    )
    params = env.unwrapped.sim.vehicle_params
    follower = PurePursuitFollower(
        env.unwrapped.track.raceline, wheelbase=float(params.lf + params.lr)
    )

    app = pg.mkQApp("F1TENTH Telemetry")
    # keep a reference alive: the dashboard owns the driving QTimer, so letting
    # it be garbage-collected would silently stop the updates.
    dash = TelemetryDashboard(env, follower, args.window, args.fps, args.rtf)  # noqa: F841
    try:
        app.exec()  # pyqtgraph>=0.13 (Qt6-compatible); older uses exec_()
    except AttributeError:  # pragma: no cover
        app.exec_()
    env.close()


if __name__ == "__main__":
    main()
