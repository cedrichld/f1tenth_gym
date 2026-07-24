from __future__ import annotations
import pathlib
from typing import Optional

import numpy as np

from ..rendering import EnvRenderer
from .cubic_spline import CubicSplineND


def _read_labeled_csv(filepath, delimiter: str, header_row: int = 0) -> dict:
    """Read a numeric CSV whose column names are on line ``header_row``.

    Column names are cleaned like the old pandas path (``#`` removed, whitespace
    stripped); returns ``{clean_name: float32 column}``. Replaces
    ``pandas.read_csv`` for the small, fixed-schema raceline/centerline files.
    """
    with open(filepath, "r") as f:
        header = f.readlines()[header_row]
    names = [c.replace("#", "").strip() for c in header.strip().split(delimiter)]
    data = np.loadtxt(
        filepath, delimiter=delimiter, skiprows=header_row + 1,
        dtype=np.float32, ndmin=2,
    )
    return {name: data[:, i] for i, name in enumerate(names)}


class Raceline:
    """
    Raceline object.

    Attributes
    ----------
    n : int
        number of waypoints
    ss : np.ndarray
        arclength along the raceline
    xs : np.ndarray
        x-coordinates of the waypoints
    ys : np.ndarray
        y-coordinates of the waypoints
    yaws : np.ndarray
        yaw angles of the waypoints
    ks : np.ndarray
        curvature of the waypoints
    vxs : np.ndarray
        velocity along the raceline
    axs : np.ndarray
        acceleration along the raceline
    length : float
        length of the raceline
    spline : CubicSplineND
        spline object through the waypoints
    """

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        velxs: np.ndarray,
        ss: Optional[np.ndarray] = None,
        psis: Optional[np.ndarray] = None,
        kappas: Optional[np.ndarray] = None,
        accxs: Optional[np.ndarray] = None,
        spline: Optional[CubicSplineND] = None,
    ):
        assert xs.shape == ys.shape == velxs.shape, "inconsistent shapes for x, y, vel"

        self.n = xs.shape[0]
        self.ss = ss
        self.xs = xs
        self.ys = ys
        self.yaws = psis
        self.ks = kappas
        self.vxs = velxs
        self.axs = accxs

        # approximate track length by linear-interpolation of x,y waypoints
        # note: we could use 'ss' but sometimes it is normalized to [0,1], so we recompute it here
        self.length = float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))

        # compute spline through waypoints if not provided
        self.spline = spline or CubicSplineND(x=xs, y=ys,
                                            psis=psis,
                                            ks=kappas,
                                            vxs=velxs,
                                            axs=accxs,
                                            ss=ss)
        self.s_frame_max = self.spline.s[-1]

        self.waypoint_renderer = None

    @staticmethod
    def from_centerline_file(
        filepath: pathlib.Path,
        delimiter: Optional[str] = ",",
        fixed_speed: Optional[float] = 1.0,
        track_scale: Optional[float] = 1.0,
    ):
        """
        Load raceline from a centerline file.

        Parameters
        ----------
        filepath : pathlib.Path
            path to the centerline file
        delimiter : str, optional
            delimiter used in the file, by default ","
        fixed_speed : float, optional
            fixed speed along the raceline, by default 1.0
        track_scale : float, optional
            scaling factor for the track, by default 1.0

        Returns
        -------
        Raceline
            raceline object
        """
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        cols = _read_labeled_csv(filepath, delimiter=delimiter, header_row=0)

        # Required columns
        required_cols = ["x_m", "y_m"]
        missing_cols = [col for col in required_cols if col not in cols]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in raceline file: {missing_cols}\n"
                f"Available columns: {list(cols)}"
            )

        # fit cubic spline to waypoints
        xx, yy = cols["x_m"], cols["y_m"]
        # scale waypoints
        xx, yy = xx * track_scale, yy * track_scale
        
        # Velocity is constant along the raceline
        vx = np.ones_like(xx) * fixed_speed
        # close loop
        spline = CubicSplineND(x=xx, y=yy, vxs=vx)

        return Raceline(
            ss=spline.ss,
            xs=spline.xs,
            ys=spline.ys,
            psis=spline.psis,
            kappas=spline.ks,
            velxs=spline.vxs,
            accxs=spline.axs,
            spline=spline,
        )

    @staticmethod
    def from_raceline_file(filepath: pathlib.Path, delimiter: str = ";", skip_rows: int = 3, track_scale: Optional[float] = 1.0) -> Raceline:
        """
        Load raceline from a raceline file of the format [s, x, y, psi, k, vx, ax].

        Parameters
        ----------
        filepath : pathlib.Path
            path to the raceline file
        delimiter : str, optional
            delimiter used in the file, by default ";"
        track_scale : float, optional
            scaling factor for the track, by default 1.0

        Returns
        -------
        Raceline
            raceline object
        """
        if type(filepath) is str:
            filepath = pathlib.Path(filepath)

        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        cols = _read_labeled_csv(filepath, delimiter=delimiter, header_row=skip_rows - 1)

        # Required columns
        required_cols = ["x_m", "y_m", "psi_rad", "vx_mps"]
        missing_cols = [col for col in required_cols if col not in cols]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in raceline file: {missing_cols}\n"
                f"Available columns: {list(cols)}"
            )

        if track_scale != 1.0:
            # scale x-y waypoints and recalculate s, psi, and k
            cols["x_m"] = cols["x_m"] * track_scale
            cols["y_m"] = cols["y_m"] * track_scale
            spline = CubicSplineND(x=cols["x_m"], y=cols["y_m"])
            ss, yaws, ks = spline.ss, spline.psis, spline.ks
            cols["psi_rad"] = yaws
            if "kappa_radpm" in cols:
                cols["kappa_radpm"] = ks
            if "s_m" in cols:
                cols["s_m"] = ss

        return Raceline(
            ss=cols["s_m"],
            xs=cols["x_m"],
            ys=cols["y_m"],
            psis=cols["psi_rad"] if "psi_rad" in cols else None,
            kappas=cols["kappa_radpm"] if "kappa_radpm" in cols else None,
            velxs=cols["vx_mps"] if "vx_mps" in cols else None,
            accxs=cols["ax_mps2"] if "ax_mps2" in cols else None,
        )

    def render_waypoints(self, e: EnvRenderer) -> None:
        """
        Callback to render waypoints.

        Parameters
        ----------
        e : EnvRenderer
            Environment renderer object.
        """
        points = np.stack([self.xs, self.ys], axis=1)
        if self.waypoint_renderer is None:
            self.waypoint_renderer = e.get_closed_lines_renderer(points, color=(0, 255, 0), size=2)
        else:
            self.waypoint_renderer.update(points)
