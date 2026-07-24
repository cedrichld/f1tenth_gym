import unittest

import numpy as np
from f1tenth_gym.envs.track import cubic_spline


class TestCubicSpline(unittest.TestCase):
    def test_calc_curvature(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSplineND(circle_x, circle_y)
        # Test the curvature at the four corners of the circle
        # The curvature of a circle is 1/radius
        self.assertAlmostEqual(track.calc_curvature(0), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(np.pi / 2), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(np.pi), 1, places=3)
        self.assertAlmostEqual(track.calc_curvature(3 * np.pi / 2), 1, places=3)

    def test_calc_yaw(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSplineND(circle_x, circle_y)
        # Test the yaw at the four corners of the circle
        # The yaw of a circle is s + pi/2
        self.assertAlmostEqual(track.calc_yaw(0), np.pi / 2, places=2)
        self.assertAlmostEqual(track.calc_yaw(np.pi / 2), -np.pi, places=2)
        self.assertAlmostEqual(track.calc_yaw(np.pi), -np.pi / 2, places=2)
        self.assertAlmostEqual(track.calc_yaw(3 * np.pi / 2), 0, places=2)

    def test_calc_position(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSplineND(circle_x, circle_y)
        # Test the position at the four corners of the circle
        # The position of a circle is (x, y) = (cos(s), sin(s))
        self.assertTrue(
            np.allclose(track.calc_position(0), np.array([1, 0]), atol=1e-3)
        )
        self.assertTrue(
            np.allclose(track.calc_position(np.pi / 2), np.array([0, 1]), atol=1e-3)
        )
        self.assertTrue(
            np.allclose(track.calc_position(np.pi), np.array([-1, 0]), atol=1e-3)
        )
        self.assertTrue(
            np.allclose(
                track.calc_position(3 * np.pi / 2), np.array([0, -1]), atol=1e-3
            )
        )

    def test_calc_arclength_slow(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSplineND(circle_x, circle_y)
        # Test the arclength at the four corners of the circle
        self.assertAlmostEqual(track.calc_arclength_slow(1, 0, 0)[0], 0, places=2)
        self.assertAlmostEqual(track.calc_arclength_slow(0, 1, 0)[0], np.pi / 2, places=2)
        self.assertAlmostEqual(
            track.calc_arclength_slow(-1, 0, np.pi / 2)[0], np.pi, places=2
        )
        self.assertAlmostEqual(
            track.calc_arclength_slow(0, -1, np.pi)[0], 3 * np.pi / 2, places=2
        )

    def test_calc_arclength(self):
        circle_x = np.cos(np.linspace(0, 2 * np.pi, 100))[:-1]
        circle_y = np.sin(np.linspace(0, 2 * np.pi, 100))[:-1]
        track = cubic_spline.CubicSplineND(circle_x, circle_y)
        # Test the arclength at the four corners of the circle
        self.assertAlmostEqual(track.calc_arclength(1, 0)[0], 0, places=2)
        self.assertAlmostEqual(
            track.calc_arclength(-1, 0)[0], np.pi, places=2
        )
        self.assertAlmostEqual(
            track.calc_arclength(-1, 0)[0], np.pi, places=2
        )
        self.assertAlmostEqual(
            track.calc_arclength(0, -1)[0], 3 * np.pi / 2, places=2
        )


class TestCombinedEval(unittest.TestCase):
    def test_calc_position_and_yaw_matches_separate(self):
        """The fused position+yaw evaluation (hot path in cartesian_to_frenet)
        must match calling calc_position and calc_yaw separately."""
        t = np.linspace(0, 2 * np.pi, 120)[:-1]
        # a non-circular closed loop so x/y/yaw all vary
        spline = cubic_spline.CubicSplineND(
            2.0 * np.cos(t) + 0.3 * np.cos(3 * t),
            1.5 * np.sin(t),
        )
        for s in np.linspace(0.05, spline.s_frame_max - 0.05, 200):
            s = float(s)
            x0, y0 = spline.calc_position(s)
            yaw0 = spline.calc_yaw(s)
            x1, y1, yaw1 = spline.calc_position_and_yaw(s)
            self.assertAlmostEqual(x0, x1, places=12)
            self.assertAlmostEqual(y0, y1, places=12)
            self.assertAlmostEqual(yaw0, yaw1, places=12)
