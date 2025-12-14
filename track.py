from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class Track:
    """Closed 2D track represented by a periodic centerline and constant width."""

    centerline: np.ndarray
    s: np.ndarray
    width: float
    length: float
    x_spline: CubicSpline
    y_spline: CubicSpline

    @classmethod
    def from_centerline(cls, centerline: np.ndarray, width: float) -> "Track":
        """Create a track from centerline samples (assumed to describe a loop)."""
        cl = np.asarray(centerline, dtype=float)
        if cl.ndim != 2 or cl.shape[1] != 2:
            raise ValueError("Centerline must be of shape (N, 2).")

        # Ensure closure.
        if np.linalg.norm(cl[0] - cl[-1]) > 1e-6:
            cl = np.vstack([cl, cl[0]])

        ds = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        length = float(s[-1])

        x_spline = CubicSpline(s, cl[:, 0], bc_type="periodic")
        y_spline = CubicSpline(s, cl[:, 1], bc_type="periodic")

        return cls(centerline=cl, s=s, width=float(width), length=length, x_spline=x_spline, y_spline=y_spline)

    @classmethod
    def default(cls, num_points: int = 400, width: float = 10.0, radius: float = 50.0) -> "Track":
        """Generate a smooth, closed procedural track."""
        theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        radial = radius * (1.0 + 0.15 * np.sin(3.0 * theta) + 0.07 * np.cos(5.0 * theta + 0.5))
        x = radial * np.cos(theta)
        y = radial * np.sin(theta)
        centerline = np.column_stack((x, y))
        return cls.from_centerline(centerline, width)

    def _wrap_s(self, s_query: np.ndarray) -> np.ndarray:
        """Wrap arc length queries onto the closed interval [0, length)."""
        return np.mod(s_query, self.length)

    def position(self, s_query: np.ndarray) -> np.ndarray:
        """Return XY positions along the centerline for the given arc lengths."""
        s_mod = self._wrap_s(np.asarray(s_query, dtype=float))
        x = self.x_spline(s_mod)
        y = self.y_spline(s_mod)
        return np.column_stack((x, y))

    def tangent(self, s_query: np.ndarray) -> np.ndarray:
        """Return unit tangent vectors along the centerline."""
        s_mod = self._wrap_s(np.asarray(s_query, dtype=float))
        dx = self.x_spline(s_mod, 1)
        dy = self.y_spline(s_mod, 1)
        tangents = np.column_stack((dx, dy))
        norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
        return tangents / norms

    def normal(self, s_query: np.ndarray) -> np.ndarray:
        """Return outward normals (left-hand side) along the centerline."""
        tangents = self.tangent(s_query)
        return np.column_stack((-tangents[:, 1], tangents[:, 0]))

    def boundaries(self, num_samples: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return sampled left/right boundaries and centerline for plotting."""
        s_grid = np.linspace(0.0, self.length, num_samples, endpoint=False)
        center = self.position(s_grid)
        normals = self.normal(s_grid)
        left = center + (self.width / 2.0) * normals
        right = center - (self.width / 2.0) * normals
        return left, right, center
