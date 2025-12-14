import numpy as np


def compute_curvature(path: np.ndarray, s: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Estimate curvature kappa(s) along a path using finite differences."""
    x = path[:, 0]
    y = path[:, 1]
    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    denom = np.power(dx * dx + dy * dy, 1.5) + eps
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa


def compute_speed_profile(
    s: np.ndarray,
    kappa: np.ndarray,
    mu: float,
    g: float = 9.81,
    v_max: float = 80.0,
    a_engine: float = 6.0,
    a_brake: float = 8.0,
    v_min: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Two-pass curvature + adhesion limited speed profile with friction circle.

    Args:
        s: cumulative arc length samples (monotonic, includes final loop closure).
        kappa: curvature values at s.
        mu: tire-road friction coefficient.
        g: gravity constant.
        v_max: hard speed cap.
        a_engine: maximum engine-driven acceleration.
        a_brake: maximum braking deceleration (positive value).
        v_min: small floor to avoid division by zero in lap time.

    Returns:
        tuple of (speed profile v, curvature-limited cap v_curve, lap_time).
    """
    s = np.asarray(s, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    if s.shape != kappa.shape:
        raise ValueError("s and kappa must have the same shape.")

    ds = np.diff(s)
    v_curve = np.sqrt(np.maximum(0.0, (mu * g) / (np.abs(kappa) + 1e-6)))
    v_curve = np.minimum(v_curve, v_max)
    v = v_curve.copy()

    # Forward pass (acceleration limited).
    for i in range(len(ds)):
        a_lat = v[i] ** 2 * abs(kappa[i])
        a_long_max = np.sqrt(max(0.0, (mu * g) ** 2 - a_lat ** 2))
        a_acc = min(a_engine, a_long_max)
        v_next = np.sqrt(max(0.0, v[i] ** 2 + 2.0 * a_acc * ds[i]))
        v[i + 1] = min(v_curve[i + 1], v_next)

    # Backward pass (braking limited).
    for i in range(len(ds) - 1, -1, -1):
        a_lat = v[i + 1] ** 2 * abs(kappa[i + 1])
        a_long_max = np.sqrt(max(0.0, (mu * g) ** 2 - a_lat ** 2))
        a_dec = min(a_brake, a_long_max)
        v_allowed = np.sqrt(max(0.0, v[i + 1] ** 2 + 2.0 * a_dec * ds[i]))
        v[i] = min(v[i], v_allowed, v_curve[i])

    lap_time = float(np.sum(ds / np.maximum(v[:-1], v_min)))
    return v, v_curve, lap_time
