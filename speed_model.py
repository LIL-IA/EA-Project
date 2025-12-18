import numpy as np
from scipy.interpolate import CubicSpline


def compute_curvature(path: np.ndarray, s: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Estimate curvature kappa(s) along a path using periodic cubic splines."""
    s = np.asarray(s, dtype=float)
    path = np.asarray(path, dtype=float)
    csx = CubicSpline(s, path[:, 0], bc_type="periodic")
    csy = CubicSpline(s, path[:, 1], bc_type="periodic")
    dx = csx(s, 1); dy = csy(s, 1)
    ddx = csx(s, 2); ddy = csy(s, 2)
    denom = np.power(dx * dx + dy * dy, 1.5) + eps
    return (dx * ddy - dy * ddx) / denom


def compute_speed_profile(
    s: np.ndarray,
    kappa: np.ndarray,
    mu: float,
    g: float = 9.81,
    v_max: float = 80.0,
    a_engine: float = 6.0,
    a_brake: float = 8.0,
    v_min: float = 1.0,
    smooth_iters: int = 2,
    smooth_alpha: float = 0.5,
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
        smooth_iters: number of smoothing + constraint iterations applied to reduce acceleration spikes.
        smooth_alpha: blend factor (0-1) for the smoothing step; higher means stronger smoothing.

    Returns:
        tuple of (speed profile v, curvature-limited cap v_curve, lap_time).
    """
    s = np.asarray(s, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    if s.shape != kappa.shape:
        raise ValueError("s and kappa must have the same shape.")

    smooth_iters = max(int(smooth_iters), 0)
    smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))

    ds = np.diff(s)
    v_curve = np.sqrt(np.maximum(0.0, (mu * g) / (np.abs(kappa) + 1e-6)))
    v_curve = np.minimum(v_curve, v_max)
    v = v_curve.copy()

    def _apply_forward_backward(v_guess: np.ndarray) -> np.ndarray:
        """Run forward/backward passes enforcing engine/brake and friction limits."""
        v_fb = v_guess.copy()

        # Forward pass (acceleration limited).
        for i in range(len(ds)):
            if ds[i] <= 0.0:
                v_fb[i + 1] = min(v_fb[i + 1], v_curve[i + 1])
                continue
            a_lat = v_fb[i] ** 2 * abs(kappa[i])
            a_long_max = np.sqrt(max(0.0, (mu * g) ** 2 - a_lat ** 2))
            a_acc = min(a_engine, a_long_max)
            v_next = np.sqrt(max(0.0, v_fb[i] ** 2 + 2.0 * a_acc * ds[i]))
            v_fb[i + 1] = min(v_curve[i + 1], v_fb[i + 1], v_next)

        # Backward pass (braking limited).
        for i in range(len(ds) - 1, -1, -1):
            if ds[i] <= 0.0:
                v_fb[i] = min(v_fb[i], v_curve[i])
                continue
            a_lat = v_fb[i + 1] ** 2 * abs(kappa[i + 1])
            a_long_max = np.sqrt(max(0.0, (mu * g) ** 2 - a_lat ** 2))
            a_dec = min(a_brake, a_long_max)
            v_allowed = np.sqrt(max(0.0, v_fb[i + 1] ** 2 + 2.0 * a_dec * ds[i]))
            v_fb[i] = min(v_fb[i], v_allowed, v_curve[i])
        return v_fb

    # Initial constraint projection.
    v = _apply_forward_backward(v)

    # Optional smoothing iterations: reduce local peaks, then re-enforce constraints.
    for _ in range(smooth_iters):
        if len(v) > 2 and smooth_alpha > 0.0:
            neighbor_avg = 0.25 * v[:-2] + 0.5 * v[1:-1] + 0.25 * v[2:]
            blended = (1.0 - smooth_alpha) * v[1:-1] + smooth_alpha * neighbor_avg
            v[1:-1] = np.minimum(v[1:-1], np.minimum(v_curve[1:-1], blended))
        v = _apply_forward_backward(v)

    lap_time = float(np.sum(ds / np.maximum(v[:-1], v_min)))
    return v, v_curve, lap_time
