from typing import Callable, Dict, Tuple
import numpy as np
from trajectory import build_trajectory, Trajectory
from speed_model import compute_curvature, compute_speed_profile
from track import Track


def evaluate_individual(
    offsets: np.ndarray,
    track: Track,
    mu: float,
    path_resolution: float,
    speed_params: Dict[str, float],
) -> Tuple[float, Dict]:
    """
    Evaluate an individual: build trajectory, compute speed profile, and return lap time.

    Returns:
        (fitness_value, info_dict) with lap time + penalties (lower is better).
    """
    trajectory: Trajectory = build_trajectory(track, offsets, path_resolution=path_resolution)
    curvature = compute_curvature(trajectory.path, trajectory.s)
    v_profile, v_curve, lap_time = compute_speed_profile(
        trajectory.s,
        curvature,
        mu=mu,
        g=speed_params.get("g", 9.81),
        v_max=speed_params.get("v_max", 80.0),
        a_engine=speed_params.get("a_engine", 6.0),
        a_brake=speed_params.get("a_brake", 8.0),
        v_min=speed_params.get("v_min", 1.0),
    )

    penalty = 0.0
    curvature_variation = np.std(np.diff(curvature))
    penalty += 0.1 * curvature_variation

    fitness_value = lap_time + penalty
    info = {
        "lap_time": float(lap_time),
        "penalty": float(penalty),
        "trajectory": trajectory,
        "curvature": curvature,
        "speed_profile": v_profile,
        "speed_curve": v_curve,
        "valid": trajectory.valid,
    }
    return float(fitness_value), info
