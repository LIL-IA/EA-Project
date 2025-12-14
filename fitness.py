from typing import Dict, Tuple
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
    lambda_smooth: float = 0.01,
    enable_smoothness: bool = True,
) -> Tuple[float, Dict]:
    """
    Evaluate an individual: build trajectory, compute speed profile, and return lap time + regularization.

    fitness = lap_time + lambda_smooth * smoothness + other penalties
    - smoothness is a discrete second-difference penalty on lateral offsets (mu-independent).
    - Set lambda_smooth=0 or enable_smoothness=False to disable the regularizer (ablation).

    Returns:
        (fitness_value, info_dict) with lap time + penalties (lower is better).
    """
    offsets_clipped = np.clip(offsets, -track.width / 2.0, track.width / 2.0)
    trajectory: Trajectory = build_trajectory(track, offsets_clipped, path_resolution=path_resolution)
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

    # Smoothness term encourages gradual changes in lateral offsets.
    smoothness = 0.0
    if enable_smoothness and len(offsets_clipped) >= 3:
        smoothness = float(np.sum(np.diff(offsets_clipped, n=2) ** 2))

    penalty = 0.0
    if not trajectory.valid:
        penalty += 1e6

    curvature_variation = np.std(np.diff(curvature))
    penalty += 0.1 * curvature_variation

    fitness_value = lap_time + lambda_smooth * smoothness + penalty
    info = {
        "lap_time": float(lap_time),
        "penalty": float(penalty),
        "smoothness": float(smoothness),
        "lambda_smooth": float(lambda_smooth),
        "trajectory": trajectory,
        "curvature": curvature,
        "speed_profile": v_profile,
        "speed_curve": v_curve,
        "valid": trajectory.valid,
    }
    return float(fitness_value), info
