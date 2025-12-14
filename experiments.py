import csv
import json
import os
from typing import Dict, List, Sequence, Tuple
import numpy as np

from ga import GAConfig, GeneticAlgorithm
from fitness import evaluate_individual
from track import Track
from plots import plot_acceleration_analysis


def run_experiments(
    track: Track | None = None,
    mu_values: Sequence[float] = (0.6, 0.9, 1.2),
    runs: int = 20,
    base_seed: int = 1234,
    output_dir: str = "results",
    show_progress: bool = True,
) -> Tuple[List[Dict], List[Dict], Dict[float, Dict]]:
    """Run GA experiments for each friction coefficient."""
    track = track or Track.default()
    os.makedirs(output_dir, exist_ok=True)

    def render_progress_bar(completed: int, total: int, width: int = 30) -> str:
        fraction = completed / total if total > 0 else 0.0
        filled = int(fraction * width)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {completed}/{total}"

    num_control_points = 20
    path_resolution = 1.0
    speed_params = {"g": 9.81, "v_max": 80.0, "a_engine": 6.0, "a_brake": 8.0, "v_min": 1.0}
    ga_config = GAConfig(
        pop_size=40,
        crossover_rate=0.9,
        mutation_sigma=0.5,
        tournament_k=3,
        elite_size=2,
        evaluation_budget=2000,
        bounds=(-track.width / 2.0, track.width / 2.0),
    )

    aggregated_stats: List[Dict] = []
    per_run_rows: List[Dict] = []
    best_trajectories: Dict[float, Dict] = {}
    total_jobs = len(mu_values) * runs
    completed_jobs = 0
    if show_progress and total_jobs > 0:
        bar = render_progress_bar(completed_jobs, total_jobs)
        print(f"{bar} | initializing...", end="", flush=True)

    for mu_idx, mu in enumerate(mu_values):
        times: List[float] = []
        mu_best: Dict | None = None
        for run_idx in range(runs):
            seed = int(base_seed + mu_idx * 1000 + run_idx)
            rng = np.random.default_rng(seed)

            def fitness_fn(offset_vector: np.ndarray):
                return evaluate_individual(offset_vector, track, mu=mu, path_resolution=path_resolution, speed_params=speed_params)

            ga = GeneticAlgorithm(num_genes=num_control_points, fitness_fn=fitness_fn, config=ga_config, rng=rng)
            best_vector, best_score, best_info = ga.run()

            times.append(best_info["lap_time"])
            per_run_rows.append(
                {
                    "mu": mu,
                    "run": run_idx,
                    "seed": seed,
                    "lap_time": float(best_info["lap_time"]),
                    "penalty": float(best_info["penalty"]),
                }
            )

            if mu_best is None or best_info["lap_time"] < mu_best["lap_time"]:
                mu_best = {
                    "lap_time": best_info["lap_time"],
                    "trajectory": best_info["trajectory"],
                    "offsets": best_vector,
                    "speed_profile": best_info["speed_profile"],
                    "curvature": best_info["curvature"],
                }

            completed_jobs += 1
            if show_progress:
                bar = render_progress_bar(completed_jobs, total_jobs)
                print(
                    f"\r{bar} | mu={mu:.2f} run {run_idx + 1}/{runs} best_lap={best_info['lap_time']:.2f}s",
                    end="",
                    flush=True,
                )

        stats = {
            "mu": mu,
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "best": float(np.min(times)),
            "worst": float(np.max(times)),
            "runs": runs,
        }
        aggregated_stats.append(stats)
        if mu_best is not None:
            best_trajectories[mu] = mu_best
            traj = mu_best["trajectory"]
            prefix = f"mu_{mu:.2f}".replace(".", "p")
            ds_segments = np.diff(traj.s) if len(traj.s) > 1 else np.array([0.0])
            plot_acceleration_analysis(
                x=traj.path[:, 0],
                y=traj.path[:, 1],
                v=mu_best["speed_profile"],
                kappa=mu_best["curvature"],
                ds=ds_segments,
                mu=mu,
                g=speed_params["g"],
                out_dir=output_dir,
                prefix=prefix,
            )

    if show_progress:
        print()  # finish progress line

    per_run_csv = os.path.join(output_dir, "per_run_results.csv")
    with open(per_run_csv, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["mu", "run", "seed", "lap_time", "penalty"])
        writer.writeheader()
        for row in per_run_rows:
            writer.writerow(row)

    aggregated_csv = os.path.join(output_dir, "aggregated_results.csv")
    with open(aggregated_csv, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["mu", "mean", "std", "best", "worst", "runs"])
        writer.writeheader()
        for row in aggregated_stats:
            writer.writerow(row)

    aggregated_json = os.path.join(output_dir, "aggregated_results.json")
    with open(aggregated_json, "w") as f_json:
        json.dump({"mu_values": list(mu_values), "stats": aggregated_stats, "per_run": per_run_rows}, f_json, indent=2)

    return aggregated_stats, per_run_rows, best_trajectories
