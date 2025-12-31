import csv
import json
from dataclasses import replace
from pathlib import Path
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
    num_control_points: int = 20,
    path_resolution: float = 1.0,
    speed_params: Dict[str, float] | None = None,
    ga_config: GAConfig | None = None,
    evaluation_budget: int = 5000,
    save_results: bool = True,
) -> Tuple[List[Dict], List[Dict], Dict[float, Dict]]:
    """Run GA experiments for each friction coefficient and optionally persist results."""
    track = track or Track.default()
    if len(mu_values) == 0:
        raise ValueError("mu_values must contain at least one friction coefficient.")
    if runs <= 0:
        raise ValueError("runs must be positive.")
    if num_control_points < 3:
        raise ValueError("num_control_points must be at least 3 to define a closed line.")
    if path_resolution <= 0:
        raise ValueError("path_resolution must be positive.")
    if evaluation_budget <= 0:
        raise ValueError("evaluation_budget must be positive.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_speed_params = {"g": 9.81, "v_max": 80.0, "a_engine": 6.0, "a_brake": 8.0, "v_min": 1.0}
    if speed_params:
        base_speed_params.update(speed_params)

    if ga_config is None:
        ga_config = GAConfig(
            pop_size=40,
            crossover_rate=0.9,
            mutation_sigma=0.5,
            tournament_k=3,
            elite_size=2,
            evaluation_budget=evaluation_budget,
            bounds=(-track.width / 2.0, track.width / 2.0),
        )
    else:
        ga_config = replace(ga_config)

    def render_progress_bar(completed: int, total: int, width: int = 30) -> str:
        fraction = completed / total if total > 0 else 0.0
        filled = int(fraction * width)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {completed}/{total}"
    
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
        mu_best_score: float | None = None
        for run_idx in range(runs):
            seed = int(base_seed + mu_idx * 1000 + run_idx)
            rng = np.random.default_rng(seed)

            def fitness_fn(offset_vector: np.ndarray):
                return evaluate_individual(
                    offset_vector, track, mu=mu, path_resolution=path_resolution, speed_params=base_speed_params
                )

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

            if (mu_best_score is None) or (best_score < mu_best_score):
                mu_best_score = float(best_score)
                mu_best = {
                    "lap_time": best_info["lap_time"],
                    "fitness": best_score,
                    "trajectory": best_info["trajectory"],
                    "offsets": best_vector,
                    "speed_profile": best_info["speed_profile"],
                    "curvature": best_info["curvature"],
                }
                # Generate plots immediately on improvement to avoid identical plots when no further gain.
                prefix = f"mu_{mu:.2f}".replace(".", "p")
                ds_segments = np.diff(best_info["trajectory"].s) if len(best_info["trajectory"].s) > 1 else np.array([0.0])
                plot_acceleration_analysis(
                    x=best_info["trajectory"].path[:, 0],
                    y=best_info["trajectory"].path[:, 1],
                    v=best_info["speed_profile"],
                    kappa=best_info["curvature"],
                    ds=ds_segments,
                    mu=mu,
                    g=base_speed_params["g"],
                    out_dir=output_path,
                    prefix=prefix,
                )

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

    if show_progress:
        print()  # finish progress line
        
    if save_results:
        _persist_results(output_path, aggregated_stats, per_run_rows)

    return aggregated_stats, per_run_rows, best_trajectories


def _persist_results(output_dir: Path, aggregated_stats: List[Dict], per_run_rows: List[Dict]) -> None:
    """Write aggregated and per-run results to disk (CSV + JSON)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    agg_csv = output_dir / "aggregated_results.csv"
    agg_json = output_dir / "aggregated_results.json"
    per_run_csv = output_dir / "per_run_results.csv"

    agg_fields = ["mu", "mean", "std", "best", "worst", "runs"]
    with agg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for row in aggregated_stats:
            writer.writerow({k: row.get(k) for k in agg_fields})

    with agg_json.open("w", encoding="utf-8") as f:
        json.dump(aggregated_stats, f, indent=2)

    per_fields = ["mu", "run", "seed", "lap_time", "penalty"]
    with per_run_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_fields)
        writer.writeheader()
        for row in sorted(per_run_rows, key=lambda r: (r["mu"], r["run"])):
            writer.writerow({k: row.get(k) for k in per_fields})
