import argparse
from pathlib import Path

from experiments import run_experiments
from plots import plot_lap_time_stats, plot_track_and_lines
from track import Track

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evolutionary racing-line experiments.")
    parser.add_argument("--mu", nargs="+", type=float, default=[0.6, 0.9, 1.2], dest="mu_values",
                        help="Space separated friction coefficients (mu) to evaluate.")
    parser.add_argument("--runs", type=int, default=20, help="Number of independent runs per mu value.")
    parser.add_argument("--base-seed", type=int, default=1234, help="Base seed; per-run seeds derive from it.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for all outputs (plots + CSV/JSON).")
    parser.add_argument("--num-control-points", type=int, default=20, help="Number of lateral offset waypoints.")
    parser.add_argument("--path-resolution", type=float, default=1.0, help="Dense path spacing in meters.")
    parser.add_argument(
        "--evaluation-budget", type=int, default=5000, help="Maximum fitness evaluations per GA run."
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable CLI progress bar.")
    parser.add_argument("--no-save", action="store_true", help="Skip writing CSV/JSON result files.")
    return parser.parse_args()


# TODO: test Differential Evolution algorithm
def main() -> None:
    """Run the full experimental pipeline and produce plots + saved results."""
    args = parse_args()
    if not args.mu_values:
        raise ValueError("Provide at least one mu value.")

    track = Track.default()
    aggregated_stats, per_run_rows, best_trajectories = run_experiments(
        track=track,
        mu_values=tuple(args.mu_values),
        runs=args.runs,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        num_control_points=args.num_control_points,
        path_resolution=args.path_resolution,
        evaluation_budget=args.evaluation_budget,
        save_results=not args.no_save,
    )

    output_root = Path(args.output_dir)
    lines_dir = output_root / "lines"
    stats_dir = output_root / "stats"
    lines_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    if best_trajectories:
        plot_track_and_lines(track, best_trajectories, output_path=str(lines_dir / "racing_lines.png"))
        for mu, info in best_trajectories.items():
            prefix = f"mu_{mu:.2f}".replace(".", "p")
            plot_track_and_lines(track, {mu: info}, output_path=str(lines_dir / f"racing_line_{prefix}.png"))

    if aggregated_stats:
        plot_lap_time_stats(aggregated_stats, output_path=str(stats_dir / "lap_time_vs_mu.png"))

    print("Experiments finished.")
    for row in aggregated_stats:
        print(f"mu={row['mu']:.2f} mean lap={row['mean']:.2f}s std={row['std']:.2f}s best={row['best']:.2f}s")


if __name__ == "__main__":
    main()
