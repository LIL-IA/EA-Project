import os
from experiments import run_experiments
from plots import plot_lap_time_stats, plot_track_and_lines
from track import Track

#TODO test Defferential evolution algorithm
def main() -> None:
    """Run the full experimental pipeline and produce plots + saved results."""
    track = Track.default()
    aggregated_stats, per_run_rows, best_trajectories = run_experiments(track=track,runs=1)

    lines_dir = os.path.join("results", "lines")
    stats_dir = os.path.join("results", "stats")
    os.makedirs(lines_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    plot_track_and_lines(track, best_trajectories, output_path=os.path.join(lines_dir, "racing_lines.png"))
    for mu, info in best_trajectories.items():
        prefix = f"mu_{mu:.2f}".replace(".", "p")
        plot_track_and_lines(track, {mu: info}, output_path=os.path.join(lines_dir, f"racing_line_{prefix}.png"))
    plot_lap_time_stats(aggregated_stats, output_path=os.path.join(stats_dir, "lap_time_vs_mu.png"))

    print("Experiments finished.")
    for row in aggregated_stats:
        print(f"mu={row['mu']:.2f} mean lap={row['mean']:.2f}s std={row['std']:.2f}s best={row['best']:.2f}s")


if __name__ == "__main__":
    main()
