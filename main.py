from experiments import run_experiments
from plots import plot_lap_time_stats, plot_track_and_lines
from track import Track


def main() -> None:
    """Run the full experimental pipeline and produce plots + saved results."""
    track = Track.default()
    aggregated_stats, per_run_rows, best_trajectories = run_experiments(track=track)

    plot_track_and_lines(track, best_trajectories, output_path="results/racing_lines.png")
    plot_lap_time_stats(aggregated_stats, output_path="results/lap_time_vs_mu.png")

    print("Experiments finished.")
    for row in aggregated_stats:
        print(f"mu={row['mu']:.2f} mean lap={row['mean']:.2f}s std={row['std']:.2f}s best={row['best']:.2f}s")


if __name__ == "__main__":
    main()
