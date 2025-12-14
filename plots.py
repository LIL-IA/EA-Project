import os
from typing import Dict
import matplotlib.pyplot as plt

from track import Track


def plot_track_and_lines(track: Track, best_trajectories: Dict[float, Dict], output_path: str) -> None:
    """Plot track boundaries and best racing lines per friction coefficient."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    left, right, center = track.boundaries(num_samples=600)

    plt.figure(figsize=(8, 8))
    plt.plot(left[:, 0], left[:, 1], "k--", linewidth=1.0, label="Left boundary")
    plt.plot(right[:, 0], right[:, 1], "k--", linewidth=1.0, label="Right boundary")
    plt.plot(center[:, 0], center[:, 1], color="gray", linewidth=1.0, label="Centerline")

    for mu, info in sorted(best_trajectories.items()):
        path = info["trajectory"].path
        plt.plot(path[:, 0], path[:, 1], linewidth=2.0, label=f"Best line mu={mu}")

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Evolved racing lines per friction coefficient")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_lap_time_stats(stats: list[dict], output_path: str) -> None:
    """Plot lap time means with error bars versus mu."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mu_vals = [row["mu"] for row in stats]
    means = [row["mean"] for row in stats]
    stds = [row["std"] for row in stats]

    plt.figure(figsize=(6, 4))
    plt.errorbar(mu_vals, means, yerr=stds, fmt="o-", capsize=5, linewidth=2.0)
    plt.xlabel("Friction coefficient mu")
    plt.ylabel("Lap time [s]")
    plt.title("Lap time vs friction coefficient")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
