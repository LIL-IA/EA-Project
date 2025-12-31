import os
from typing import Dict, Iterable
import numpy as np
import matplotlib.pyplot as plt

from track import Track


def plot_track_and_lines(track: Track, best_trajectories: Dict[float, Dict], output_path: str) -> None:
    """Plot track boundaries, centerline, best lines, and their waypoints."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    left, right, center = track.boundaries(num_samples=600)

    plt.figure(figsize=(8, 8))
    plt.plot(left[:, 0], left[:, 1], "k--", linewidth=1.0, label="Left boundary")
    plt.plot(right[:, 0], right[:, 1], "k--", linewidth=1.0, label="Right boundary")
    plt.plot(center[:, 0], center[:, 1], color="gray", linewidth=1.0, label="Centerline")

    for mu, info in sorted(best_trajectories.items()):
        path = info["trajectory"].path
        line_handle, = plt.plot(path[:, 0], path[:, 1], linewidth=2.0, label=f"Best line mu={mu}")

        # Waypoints (control offsets projected on track normals).
        offsets_base = info["trajectory"].offsets_base
        base_s = np.linspace(0.0, track.length, len(offsets_base), endpoint=False)
        ctrl_positions = track.position(base_s) + track.normal(base_s) * offsets_base[:, None]
        plt.scatter(
            ctrl_positions[:, 0],
            ctrl_positions[:, 1],
            s=14,
            marker="o",
            color=line_handle.get_color(),
            edgecolors="k",
            linewidths=0.4,
            alpha=0.9,
            label=f"Waypoints mu={mu}",
        )

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


def plot_acceleration_analysis(
    x: Iterable[float],
    y: Iterable[float],
    v: Iterable[float],
    kappa: Iterable[float],
    ds: Iterable[float],
    mu: float,
    g: float,
    out_dir: str,
    prefix: str,
) -> None:
    """
    Generate acceleration analysis plots for a trajectory and speed profile.

    Args:
        x, y: coordinates of the dense path samples.
        v: speed profile (m/s) along the path.
        kappa: curvature at each sample (1/m).
        ds: segment lengths between samples (m), length len(v)-1 (or scalar); last a_long is reused.
        mu: friction coefficient.
        g: gravity constant.
        out_dir: base directory for output images; subfolders are created per plot type.
        prefix: filename prefix (e.g., 'mu_0p90').
    """
    accel_dir = os.path.join(out_dir, "acceleration_profiles")
    friction_dir = os.path.join(out_dir, "friction_circles")
    colored_dir = os.path.join(out_dir, "colored_paths_lateral")
    colored_long_dir = os.path.join(out_dir, "colored_paths_longitudinal")
    for d in (accel_dir, friction_dir, colored_dir, colored_long_dir):
        os.makedirs(d, exist_ok=True)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    ds = np.asarray(ds, dtype=float)

    if ds.ndim == 0:
        ds = np.full(max(len(v) - 1, 1), float(ds))
    if ds.size < len(v) - 1:
        ds = np.pad(ds, (0, (len(v) - 1) - ds.size), constant_values=ds[-1] if ds.size > 0 else 0.0)

    # Build arc-length axis.
    s_axis = np.concatenate([[0.0], np.cumsum(ds[: len(v) - 1])])

    # Accelerations.
    a_lat = v ** 2 * np.abs(kappa)
    a_long = np.zeros_like(v)
    for i in range(len(v) - 1):
        if ds[i] > 0:
            a_long[i] = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds[i])
    if len(v) > 1:
        a_long[-1] = a_long[-2]

    # Plot A: acceleration profiles vs s (lat + long).
    plt.figure(figsize=(8, 4))
    plt.plot(s_axis, a_lat, label="lateral a_lat", linewidth=1.5)
    plt.plot(s_axis, a_long, label="longitudinal a_long", linewidth=1.5)
    plt.xlabel("Arc length s [m]")
    plt.ylabel("Acceleration [m/s^2]")
    plt.title(f"Acceleration profile (mu={mu:.2f})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(accel_dir, f"accel_profiles_{prefix}.png"), dpi=200)
    plt.close()

    # Plot B: friction circle scatter.
    radius = mu * g
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)

    plt.figure(figsize=(5, 5))
    plt.scatter(a_long, a_lat, s=8, alpha=0.6, label="samples")
    plt.plot(circle_x, circle_y, "r--", label=f"mu*g={radius:.2f} m/s^2")
    plt.xlabel("Longitudinal acceleration [m/s^2]")
    plt.ylabel("Lateral acceleration [m/s^2]")
    plt.title(f"Friction circle (mu={mu:.2f})")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(friction_dir, f"friction_circle_{prefix}.png"), dpi=200)
    plt.close()

    # Optional Plot C: trajectory colored by lateral acceleration.
    plt.figure(figsize=(7, 7))
    sc = plt.scatter(x, y, c=a_lat, cmap="plasma", s=6)
    plt.colorbar(sc, label="Lateral acceleration [m/s^2]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Trajectory colored by a_lat (mu={mu:.2f})")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(colored_dir, f"track_colored_alat_{prefix}.png"), dpi=200)
    plt.close()

    # Plot D: trajectory colored by longitudinal acceleration.
    plt.figure(figsize=(7, 7))
    sc_long = plt.scatter(x, y, c=a_long, cmap="plasma", s=6)
    plt.colorbar(sc_long, label="Longitudinal acceleration [m/s^2]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Trajectory colored by a_long (mu={mu:.2f})")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(colored_long_dir, f"track_colored_along_{prefix}.png"), dpi=200)
    plt.close()
