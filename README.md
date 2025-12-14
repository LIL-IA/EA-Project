# Evolutionary Racing Line Optimization

This project studies how the tire-road friction coefficient `mu` affects an evolved racing line and lap time for a simplified Formula 1-style vehicle on a 2D closed circuit. A Genetic Algorithm (GA) evolves lateral offsets from a procedural track centerline; a curvature + adhesion limited speed model computes lap times.

## Project layout
```
track.py         # Track generation and geometry utilities
trajectory.py    # Offset-to-path conversion via splines
speed_model.py   # Curvature estimation and two-pass speed profile
fitness.py       # Trajectory evaluation (lap time + penalties)
ga.py            # Real-valued GA with tournament selection, crossover, mutation, elitism
experiments.py   # Batch experiments across mu values and seeds, saves CSV/JSON
plots.py         # Plotting utilities (track + lap time vs mu)
main.py          # End-to-end entry point
requirements.txt # Minimal dependencies: numpy, scipy, matplotlib
```

## Installation
1. Create/activate a Python 3 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running experiments
Execute the full pipeline (GA runs for each `mu`, save results, generate plots):
```bash
python main.py
```
CLI progress is shown while GA runs; set `show_progress=False` in `run_experiments` if you need silent execution.
Outputs are written to `results/`:
- `per_run_results.csv`: per-seed lap times and penalties
- `aggregated_results.csv` and `aggregated_results.json`: mean/std/best/worst per `mu`
- Plots grouped in subfolders:
  - `lines/racing_lines.png`: best evolved racing lines + waypoints (all mu)
  - `lines/racing_line_mu_<mu>.png`: best evolved line for a given mu
  - `stats/lap_time_vs_mu.png`: lap time vs friction with error bars
  - `acceleration_profiles/accel_profiles_<mu>.png`: lateral + longitudinal acceleration vs s
  - `friction_circles/friction_circle_<mu>.png`: friction-circle scatter
  - `colored_paths/track_colored_alat_<mu>.png`: trajectory colored by lateral acceleration
  - `colored_paths_longitudinal/track_colored_along_<mu>.png`: trajectory colored by longitudinal acceleration

## Model overview
- **Track**: Smooth closed loop generated procedurally; constant width. Centerline is interpolated with periodic cubic splines and exposes tangent/normal vectors and left/right boundaries.
- **Trajectory representation**: An individual is a vector of lateral offsets along arc length. Offsets are interpolated with a periodic cubic spline, clipped to the track width, and applied along the centerline normals to form a dense path.
- **Speed model**: Curvature is estimated from the dense path. Speeds follow the specified friction-circle constrained, two-pass profile (forward acceleration, backward braking) using configurable `a_engine`, `a_brake`, and `v_max`. Lap time sums `ds / max(v, v_min)`.
- **Constraints/penalties**: Offsets and dense path points are clipped to stay within boundaries; infeasible geometry is penalized heavily. A small smoothness penalty discourages curvature spikes.

## Genetic Algorithm
- Real-valued genome (offset vector).
- Tournament selection (`k=3`), arithmetic crossover, Gaussian mutation (sigma configurable), and elitism.
- Stop condition: fixed evaluation budget for fair comparison across runs.

Default experiment settings (see `experiments.py` for tweaks):
- `mu_values = [0.6, 0.9, 1.2]`
- `runs = 20` independent seeds per `mu`
- GA: `pop_size=40`, `evaluation_budget=2000`, `crossover_rate=0.9`, `mutation_sigma=0.5`, `elite_size=2`
- Trajectory: `num_control_points=60`, `path_resolution=1.0 m`
- Speed model: `a_engine=6 m/s^2`, `a_brake=8 m/s^2`, `v_max=80 m/s`, `v_min=1 m/s`
- Smoothness: `lambda_smooth=0.01` (set to `0` or disable in `experiments.py` for ablation; higher values trade lap-time optimality for gentler offset changes)

## Reproducibility
Provide a base seed (`base_seed` in `experiments.py`); per-run seeds derive deterministically from it and the `mu` index. Adjust parameters directly in the module or wrap `run_experiments` from your own scripts.

## Notes
- Only `numpy`, `scipy`, and `matplotlib` are required.
- The procedural track can be swapped: supply your own `Track` instance to `run_experiments`.
