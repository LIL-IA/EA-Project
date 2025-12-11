# EA-Project

This repository contains a lightweight traffic simulation built around the Intelligent Driver Model (IDM).

## Features
- Directed road network with spawners/destinations, traffic-light crossroads, and roundabouts.
- Vehicles pick random destinations and follow routes found with a simple BFS.
- IDM-based longitudinal control with red-light stopping and roundabout yielding heuristics.
- Live speed multiplier updates while the simulation is running.

## Running the demo
Create and explore the sample city defined in `main.py`:

```bash
python main.py            # interactive mode; type a number to change speed, or 'q' to quit
python main.py --steps 50 --no-interactive  # run a fixed number of steps without live controls
```

Arguments:
- `--dt`: Base simulation timestep in seconds (default: `1.0`).
- `--speed`: Initial speed multiplier (default: `1.0`).
- `--steps`: Number of printed steps before stopping; `0` runs continuously in interactive mode.
- `--no-interactive`: Disable the control thread and run synchronously.

The console output prints the simulated time, number of active vehicles, and per-road vehicle states (position and speed).
