from __future__ import annotations

import argparse
import threading
import time
from typing import List

from traffic_sim import CityMap, IntersectionType, Simulation, Spawner


def build_demo_city() -> tuple[CityMap, List[Spawner]]:
    city = CityMap()
    # Nodes: coordinates not used for physics but helpful for future visualization
    city.add_node("A", 0, 0, is_spawner=True, intersection_type=IntersectionType.TRAFFIC_LIGHT)
    city.add_node("B", 100, 0, is_destination=True, intersection_type=IntersectionType.TRAFFIC_LIGHT)
    city.add_node("C", 200, 0, is_spawner=True)
    city.add_node("D", 100, 80, is_destination=True, intersection_type=IntersectionType.ROUNDABOUT)
    city.add_node("E", 0, 80, is_destination=True)

    # Roads (directed)
    city.add_road("A_B", "A", "B", length=120, speed_limit=13.9)
    city.add_road("B_C", "B", "C", length=120, speed_limit=13.9)
    city.add_road("C_B", "C", "B", length=120, speed_limit=13.9)
    city.add_road("B_D", "B", "D", length=60, speed_limit=10.0)
    city.add_road("D_B", "D", "B", length=60, speed_limit=10.0)
    city.add_road("A_E", "A", "E", length=80, speed_limit=13.9)
    city.add_road("E_A", "E", "A", length=80, speed_limit=13.9)
    city.add_road("E_D", "E", "D", length=110, speed_limit=13.9)
    city.add_road("D_E", "D", "E", length=110, speed_limit=13.9)

    spawners = [
        Spawner("A", spawn_rate_per_minute=20),
        Spawner("C", spawn_rate_per_minute=15),
    ]
    return city, spawners


def snapshot_to_text(sim: Simulation) -> str:
    lines = [f"Time: {sim.time:.1f}s  Vehicles: {len(sim.vehicles)}  Speed x{sim.speed_multiplier:.1f}"]
    for road_id, entries in sim.snapshot().items():
        if not entries:
            continue
        items = ", ".join(
            f"#{vid}: pos={pos:.1f}m v={spd:.1f}m/s" for vid, pos, spd in entries
        )
        lines.append(f"  {road_id}: {items}")
    return "\n".join(lines)


def interactive_control(sim: Simulation) -> None:
    print("Controls: enter a new speed multiplier (e.g. 2.0) or 'q' to quit.")
    while True:
        try:
            user_input = input()
        except EOFError:
            break
        if user_input.strip().lower() == "q":
            sim.stop()
            break
        try:
            new_speed = float(user_input.strip())
            sim.set_speed_multiplier(new_speed)
            print(f"Updated speed multiplier to x{sim.speed_multiplier:.1f}")
        except ValueError:
            print("Please enter a number or 'q' to quit")


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic simulation demo using IDM")
    parser.add_argument("--steps", type=int, default=0, help="Number of steps to run before stopping (0 = infinite)")
    parser.add_argument("--dt", type=float, default=1.0, help="Base simulation time step in seconds")
    parser.add_argument("--speed", type=float, default=1.0, help="Initial speed multiplier")
    parser.add_argument("--no-interactive", action="store_true", help="Disable live speed control")
    args = parser.parse_args()

    city, spawners = build_demo_city()
    sim = Simulation(city, spawners, dt=args.dt)
    sim.set_speed_multiplier(args.speed)

    if args.no_interactive:
        for _ in range(args.steps or 60):
            sim.step()
            print(snapshot_to_text(sim))
        return

    sim.start_async()
    control_thread = threading.Thread(target=interactive_control, args=(sim,), daemon=True)
    control_thread.start()

    max_steps = args.steps or 0
    steps = 0
    try:
        while not sim._stop_event.is_set():  # type: ignore[attr-defined]
            print(snapshot_to_text(sim))
            steps += 1
            if max_steps and steps >= max_steps:
                sim.stop()
                break
            time.sleep(1.0)
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
