from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .city import CityMap, IntersectionType
from .vehicle import Vehicle


@dataclass
class Spawner:
    node_id: str
    spawn_rate_per_minute: float

    @property
    def spawn_probability(self) -> float:
        # convert to probability per second assuming dt ~1s
        return self.spawn_rate_per_minute / 60.0


class Simulation:
    def __init__(self, city: CityMap, spawners: List[Spawner], *, dt: float = 1.0):
        self.city = city
        self.spawners = spawners
        self.dt = dt
        self.time = 0.0
        self.speed_multiplier = 1.0

        self.vehicles: Dict[int, Vehicle] = {}
        self.road_vehicles: Dict[str, List[int]] = {}
        self.intersection_last_entry: Dict[str, float] = {}
        self.next_vehicle_id = 1

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Traffic light phases per intersection
        self.light_cycle_duration = 20.0
        self.light_green_duration = 10.0

    # --- Public controls -------------------------------------------------
    def set_speed_multiplier(self, multiplier: float) -> None:
        self.speed_multiplier = max(0.1, multiplier)

    def start_async(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    # --- Simulation loop -------------------------------------------------
    def run(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()
            self.step()
            elapsed = time.time() - start
            sleep_time = max(0.0, (self.dt / self.speed_multiplier) - elapsed)
            time.sleep(sleep_time)

    def step(self) -> None:
        self._spawn_new_vehicles()
        self._update_vehicle_positions()
        self._cleanup_finished()
        self.time += self.dt

    # --- Helpers ---------------------------------------------------------
    def _spawn_new_vehicles(self) -> None:
        destinations = [node.node_id for node in self.city.destinations()]
        for spawner in self.spawners:
            if random.random() < spawner.spawn_probability * self.dt:
                if not destinations:
                    continue
                destination = random.choice(destinations)
                if destination == spawner.node_id:
                    continue
                path = self.city.find_path(spawner.node_id, destination)
                if not path:
                    continue
                vehicle = Vehicle(vehicle_id=self.next_vehicle_id, route=path)
                self.next_vehicle_id += 1
                self.vehicles[vehicle.vehicle_id] = vehicle
                self.road_vehicles.setdefault(vehicle.current_road(), []).append(vehicle.vehicle_id)

    def _update_vehicle_positions(self) -> None:
        # ensure vehicles sorted on each road
        for road_id, ids in list(self.road_vehicles.items()):
            ids = [vid for vid in ids if not self.vehicles[vid].finished]
            ids.sort(key=lambda vid: self.vehicles[vid].position)
            self.road_vehicles[road_id] = ids

        for road_id, vehicle_ids in list(self.road_vehicles.items()):
            road = self.city.roads[road_id]
            for idx, vid in enumerate(vehicle_ids):
                vehicle = self.vehicles[vid]
                leader_gap, relative_speed = self._leader_info(vehicle, road_id, vehicle_ids, idx)
                gap_to_stop, stop_required = self._intersection_gap(vehicle, road)
                effective_gap = leader_gap
                effective_relative_speed = relative_speed

                if stop_required and (gap_to_stop < effective_gap):
                    effective_gap = gap_to_stop
                    effective_relative_speed = vehicle.speed

                acceleration = self._idm_acceleration(
                    vehicle,
                    road,
                    effective_gap,
                    effective_relative_speed,
                )
                vehicle.project_state(self.dt, acceleration)

                # Move across intersection if passed the end of road
                while not vehicle.finished and vehicle.position >= road.length:
                    excess = vehicle.position - road.length
                    vehicle.advance_road()
                    if vehicle.finished:
                        break
                    self._enter_intersection(road.end)
                    next_road = self.city.roads[vehicle.current_road()]
                    vehicle.position = excess
                    self._move_vehicle_between_roads(vid, road_id, next_road.road_id)
                    road_id = next_road.road_id
                    road = next_road

    def _leader_info(self, vehicle: Vehicle, road_id: str, vehicle_ids: List[int], index: int) -> Tuple[float, float]:
        if index + 1 >= len(vehicle_ids):
            return 1e6, 0.0
        leader = self.vehicles[vehicle_ids[index + 1]]
        gap = leader.position - vehicle.position
        relative_speed = vehicle.speed - leader.speed
        return max(gap, 0.1), relative_speed

    def _intersection_gap(self, vehicle: Vehicle, road) -> Tuple[float, bool]:
        node = self.city.nodes[road.end]
        remaining = road.length - vehicle.position
        if node.intersection_type is None:
            return 1e6, False

        if node.intersection_type == IntersectionType.TRAFFIC_LIGHT:
            if not self._is_green(road_id=road.road_id, node_id=node.node_id):
                return max(remaining, 0.1), True
            return 1e6, False

        if node.intersection_type == IntersectionType.ROUNDABOUT:
            last_entry = self.intersection_last_entry.get(node.node_id, -1e9)
            if (self.time - last_entry) < 2.0 and remaining < 15.0:
                return max(remaining, 0.1), True
            return 1e6, False

        return 1e6, False

    def _enter_intersection(self, node_id: str) -> None:
        node = self.city.nodes[node_id]
        if node.intersection_type == IntersectionType.ROUNDABOUT:
            self.intersection_last_entry[node_id] = self.time

    def _move_vehicle_between_roads(self, vehicle_id: int, from_road: str, to_road: str) -> None:
        if from_road in self.road_vehicles:
            self.road_vehicles[from_road] = [vid for vid in self.road_vehicles[from_road] if vid != vehicle_id]
        self.road_vehicles.setdefault(to_road, []).append(vehicle_id)

    def _cleanup_finished(self) -> None:
        finished_ids = [vid for vid, v in self.vehicles.items() if v.finished]
        for vid in finished_ids:
            road_id = self.vehicles[vid].current_road()
            if road_id in self.road_vehicles:
                self.road_vehicles[road_id] = [i for i in self.road_vehicles[road_id] if i != vid]
            del self.vehicles[vid]

    def _idm_acceleration(self, vehicle: Vehicle, road, gap: float, relative_speed: float) -> float:
        desired_speed = min(vehicle.desired_speed, road.speed_limit)
        s_star = vehicle.min_spacing + vehicle.speed * vehicle.desired_headway + (
            vehicle.speed * relative_speed
        ) / (2 * (vehicle.max_accel * vehicle.comfortable_decel) ** 0.5)
        acceleration = vehicle.max_accel * (1 - (vehicle.speed / desired_speed) ** 4 - (s_star / gap) ** 2)
        return max(-vehicle.comfortable_decel, min(acceleration, vehicle.max_accel))

    def _is_green(self, road_id: str, node_id: str) -> bool:
        incoming_roads = [rid for rid, r in self.city.roads.items() if r.end == node_id]
        if not incoming_roads:
            return True
        incoming_roads.sort()
        cycle_position = self.time % self.light_cycle_duration
        active_index = int(cycle_position // self.light_green_duration) % len(incoming_roads)
        return incoming_roads[active_index] == road_id

    def snapshot(self) -> Dict[str, List[Tuple[int, float, float]]]:
        """Return simple snapshot for printing or visualization."""
        snapshot: Dict[str, List[Tuple[int, float, float]]] = {}
        for road_id, vids in self.road_vehicles.items():
            road_snapshot: List[Tuple[int, float, float]] = []
            for vid in vids:
                vehicle = self.vehicles[vid]
                road_snapshot.append((vehicle.vehicle_id, vehicle.position, vehicle.speed))
            snapshot[road_id] = road_snapshot
        return snapshot


__all__ = ["Simulation", "Spawner"]
