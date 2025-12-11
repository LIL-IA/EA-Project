from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Vehicle:
    vehicle_id: int
    route: List[str]
    position: float = 0.0
    speed: float = 0.0
    current_road_index: int = 0
    finished: bool = False

    max_accel: float = 1.5
    comfortable_decel: float = 2.0
    desired_headway: float = 1.5
    min_spacing: float = 2.0
    desired_speed: float = 15.0

    def current_road(self) -> str:
        return self.route[self.current_road_index]

    def advance_road(self) -> None:
        if self.current_road_index + 1 < len(self.route):
            self.current_road_index += 1
            self.position = 0.0
        else:
            self.finished = True

    def project_state(self, dt: float, acceleration: float) -> None:
        self.speed = max(0.0, self.speed + acceleration * dt)
        self.position += self.speed * dt
