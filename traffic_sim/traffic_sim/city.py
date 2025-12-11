from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class IntersectionType(str, Enum):
    TRAFFIC_LIGHT = "traffic_light"
    ROUNDABOUT = "roundabout"


@dataclass
class Node:
    node_id: str
    x: float
    y: float
    is_spawner: bool = False
    is_destination: bool = False
    intersection_type: Optional[IntersectionType] = None


@dataclass
class Road:
    road_id: str
    start: str
    end: str
    length: float
    speed_limit: float = 13.9  # ~50 km/h


@dataclass
class CityMap:
    nodes: Dict[str, Node] = field(default_factory=dict)
    roads: Dict[str, Road] = field(default_factory=dict)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)

    def add_node(
        self,
        node_id: str,
        x: float,
        y: float,
        *,
        is_spawner: bool = False,
        is_destination: bool = False,
        intersection_type: Optional[IntersectionType] = None,
    ) -> None:
        self.nodes[node_id] = Node(
            node_id=node_id,
            x=x,
            y=y,
            is_spawner=is_spawner,
            is_destination=is_destination,
            intersection_type=intersection_type,
        )
        self.adjacency.setdefault(node_id, [])

    def add_road(self, road_id: str, start: str, end: str, length: float, speed_limit: float = 13.9) -> None:
        if start not in self.nodes or end not in self.nodes:
            raise ValueError("Road endpoints must be valid nodes")
        self.roads[road_id] = Road(road_id, start, end, length, speed_limit)
        self.adjacency.setdefault(start, []).append(road_id)

    def outgoing_roads(self, node_id: str) -> List[Road]:
        return [self.roads[rid] for rid in self.adjacency.get(node_id, [])]

    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """Simple BFS for a path returning list of road ids."""
        from collections import deque

        queue: deque[Tuple[str, List[str]]] = deque([(start, [])])
        visited = {start}
        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            for road_id in self.adjacency.get(current, []):
                road = self.roads[road_id]
                if road.end in visited:
                    continue
                visited.add(road.end)
                queue.append((road.end, path + [road_id]))
        return None

    def spawners(self) -> List[Node]:
        return [node for node in self.nodes.values() if node.is_spawner]

    def destinations(self) -> List[Node]:
        return [node for node in self.nodes.values() if node.is_destination]


__all__ = ["CityMap", "Road", "Node", "IntersectionType"]
