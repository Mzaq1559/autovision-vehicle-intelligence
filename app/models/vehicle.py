"""Domain models shared across the detection/tracking/speed/analytics layers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

Point = Tuple[float, float]


@dataclass
class Detection:
    """A single detection produced by the detector for one frame."""

    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    frame_index: int
    timestamp: float

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> Point:
        """Bottom-center of the bounding box; a stabler point for ground-plane
        speed/position estimation than the box centroid, since it tracks
        where the vehicle touches the road rather than its visual middle."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


@dataclass
class VehicleTrack:
    """Accumulated state for a single tracked vehicle across frames."""

    track_id: int
    vehicle_type: str
    confidence: float
    first_seen: float
    last_seen: float
    trajectory_length: int = 40

    positions: Deque[Tuple[float, float, float]] = field(default_factory=deque)
    speed_samples: List[float] = field(default_factory=list)

    current_speed_kmh: float = 0.0
    average_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0

    direction: str = "Unknown"
    counted: bool = False
    crossed_zone: bool = False
    violation: bool = False

    def update_detection(self, detection: Detection) -> None:
        if detection.timestamp < self.last_seen:
            # Timestamp moved backwards (e.g. video restart); clear old track history
            self.positions.clear()
            self.speed_samples.clear()
            self.first_seen = detection.timestamp
            self.current_speed_kmh = 0.0
            self.average_speed_kmh = 0.0
            self.max_speed_kmh = 0.0

        self.vehicle_type = detection.class_name
        self.confidence = detection.confidence
        self.last_seen = detection.timestamp
        point = detection.foot_point
        self.positions.append((point[0], point[1], detection.timestamp))
        while len(self.positions) > self.trajectory_length:
            self.positions.popleft()

    def record_speed_sample(self, speed_kmh: float) -> None:
        self.speed_samples.append(speed_kmh)
        self.current_speed_kmh = speed_kmh
        self.max_speed_kmh = max(self.max_speed_kmh, speed_kmh)
        self.average_speed_kmh = sum(self.speed_samples) / len(self.speed_samples)

    @property
    def time_in_scene(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def trajectory(self) -> List[Point]:
        return [(x, y) for (x, y, _t) in self.positions]

    def to_summary(self) -> dict:
        return {
            "track_id": self.track_id,
            "type": self.vehicle_type,
            "confidence": round(self.confidence * 100, 1),
            "current_speed_kmh": round(self.current_speed_kmh, 1),
            "average_speed_kmh": round(self.average_speed_kmh, 1),
            "max_speed_kmh": round(self.max_speed_kmh, 1),
            "direction": self.direction,
            "time_in_scene_s": round(self.time_in_scene, 1),
            "violation": self.violation,
        }
