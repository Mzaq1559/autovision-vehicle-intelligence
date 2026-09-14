"""Traffic analytics: vehicle counting, speed violations, and history.

Keeps the pure counting/violation rules as standalone functions (easy to
unit test without any video or model) and a :class:`TrafficAnalytics`
orchestrator that wires them together with the live set of
:class:`~app.models.vehicle.VehicleTrack` objects for the Streamlit app.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.models.vehicle import VehicleTrack
from app.speed.estimator import is_speed_violation


def has_crossed_line(prev_y: float, curr_y: float, line_y: float) -> bool:
    """True if a point moving from prev_y to curr_y crossed the horizontal
    line at line_y, in either direction. Used to trigger a one-time count
    when a vehicle crosses the configured measurement/counting line."""

    if prev_y == curr_y:
        return False
    lo, hi = sorted((prev_y, curr_y))
    return lo <= line_y <= hi


def should_count(track: VehicleTrack, crossed: bool) -> bool:
    """A vehicle is counted at most once, only on the frame it crosses the
    line and only if it has not already been counted."""

    return crossed and not track.counted


@dataclass
class TrafficSnapshot:
    timestamp: float
    active_vehicles: int
    total_counted: int
    average_speed_kmh: float
    max_speed_kmh: float
    violations: int
    counts_by_type: Dict[str, int]


@dataclass
class TrafficAnalytics:
    """Owns the live set of tracked vehicles and derived statistics."""

    speed_limit_kmh: float = 60.0
    line_y_fraction: float = 0.6

    tracks: Dict[int, VehicleTrack] = field(default_factory=dict)
    counts_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_counted: int = 0
    violation_log: List[dict] = field(default_factory=list)
    history: List[TrafficSnapshot] = field(default_factory=list)

    def get_or_create_track(
        self,
        track_id: int,
        vehicle_type: str,
        confidence: float,
        timestamp: float,
        trajectory_length: int,
    ) -> VehicleTrack:
        if track_id not in self.tracks:
            self.tracks[track_id] = VehicleTrack(
                track_id=track_id,
                vehicle_type=vehicle_type,
                confidence=confidence,
                first_seen=timestamp,
                last_seen=timestamp,
                trajectory_length=trajectory_length,
            )
        return self.tracks[track_id]

    def maybe_count(self, track: VehicleTrack, frame_height: float) -> bool:
        """Check the track's two most recent positions against the counting
        line and register a count exactly once per vehicle. Returns True if
        a new count was registered this call."""

        if len(track.positions) < 2:
            return False
        (_, prev_y, _), (_, curr_y, _) = track.positions[-2], track.positions[-1]
        line_y = frame_height * self.line_y_fraction
        crossed = has_crossed_line(prev_y, curr_y, line_y)

        if not should_count(track, crossed):
            return False

        track.counted = True
        track.crossed_zone = True
        self.total_counted += 1
        self.counts_by_type[track.vehicle_type] += 1
        return True

    def evaluate_violation(self, track: VehicleTrack) -> bool:
        violation = is_speed_violation(track.current_speed_kmh, self.speed_limit_kmh)
        was_new = violation and not track.violation
        track.violation = track.violation or violation
        if was_new:
            self.violation_log.append(
                {
                    "track_id": track.track_id,
                    "type": track.vehicle_type,
                    "speed_kmh": round(track.current_speed_kmh, 1),
                    "limit_kmh": self.speed_limit_kmh,
                    "timestamp": track.last_seen,
                }
            )
        return violation

    def active_track_ids(self, stale_after_s: float, now: Optional[float] = None) -> List[int]:
        now = now if now is not None else time.time()
        return [
            tid for tid, t in self.tracks.items() if (now - t.last_seen) <= stale_after_s
        ]

    def snapshot(self, active_ids: List[int], record_history: bool = True) -> TrafficSnapshot:
        active = [self.tracks[tid] for tid in active_ids if tid in self.tracks]
        speeds = [t.current_speed_kmh for t in active if t.current_speed_kmh > 0]
        snap = TrafficSnapshot(
            timestamp=time.time(),
            active_vehicles=len(active),
            total_counted=self.total_counted,
            average_speed_kmh=round(sum(speeds) / len(speeds), 1) if speeds else 0.0,
            max_speed_kmh=round(max((t.max_speed_kmh for t in active), default=0.0), 1),
            violations=len(self.violation_log),
            counts_by_type=dict(self.counts_by_type),
        )
        # Only append to history on UI-update frames so the chart data stays
        # meaningful and the list does not grow to thousands of entries per run.
        # The snapshot itself is always fully computed so the frame overlay text
        # (active vehicles, avg speed) is always current.
        if record_history:
            self.history.append(snap)
        return snap

    def vehicle_table_rows(self, active_ids: List[int]) -> List[dict]:
        return [self.tracks[tid].to_summary() for tid in active_ids if tid in self.tracks]

    def type_distribution(self) -> Counter:
        return Counter(self.counts_by_type)
