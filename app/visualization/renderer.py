"""Drawing utilities that overlay detection/tracking/speed info on frames.

All functions take and return a numpy BGR frame (OpenCV convention) and are
side-effect-free with respect to their inputs (they draw on a copy).
Isolating this in one module keeps OpenCV drawing calls out of the
Streamlit app and the analytics logic, and makes it easy to swap the
rendering style later without touching pipeline code.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from app.models.vehicle import Detection, VehicleTrack

Color = Tuple[int, int, int]

NORMAL_COLOR: Color = (46, 204, 113)  # green (BGR)
VIOLATION_COLOR: Color = (39, 39, 231)  # red (BGR)
LINE_COLOR: Color = (255, 191, 0)  # blue-ish (BGR)
TRAJECTORY_COLOR: Color = (255, 200, 0)


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    tracks: Dict[int, VehicleTrack],
) -> np.ndarray:
    """Draw bounding boxes, IDs, class, confidence, and speed for each
    detection, colored red when the associated track is a speed violation."""

    out = frame.copy()
    for det in detections:
        track = tracks.get(det.track_id)
        violation = bool(track and track.violation)
        color = VIOLATION_COLOR if violation else NORMAL_COLOR

        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        speed_txt = ""
        if track and track.current_speed_kmh > 0:
            speed_txt = f" {track.current_speed_kmh:.0f}km/h"
        label = f"#{det.track_id} {det.class_name} {det.confidence * 100:.0f}%{speed_txt}"
        if violation:
            label += " VIOLATION"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_trajectories(frame: np.ndarray, tracks: Iterable[VehicleTrack]) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        points = [(int(x), int(y)) for (x, y) in track.trajectory]
        for i in range(1, len(points)):
            cv2.line(out, points[i - 1], points[i], TRAJECTORY_COLOR, 2)
    return out


def draw_measurement_line(frame: np.ndarray, line_y_fraction: float) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    y = int(height * line_y_fraction)
    cv2.line(out, (0, y), (width, y), LINE_COLOR, 2)
    cv2.putText(
        out,
        "measurement / counting line",
        (8, max(16, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        LINE_COLOR,
        1,
        cv2.LINE_AA,
    )
    return out


def draw_summary_overlay(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    out = frame.copy()
    x, y = 10, 24
    for line in lines:
        cv2.putText(
            out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA
        )
        y += 22
    return out


def render_frame(
    frame: np.ndarray,
    detections: Iterable[Detection],
    tracks: Dict[int, VehicleTrack],
    line_y_fraction: float,
    show_trajectories: bool = True,
    summary_lines: List[str] | None = None,
) -> np.ndarray:
    """Compose all overlays in a sensible draw order."""

    out = draw_measurement_line(frame, line_y_fraction)
    if show_trajectories:
        out = draw_trajectories(out, tracks.values())
    out = draw_detections(out, detections, tracks)
    if summary_lines:
        out = draw_summary_overlay(out, summary_lines)
    return out
