"""Multi-object tracking on top of Ultralytics' built-in tracker support.

Ultralytics YOLO ships with ByteTrack and BoT-SORT trackers accessible via
``model.track(...)``. Rather than reimplementing a tracker, AutoVision wraps
that interface so the rest of the codebase only depends on our own
:class:`~app.models.vehicle.Detection` type, not Ultralytics internals. This
keeps the detector/tracker swappable and keeps the pure-logic pieces (like
direction estimation below) independently testable without a model.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, List, Optional, Sequence, Tuple

from app.config.settings import AppConfig
from app.models.vehicle import Detection

logger = logging.getLogger(__name__)

Point = Tuple[float, float]

_TRACKER_YAML = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


class VehicleTracker:
    """Runs detection + tracking on a video source using Ultralytics YOLO.

    This class intentionally does the least amount of custom work possible:
    Ultralytics already provides a well-tested detector and two modern
    trackers (ByteTrack, BoT-SORT). AutoVision's job is to configure them
    sensibly, convert results into our own dataclasses, and hand off to the
    speed/analytics/visualization layers.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._model = None
        self._tracker_yaml = _TRACKER_YAML.get(
            config.tracker.type.lower(), "bytetrack.yaml"
        )

    def _ensure_model_loaded(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised only w/o deps
            raise RuntimeError(
                "The 'ultralytics' package is required for live detection but "
                "is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        device = self._resolve_device()
        logger.info("Loading YOLO weights '%s' on device '%s'", self.config.model.weights, device)
        self._model = YOLO(self.config.model.weights)
        self._device = device
        return self._model

    def _resolve_device(self) -> str:
        configured = self.config.model.device.lower()
        if configured != "auto":
            return configured
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"

    def track_frame(
        self,
        frame,
        frame_index: int,
        video_timestamp: Optional[float] = None,
    ) -> List[Detection]:
        """Run detection+tracking on a single frame, returning Detections.

        Args:
            frame: BGR numpy array.
            frame_index: 0-based index of this frame in the source video.
                Always use the *original* source index even when frame-skipping
                so that video-time calculations remain correct.
            video_timestamp: Seconds elapsed in the video at this frame
                (``frame_index / source.fps``).  Pass this for file sources so
                that speed estimation uses video time rather than wall-clock
                time.  When ``None`` (e.g. for a live webcam stream where
                wall-clock *is* the correct reference), ``time.time()`` is used
                as the fallback.
        """

        model = self._ensure_model_loaded()
        allowed_classes = set(c.lower() for c in self.config.model.classes)

        results = model.track(
            frame,
            persist=True,
            conf=self.config.model.confidence,
            iou=self.config.model.iou_threshold,
            tracker=self._tracker_yaml,
            device=getattr(self, "_device", None),
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            return detections

        names = result.names
        # Use caller-supplied video timestamp when available so that speed
        # estimation reflects video time, not CPU processing time.
        timestamp = video_timestamp if video_timestamp is not None else time.time()

        for box, track_id, cls_idx, conf in zip(
            boxes.xyxy.tolist(),
            boxes.id.tolist(),
            boxes.cls.tolist(),
            boxes.conf.tolist(),
        ):
            class_name = names.get(int(cls_idx), str(int(cls_idx))) if isinstance(
                names, dict
            ) else str(names[int(cls_idx)])
            class_name = class_name.lower()
            if allowed_classes and class_name not in allowed_classes:
                continue

            detections.append(
                Detection(
                    track_id=int(track_id),
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=tuple(box),  # type: ignore[arg-type]
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )

        return detections


def estimate_direction(
    trajectory: Sequence[Point],
    mode: str = "screen_relative",
    min_displacement_px: float = 6.0,
    geographic_labels: Optional[dict] = None,
) -> str:
    """Estimate a vehicle's travel direction from its recent trajectory.

    Uses the net displacement between the oldest and newest points in the
    trajectory so brief jitter does not flip the label frame-to-frame. When
    displacement is too small to be meaningful, returns "Stationary".
    """

    if len(trajectory) < 2:
        return "Unknown"

    start = trajectory[0]
    end = trajectory[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    if abs(dx) < min_displacement_px and abs(dy) < min_displacement_px:
        return "Stationary"

    if abs(dx) >= abs(dy):
        screen_label = "right" if dx > 0 else "left"
    else:
        screen_label = "down" if dy > 0 else "up"

    if mode == "geographic":
        labels = geographic_labels or {
            "up": "Northbound",
            "down": "Southbound",
            "left": "Westbound",
            "right": "Eastbound",
        }
        return labels.get(screen_label, "Unknown")

    display = {
        "up": "Moving Up",
        "down": "Moving Down",
        "left": "Moving Left",
        "right": "Moving Right",
    }
    return display[screen_label]
