"""Standalone, tracking-free vehicle detection.

This wraps a raw YOLO forward pass for cases where only per-frame detections
are needed (e.g. previewing the model, sanity-checking a calibration frame,
or unit-testing detection-adjacent utilities without invoking a tracker).
The live pipeline used by the Streamlit app instead goes through
:class:`app.tracking.tracker.VehicleTracker`, which calls Ultralytics'
fused detect+track API directly for efficiency — see that module's
docstring for why detection and tracking are combined there.
"""

from __future__ import annotations

import logging
import time
from typing import List

from app.config.settings import AppConfig
from app.models.vehicle import Detection

logger = logging.getLogger(__name__)


class VehicleDetector:
    """Runs YOLO object detection (no identity tracking) on a single frame."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._model = None

    def _ensure_model_loaded(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'ultralytics' package is required for detection but is "
                "not installed. Run `pip install -r requirements.txt`."
            ) from exc
        self._model = YOLO(self.config.model.weights)
        return self._model

    def detect_frame(self, frame, frame_index: int = 0) -> List[Detection]:
        model = self._ensure_model_loaded()
        allowed_classes = set(c.lower() for c in self.config.model.classes)

        results = model.predict(
            frame,
            conf=self.config.model.confidence,
            iou=self.config.model.iou_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        timestamp = time.time()
        detections: List[Detection] = []

        for box, cls_idx, conf in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
        ):
            class_name = (
                names.get(int(cls_idx), str(int(cls_idx)))
                if isinstance(names, dict)
                else str(names[int(cls_idx)])
            ).lower()
            if allowed_classes and class_name not in allowed_classes:
                continue
            detections.append(
                Detection(
                    track_id=-1,
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=tuple(box),  # type: ignore[arg-type]
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )
        return detections
