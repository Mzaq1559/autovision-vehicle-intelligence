"""Configuration loading and validation for AutoVision.

Configuration is expressed as YAML (see ``configs/default.yaml``) and loaded
into a set of small, typed dataclasses. Nothing in the rest of the codebase
should hardcode thresholds, paths, or calibration numbers — everything flows
through this module so a user can adapt the system to a new camera/scene
without touching source code.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    weights: str = "yolov8n.pt"
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    confidence: float = 0.35
    iou_threshold: float = 0.45
    classes: List[str] = field(
        default_factory=lambda: ["car", "motorcycle", "bus", "truck"]
    )


@dataclass
class TrackerConfig:
    type: str = "bytetrack"  # "bytetrack" | "botsort"
    track_buffer: int = 30
    match_threshold: float = 0.8
    trajectory_length: int = 40


@dataclass
class CalibrationConfig:
    # Real-world distance, in meters, that the measurement zone represents.
    reference_distance_m: float = 20.0
    # Pixel coordinates (x, y) of the two points that define the reference
    # distance above, e.g. two lane-marking points measured on-site.
    reference_points_px: List[List[float]] = field(
        default_factory=lambda: [[100.0, 600.0], [100.0, 200.0]]
    )
    fps: Optional[float] = None  # None => read FPS from the video/stream


@dataclass
class MeasurementZoneConfig:
    # A horizontal line (as a fraction of frame height, 0-1) vehicles must
    # cross to register a speed/count sample. Kept simple and configurable
    # rather than assuming a fixed resolution.
    line_y_fraction: float = 0.6
    line_thickness_px: int = 4


@dataclass
class SpeedConfig:
    limit_kmh: float = 60.0
    smoothing_window: int = 5
    min_samples_for_estimate: int = 3


@dataclass
class DirectionConfig:
    mode: str = "screen_relative"  # "screen_relative" | "geographic"
    # Used only when mode == "geographic"; maps a screen-space bearing name
    # to a geographic label.
    geographic_labels: Dict[str, str] = field(
        default_factory=lambda: {
            "up": "Northbound",
            "down": "Southbound",
            "left": "Westbound",
            "right": "Eastbound",
        }
    )
    min_displacement_px: float = 6.0


@dataclass
class CountingConfig:
    line_y_fraction: float = 0.6
    line_thickness_px: int = 4


@dataclass
class VideoConfig:
    default_fps: float = 30.0
    max_width: int = 960
    # Resolution for YOLO inference. 0 = same as max_width (no extra resize).
    # When non-zero, frames are resized to this width before model.track() and
    # scaled back to max_width for display.  All coordinates (boxes, positions)
    # are expressed in display-resolution space after the inverse scale.
    processing_width: int = 0
    # How many processed frames to skip between Streamlit metrics/table refreshes.
    # The annotated video frame (frame_slot.image) still updates every frame.
    ui_update_interval: int = 5
    # Number of source frames dropped between each processed frame.
    #   0 = every frame (default)
    #   1 = every 2nd frame
    #   2 = every 3rd frame
    # Original frame_index values are preserved for correct video-time maths.
    frame_skip: int = 0


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    measurement_zone: MeasurementZoneConfig = field(
        default_factory=MeasurementZoneConfig
    )
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    log_level: str = "INFO"


def _merge_into_dataclass(instance: Any, data: Dict[str, Any]) -> Any:
    """Recursively merge a dict of overrides into a dataclass instance."""

    if not data:
        return instance

    for f in fields(instance):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(instance, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(instance, f.name, _merge_into_dataclass(current, value))
        else:
            setattr(instance, f.name, value)
    return instance


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML, falling back to defaults.

    Environment variables prefixed with ``AUTOVISION_`` override individual
    top-level scalar fields, e.g. ``AUTOVISION_LOG_LEVEL=DEBUG``.
    """

    cfg = AppConfig()

    resolved_path = path or os.environ.get("AUTOVISION_CONFIG")
    if resolved_path:
        cfg = load_config_from_file(resolved_path)

    env_override = os.environ.get("AUTOVISION_LOG_LEVEL")
    if env_override:
        cfg.log_level = env_override

    return cfg


def load_config_from_file(path: str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration file {path} must contain a YAML mapping")

    base = AppConfig()
    return _merge_into_dataclass(base, copy.deepcopy(raw))
