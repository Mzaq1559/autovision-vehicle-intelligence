"""Video/camera source handling.

Wraps OpenCV's VideoCapture with clearer error messages and a couple of
convenience helpers (FPS resolution, frame resizing) used by both the
Streamlit app and, potentially, a future CLI entry point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class VideoSourceError(RuntimeError):
    """Raised for any problem opening or reading a video/camera source."""


@dataclass
class VideoSource:
    """Opens a video file or camera index and yields frames.

    Use as a context manager::

        with VideoSource.from_file("clip.mp4") as source:
            for frame_index, frame in source.frames():
                ...
    """

    capture: cv2.VideoCapture
    fps: float
    width: int
    height: int
    frame_count: Optional[int] = None

    @classmethod
    def from_file(cls, path: str, default_fps: float = 30.0) -> "VideoSource":
        file_path = Path(path)
        if not file_path.exists():
            raise VideoSourceError(f"Video file not found: {path}")
        if file_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise VideoSourceError(
                f"Unsupported video format '{file_path.suffix}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
            )

        capture = cv2.VideoCapture(str(file_path))
        if not capture.isOpened():
            raise VideoSourceError(
                f"Could not open video file '{path}'. It may be corrupted or "
                "encoded with an unsupported codec."
            )
        return cls._from_capture(capture, default_fps)

    @classmethod
    def from_camera(cls, index: int = 0, default_fps: float = 30.0) -> "VideoSource":
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise VideoSourceError(
                f"Could not open camera at index {index}. It may be in use by "
                "another application, unavailable in this environment (e.g. a "
                "headless server or container), or require OS camera "
                "permissions. See README 'Webcam' section for the browser-"
                "based fallback."
            )
        return cls._from_capture(capture, default_fps)

    @classmethod
    def _from_capture(cls, capture: cv2.VideoCapture, default_fps: float) -> "VideoSource":
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if not fps or fps != fps or fps <= 0 or fps > 120.0:  # NaN-safe and sane-range check
            fps = default_fps
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        raw_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = raw_count if raw_count > 0 else None
        return cls(capture=capture, fps=fps, width=width, height=height, frame_count=frame_count)

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                break
            yield index, frame
            index += 1

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def resize_keep_aspect(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
