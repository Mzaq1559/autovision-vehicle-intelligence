"""Speed estimation from a sequence of timestamped ground-plane positions.

The estimator converts pixel displacement between two timestamped points
into a real-world speed (km/h) using a :class:`PixelToMeterCalibrator`. It
deliberately does the minimum necessary — no perspective correction, no lane
geometry — and relies on the calibration reference being representative of
the measurement zone. This approximation, and how to reduce its error, is
documented in ``docs/calibration.md``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

from app.speed.calibration import PixelToMeterCalibrator

Point = Tuple[float, float]

MPS_TO_KMH = 3.6
MAX_VALID_SPEED_KMH = 250.0
MAX_VALID_DT_SECONDS = 1.5


def instantaneous_speed_kmh(
    p1: Point,
    t1: float,
    p2: Point,
    t2: float,
    calibrator: PixelToMeterCalibrator,
    max_valid_speed_kmh: float = MAX_VALID_SPEED_KMH,
    max_dt_s: float = MAX_VALID_DT_SECONDS,
) -> float:
    """Speed, in km/h, implied by moving from (p1, t1) to (p2, t2)."""

    dt = t2 - t1
    if dt <= 0 or dt > max_dt_s:
        return 0.0
    distance_m = calibrator.pixel_points_to_meters(p1, p2)
    speed_mps = distance_m / dt
    speed_kmh = speed_mps * MPS_TO_KMH
    if speed_kmh > max_valid_speed_kmh:
        return 0.0
    return speed_kmh


@dataclass
class SpeedEstimator:
    """Maintains a smoothed speed estimate for a single tracked vehicle.

    Args:
        calibrator: Pixel-to-meter scale for the scene.
        smoothing_window: Number of most-recent instantaneous samples to
            average for the reported "current" speed, reducing jitter from
            detection/tracking noise.
        min_samples_for_estimate: Minimum position samples required before a
            speed is reported at all (avoids noisy single-frame estimates).
    """

    calibrator: PixelToMeterCalibrator
    smoothing_window: int = 5
    min_samples_for_estimate: int = 3

    _samples: Deque[float] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if self.min_samples_for_estimate < 1:
            raise ValueError("min_samples_for_estimate must be >= 1")

    def update(self, p1: Point, t1: float, p2: Point, t2: float) -> Optional[float]:
        """Feed a new displacement sample; returns the smoothed speed so far.

        Returns ``None`` until ``min_samples_for_estimate`` samples have been
        collected, signalling "not enough data yet" rather than a
        misleadingly confident number.
        """

        dt = t2 - t1
        if dt <= 0 or dt > MAX_VALID_DT_SECONDS:
            if len(self._samples) < self.min_samples_for_estimate:
                return None
            return sum(self._samples) / len(self._samples)

        speed = instantaneous_speed_kmh(p1, t1, p2, t2, self.calibrator)
        if speed > 0.0:
            self._samples.append(speed)
            while len(self._samples) > self.smoothing_window:
                self._samples.popleft()

        if len(self._samples) < self.min_samples_for_estimate:
            return None
        return sum(self._samples) / len(self._samples)

    def reset(self) -> None:
        self._samples.clear()


def is_speed_violation(speed_kmh: float, speed_limit_kmh: float) -> bool:
    return speed_kmh > speed_limit_kmh
