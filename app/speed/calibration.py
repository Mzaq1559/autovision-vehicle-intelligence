"""Scene calibration: converting pixel distances into real-world meters.

Monocular cameras have no inherent sense of scale. AutoVision does not
pretend otherwise — it requires the operator to supply a calibration
reference (two points a known real-world distance apart, e.g. lane markings
measured on-site) and uses that single scale factor to convert pixel
displacement into meters. This is a linear approximation that assumes the
measurement zone is roughly perpendicular to the camera's viewing axis and
close to the calibrated reference; see docs/calibration.md for guidance and
caveats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

Point = Tuple[float, float]


class InvalidCalibrationError(ValueError):
    """Raised when calibration inputs cannot produce a usable scale factor."""


@dataclass
class PixelToMeterCalibrator:
    """Derives a meters-per-pixel scale factor from a reference measurement.

    Args:
        reference_points_px: Two (x, y) pixel coordinates known to be
            ``reference_distance_m`` meters apart in the real world.
        reference_distance_m: The real-world distance, in meters, between
            the two reference points.
    """

    reference_points_px: Sequence[Point]
    reference_distance_m: float

    def __post_init__(self) -> None:
        if self.reference_distance_m <= 0:
            raise InvalidCalibrationError("reference_distance_m must be positive")
        if len(self.reference_points_px) != 2:
            raise InvalidCalibrationError(
                "reference_points_px must contain exactly two points"
            )
        p1, p2 = self.reference_points_px
        pixel_distance = math.dist(p1, p2)
        if pixel_distance <= 0:
            raise InvalidCalibrationError(
                "reference_points_px must not be identical points"
            )
        self._meters_per_pixel = self.reference_distance_m / pixel_distance

    @property
    def meters_per_pixel(self) -> float:
        return self._meters_per_pixel

    def pixels_to_meters(self, pixel_distance: float) -> float:
        if pixel_distance < 0:
            raise InvalidCalibrationError("pixel_distance must be non-negative")
        return pixel_distance * self._meters_per_pixel

    def pixel_points_to_meters(self, p1: Point, p2: Point) -> float:
        return self.pixels_to_meters(math.dist(p1, p2))
