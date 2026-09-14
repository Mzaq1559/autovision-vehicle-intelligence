import math

import pytest

from app.speed.calibration import InvalidCalibrationError, PixelToMeterCalibrator
from app.speed.estimator import SpeedEstimator, instantaneous_speed_kmh, is_speed_violation


def test_calibrator_meters_per_pixel():
    # 100 px represents 10 real-world meters => 0.1 m/px
    calibrator = PixelToMeterCalibrator(
        reference_points_px=[(0.0, 0.0), (100.0, 0.0)], reference_distance_m=10.0
    )
    assert calibrator.meters_per_pixel == pytest.approx(0.1)


def test_calibrator_pixels_to_meters():
    calibrator = PixelToMeterCalibrator(
        reference_points_px=[(0.0, 0.0), (50.0, 0.0)], reference_distance_m=5.0
    )
    assert calibrator.pixels_to_meters(50.0) == pytest.approx(5.0)
    assert calibrator.pixels_to_meters(0.0) == pytest.approx(0.0)


def test_calibrator_rejects_zero_distance_points():
    with pytest.raises(InvalidCalibrationError):
        PixelToMeterCalibrator(
            reference_points_px=[(10.0, 10.0), (10.0, 10.0)], reference_distance_m=5.0
        )


def test_calibrator_rejects_non_positive_reference_distance():
    with pytest.raises(InvalidCalibrationError):
        PixelToMeterCalibrator(
            reference_points_px=[(0.0, 0.0), (10.0, 0.0)], reference_distance_m=0.0
        )


def test_instantaneous_speed_known_values():
    # Calibration: 10 px == 1 meter => 0.1 m/px
    calibrator = PixelToMeterCalibrator(
        reference_points_px=[(0.0, 0.0), (10.0, 0.0)], reference_distance_m=1.0
    )
    # Vehicle moves 100 px in 1 second => 10 meters in 1s => 10 m/s => 36 km/h
    speed = instantaneous_speed_kmh((0.0, 0.0), 0.0, (100.0, 0.0), 1.0, calibrator)
    assert speed == pytest.approx(36.0)


def test_instantaneous_speed_zero_or_negative_dt_returns_zero():
    calibrator = PixelToMeterCalibrator(
        reference_points_px=[(0.0, 0.0), (10.0, 0.0)], reference_distance_m=1.0
    )
    assert instantaneous_speed_kmh((0.0, 0.0), 1.0, (50.0, 0.0), 1.0, calibrator) == 0.0
    assert instantaneous_speed_kmh((0.0, 0.0), 1.0, (50.0, 0.0), 0.5, calibrator) == 0.0


def test_speed_estimator_requires_minimum_samples():
    calibrator = PixelToMeterCalibrator(
        reference_points_px=[(0.0, 0.0), (10.0, 0.0)], reference_distance_m=1.0
    )
    estimator = SpeedEstimator(calibrator=calibrator, smoothing_window=5, min_samples_for_estimate=3)

    assert estimator.update((0.0, 0.0), 0.0, (10.0, 0.0), 1.0) is None
    assert estimator.update((10.0, 0.0), 1.0, (20.0, 0.0), 2.0) is None
    result = estimator.update((20.0, 0.0), 2.0, (30.0, 0.0), 3.0)
    assert result is not None
    assert result == pytest.approx(3.6)


def test_speed_estimator_smooths_over_window():
    calibrator = PixelToMeterCalibrator(
        reference_points_px=[(0.0, 0.0), (10.0, 0.0)], reference_distance_m=1.0
    )
    estimator = SpeedEstimator(calibrator=calibrator, smoothing_window=2, min_samples_for_estimate=1)

    # First sample: 10 px in 1s -> 3.6 km/h
    first = estimator.update((0.0, 0.0), 0.0, (10.0, 0.0), 1.0)
    assert first == pytest.approx(3.6)

    # Second sample: 20 px in 1s -> 7.2 km/h; window=2 averages the two
    second = estimator.update((10.0, 0.0), 1.0, (30.0, 0.0), 2.0)
    assert second == pytest.approx((3.6 + 7.2) / 2)


def test_speed_estimator_rejects_invalid_window():
    calibrator = PixelToMeterCalibrator(
        reference_points_px=[(0.0, 0.0), (10.0, 0.0)], reference_distance_m=1.0
    )
    with pytest.raises(ValueError):
        SpeedEstimator(calibrator=calibrator, smoothing_window=0)


def test_is_speed_violation():
    assert is_speed_violation(70.0, 60.0) is True
    assert is_speed_violation(60.0, 60.0) is False
    assert is_speed_violation(59.9, 60.0) is False


def test_instantaneous_speed_rejects_extreme_speed_or_large_dt():
    calibrator = PixelToMeterCalibrator(
        reference_points_px=[(0.0, 0.0), (10.0, 0.0)], reference_distance_m=1.0
    )
    # Huge jump: 500 meters in 1s => 1800 km/h -> should return 0.0
    huge_speed = instantaneous_speed_kmh((0.0, 0.0), 0.0, (5000.0, 0.0), 1.0, calibrator)
    assert huge_speed == 0.0

    # Large gap: dt = 2.0s (> 1.5s limit) -> should return 0.0
    large_dt = instantaneous_speed_kmh((0.0, 0.0), 0.0, (10.0, 0.0), 2.0, calibrator)
    assert large_dt == 0.0


def test_speed_estimator_ignores_anomaly_spikes():
    calibrator = PixelToMeterCalibrator(
        reference_points_px=[(0.0, 0.0), (10.0, 0.0)], reference_distance_m=1.0
    )
    estimator = SpeedEstimator(calibrator=calibrator, smoothing_window=2, min_samples_for_estimate=1)
    estimator.update((0.0, 0.0), 0.0, (10.0, 0.0), 1.0)  # 3.6 km/h

    # Extreme jump (e.g. tracking ID swap): 5000 px in 0.033s => discarded
    res = estimator.update((10.0, 0.0), 1.0, (5010.0, 0.0), 1.033)
    assert res == pytest.approx(3.6)
