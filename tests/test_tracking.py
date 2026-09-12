import pytest

from app.models.vehicle import Detection, VehicleTrack
from app.tracking.tracker import estimate_direction


def test_estimate_direction_needs_two_points():
    assert estimate_direction([]) == "Unknown"
    assert estimate_direction([(0.0, 0.0)]) == "Unknown"


def test_estimate_direction_stationary_below_threshold():
    trajectory = [(100.0, 100.0), (101.0, 100.0), (102.0, 101.0)]
    assert estimate_direction(trajectory, min_displacement_px=6.0) == "Stationary"


def test_estimate_direction_screen_relative_horizontal():
    trajectory = [(0.0, 100.0), (50.0, 101.0)]
    assert estimate_direction(trajectory) == "Moving Right"

    trajectory_left = [(50.0, 100.0), (0.0, 101.0)]
    assert estimate_direction(trajectory_left) == "Moving Left"


def test_estimate_direction_screen_relative_vertical():
    trajectory_down = [(100.0, 0.0), (101.0, 50.0)]
    assert estimate_direction(trajectory_down) == "Moving Down"

    trajectory_up = [(100.0, 50.0), (101.0, 0.0)]
    assert estimate_direction(trajectory_up) == "Moving Up"


def test_estimate_direction_geographic_mode():
    trajectory = [(0.0, 100.0), (50.0, 101.0)]
    result = estimate_direction(trajectory, mode="geographic")
    assert result == "Eastbound"

    trajectory_up = [(100.0, 50.0), (101.0, 0.0)]
    assert estimate_direction(trajectory_up, mode="geographic") == "Northbound"


def test_estimate_direction_custom_geographic_labels():
    trajectory = [(0.0, 100.0), (50.0, 101.0)]
    labels = {"up": "A", "down": "B", "left": "C", "right": "D"}
    assert estimate_direction(trajectory, mode="geographic", geographic_labels=labels) == "D"


def test_vehicle_track_update_detection_trims_trajectory():
    track = VehicleTrack(
        track_id=1,
        vehicle_type="car",
        confidence=0.9,
        first_seen=0.0,
        last_seen=0.0,
        trajectory_length=3,
    )
    for i in range(5):
        det = Detection(
            track_id=1,
            class_name="car",
            confidence=0.9,
            bbox=(float(i), 0.0, float(i) + 10, 10.0),
            frame_index=i,
            timestamp=float(i),
        )
        track.update_detection(det)

    assert len(track.positions) == 3
    assert track.last_seen == 4.0


def test_vehicle_track_record_speed_sample_updates_stats():
    track = VehicleTrack(
        track_id=1, vehicle_type="car", confidence=0.9, first_seen=0.0, last_seen=0.0
    )
    track.record_speed_sample(30.0)
    track.record_speed_sample(50.0)

    assert track.current_speed_kmh == 50.0
    assert track.max_speed_kmh == 50.0
    assert track.average_speed_kmh == pytest.approx(40.0)


def test_detection_center_and_foot_point():
    det = Detection(
        track_id=1,
        class_name="car",
        confidence=0.9,
        bbox=(0.0, 0.0, 10.0, 20.0),
        frame_index=0,
        timestamp=0.0,
    )
    assert det.center == (5.0, 10.0)
    assert det.foot_point == (5.0, 20.0)
