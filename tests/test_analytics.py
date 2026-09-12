import pytest

from app.analytics.traffic import TrafficAnalytics, has_crossed_line, should_count
from app.models.vehicle import VehicleTrack


def test_has_crossed_line_detects_crossing_either_direction():
    assert has_crossed_line(prev_y=90.0, curr_y=110.0, line_y=100.0) is True
    assert has_crossed_line(prev_y=110.0, curr_y=90.0, line_y=100.0) is True


def test_has_crossed_line_no_crossing():
    assert has_crossed_line(prev_y=10.0, curr_y=20.0, line_y=100.0) is False


def test_has_crossed_line_no_movement():
    assert has_crossed_line(prev_y=100.0, curr_y=100.0, line_y=100.0) is False


def test_should_count_only_once():
    track = VehicleTrack(track_id=1, vehicle_type="car", confidence=0.9, first_seen=0.0, last_seen=0.0)
    assert should_count(track, crossed=True) is True
    track.counted = True
    assert should_count(track, crossed=True) is False


def test_maybe_count_registers_exactly_once():
    analytics = TrafficAnalytics(speed_limit_kmh=60.0, line_y_fraction=0.5)
    track = analytics.get_or_create_track(1, "car", 0.9, timestamp=0.0, trajectory_length=10)

    track.positions.append((100.0, 40.0, 0.0))
    track.positions.append((100.0, 60.0, 1.0))  # crosses y=50 (frame_height=100 * 0.5)

    counted = analytics.maybe_count(track, frame_height=100.0)
    assert counted is True
    assert analytics.total_counted == 1
    assert analytics.counts_by_type["car"] == 1

    # A further update that doesn't cross again should not double-count.
    track.positions.append((100.0, 65.0, 2.0))
    counted_again = analytics.maybe_count(track, frame_height=100.0)
    assert counted_again is False
    assert analytics.total_counted == 1


def test_evaluate_violation_logs_once_and_marks_track():
    analytics = TrafficAnalytics(speed_limit_kmh=60.0)
    track = analytics.get_or_create_track(1, "car", 0.9, timestamp=0.0, trajectory_length=10)
    track.current_speed_kmh = 75.0

    is_violation = analytics.evaluate_violation(track)
    assert is_violation is True
    assert track.violation is True
    assert len(analytics.violation_log) == 1

    # Calling again with the same track shouldn't duplicate the log entry.
    analytics.evaluate_violation(track)
    assert len(analytics.violation_log) == 1


def test_evaluate_violation_false_under_limit():
    analytics = TrafficAnalytics(speed_limit_kmh=60.0)
    track = analytics.get_or_create_track(1, "car", 0.9, timestamp=0.0, trajectory_length=10)
    track.current_speed_kmh = 40.0

    assert analytics.evaluate_violation(track) is False
    assert track.violation is False
    assert analytics.violation_log == []


def test_snapshot_aggregates_active_tracks():
    analytics = TrafficAnalytics(speed_limit_kmh=60.0)
    t1 = analytics.get_or_create_track(1, "car", 0.9, timestamp=0.0, trajectory_length=10)
    t1.current_speed_kmh = 50.0
    t1.max_speed_kmh = 55.0
    t2 = analytics.get_or_create_track(2, "truck", 0.9, timestamp=0.0, trajectory_length=10)
    t2.current_speed_kmh = 30.0
    t2.max_speed_kmh = 35.0

    snap = analytics.snapshot(active_ids=[1, 2])

    assert snap.active_vehicles == 2
    assert snap.average_speed_kmh == pytest.approx(40.0)
    assert snap.max_speed_kmh == pytest.approx(55.0)
    assert len(analytics.history) == 1


def test_active_track_ids_filters_stale_tracks():
    analytics = TrafficAnalytics(speed_limit_kmh=60.0)
    analytics.get_or_create_track(1, "car", 0.9, timestamp=0.0, trajectory_length=10)
    analytics.tracks[1].last_seen = 0.0

    active_now = analytics.active_track_ids(stale_after_s=1.0, now=5.0)
    assert active_now == []

    active_recent = analytics.active_track_ids(stale_after_s=10.0, now=5.0)
    assert active_recent == [1]
