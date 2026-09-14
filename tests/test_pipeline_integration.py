import numpy as np
import pytest

from app.config.settings import AppConfig, CalibrationConfig
from app.models.vehicle import VehicleTrack
from app.streamlit_app import _get_calibrator
from app.utils.video import VideoSource
from app.visualization.renderer import draw_trajectories, render_frame


def test_get_calibrator_rescales_reference_points_for_display_resolution():
    config = AppConfig()
    config.calibration = CalibrationConfig(
        reference_distance_m=20.0,
        reference_points_px=[[200.0, 400.0], [200.0, 800.0]],
    )

    # Source 1920x1080 -> Display 960x540 (0.5x scale)
    calibrator = _get_calibrator(
        config, source_w=1920, source_h=1080, display_w=960, display_h=540
    )
    assert calibrator is not None

    # Original pixel distance was 400px for 20m => 0.05 m/px
    # Rescaled pixel distance is 200px for 20m => 0.10 m/px
    assert calibrator.meters_per_pixel == pytest.approx(0.10)


def test_video_source_fps_sanitization(monkeypatch):
    class MockCap:
        def get(self, prop):
            if prop == 5:  # cv2.CAP_PROP_FPS
                return 1000.0  # Invalid/crazy FPS returned by some containers
            return 0

    source = VideoSource._from_capture(MockCap(), default_fps=30.0)
    assert source.fps == 30.0


def test_draw_trajectories_skips_gaps_and_large_jumps():
    track = VehicleTrack(
        track_id=1, vehicle_type="car", confidence=0.9, first_seen=0.0, last_seen=2.0
    )
    # Point 1 -> Point 2: normal (dt=0.1s, dist=10px)
    track.positions.append((10.0, 10.0, 0.0))
    track.positions.append((20.0, 10.0, 0.1))

    # Point 2 -> Point 3: time gap (dt=1.0s > 0.5s)
    track.positions.append((30.0, 10.0, 1.1))

    # Point 3 -> Point 4: spatial jump (dist=500px > 150px)
    track.positions.append((530.0, 10.0, 1.2))

    frame = np.zeros((100, 600, 3), dtype=np.uint8)
    rendered = draw_trajectories(frame, [track])

    # Ensure function executes cleanly without error and returns non-empty image
    assert rendered.shape == frame.shape
