"""AutoVision Streamlit dashboard.

Run with:

    streamlit run app/streamlit_app.py

This module is intentionally "thin" — it wires together the detection,
tracking, speed-estimation, analytics, and visualization layers and renders
them with Streamlit + Plotly. Business logic lives in the other ``app.*``
modules so it stays independently testable.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.analytics.traffic import TrafficAnalytics
from app.config.settings import AppConfig, load_config
from app.speed.calibration import InvalidCalibrationError, PixelToMeterCalibrator
from app.speed.estimator import SpeedEstimator
from app.tracking.tracker import VehicleTracker, estimate_direction
from app.utils.video import VideoSource, VideoSourceError, resize_keep_aspect
from app.visualization.renderer import render_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("autovision")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
STALE_TRACK_SECONDS = 3.0


def _init_session_state(config: AppConfig) -> None:
    if "analytics" not in st.session_state:
        st.session_state.analytics = TrafficAnalytics(
            speed_limit_kmh=config.speed.limit_kmh,
            line_y_fraction=config.measurement_zone.line_y_fraction,
        )
    if "speed_estimators" not in st.session_state:
        st.session_state.speed_estimators = {}
    if "processing" not in st.session_state:
        st.session_state.processing = False


def _get_calibrator(config: AppConfig) -> Optional[PixelToMeterCalibrator]:
    try:
        points = [tuple(p) for p in config.calibration.reference_points_px]
        return PixelToMeterCalibrator(
            reference_points_px=points,
            reference_distance_m=config.calibration.reference_distance_m,
        )
    except InvalidCalibrationError as exc:
        st.sidebar.error(f"Invalid calibration configuration: {exc}")
        return None


def _sidebar_config(config: AppConfig) -> AppConfig:
    st.sidebar.header("Configuration")

    config.model.confidence = st.sidebar.slider(
        "Detection confidence", 0.05, 0.95, float(config.model.confidence), 0.05
    )
    config.model.iou_threshold = st.sidebar.slider(
        "IoU threshold", 0.1, 0.9, float(config.model.iou_threshold), 0.05
    )
    config.speed.limit_kmh = st.sidebar.number_input(
        "Speed limit (km/h)", min_value=1.0, value=float(config.speed.limit_kmh)
    )
    config.measurement_zone.line_y_fraction = st.sidebar.slider(
        "Measurement / counting line (fraction of frame height)",
        0.05,
        0.95,
        float(config.measurement_zone.line_y_fraction),
        0.05,
    )
    config.calibration.reference_distance_m = st.sidebar.number_input(
        "Calibration reference distance (m)",
        min_value=0.1,
        value=float(config.calibration.reference_distance_m),
        help="Real-world distance between the two calibration points. "
        "See docs/calibration.md.",
    )
    show_trajectories = st.sidebar.checkbox("Show trajectories", value=True)

    with st.sidebar.expander("About speed estimates"):
        st.caption(
            "Speeds are approximate. They come from pixel displacement across "
            "a single, user-calibrated reference distance and assume the "
            "measurement zone is roughly perpendicular to the camera and "
            "close to the calibrated reference plane. See docs/calibration.md."
        )

    st.session_state.show_trajectories = show_trajectories
    return config


def _process_video(source: VideoSource, config: AppConfig) -> None:
    analytics: TrafficAnalytics = st.session_state.analytics
    analytics.speed_limit_kmh = config.speed.limit_kmh
    analytics.line_y_fraction = config.measurement_zone.line_y_fraction

    calibrator = _get_calibrator(config)
    if calibrator is None:
        return

    tracker = VehicleTracker(config)

    frame_slot = st.empty()
    stats_slot = st.empty()
    table_slot = st.empty()

    stop_button = st.button("Stop processing")

    for frame_index, frame in source.frames():
        if stop_button:
            break

        frame = resize_keep_aspect(frame, config.video.max_width)
        try:
            detections = tracker.track_frame(frame, frame_index)
        except RuntimeError as exc:
            st.error(str(exc))
            break

        now = time.time()
        for det in detections:
            track = analytics.get_or_create_track(
                det.track_id,
                det.class_name,
                det.confidence,
                det.timestamp,
                config.tracker.trajectory_length,
            )
            track.update_detection(det)

            estimator = st.session_state.speed_estimators.get(det.track_id)
            if estimator is None:
                estimator = SpeedEstimator(
                    calibrator=calibrator,
                    smoothing_window=config.speed.smoothing_window,
                    min_samples_for_estimate=config.speed.min_samples_for_estimate,
                )
                st.session_state.speed_estimators[det.track_id] = estimator

            if len(track.positions) >= 2:
                (x1, y1, t1), (x2, y2, t2) = track.positions[-2], track.positions[-1]
                speed = estimator.update((x1, y1), t1, (x2, y2), t2)
                if speed is not None:
                    track.record_speed_sample(speed)
                    analytics.evaluate_violation(track)

            track.direction = estimate_direction(
                track.trajectory,
                mode=config.direction.mode,
                min_displacement_px=config.direction.min_displacement_px,
                geographic_labels=config.direction.geographic_labels,
            )

            analytics.maybe_count(track, frame.shape[0])

        active_ids = analytics.active_track_ids(STALE_TRACK_SECONDS, now=now)
        snapshot = analytics.snapshot(active_ids)

        tracks_by_id = {tid: analytics.tracks[tid] for tid in active_ids}
        rendered = render_frame(
            frame,
            detections,
            tracks_by_id,
            config.measurement_zone.line_y_fraction,
            show_trajectories=st.session_state.get("show_trajectories", True),
            summary_lines=[
                f"Active: {snapshot.active_vehicles}  Total: {snapshot.total_counted}",
                f"Avg speed: {snapshot.average_speed_kmh} km/h  Violations: {snapshot.violations}",
            ],
        )
        frame_slot.image(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB), channels="RGB")

        _render_stats(stats_slot, snapshot)
        _render_table(table_slot, analytics, active_ids)

    source.release()


def _render_stats(slot, snapshot) -> None:
    with slot.container():
        cols = st.columns(6)
        cols[0].metric("Total vehicles", snapshot.total_counted)
        cols[1].metric("Currently visible", snapshot.active_vehicles)
        cols[2].metric("Avg speed (km/h)", snapshot.average_speed_kmh)
        cols[3].metric("Max speed (km/h)", snapshot.max_speed_kmh)
        cols[4].metric("Violations", snapshot.violations)
        counts = snapshot.counts_by_type
        summary = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "—"
        cols[5].metric("By type", "")
        st.caption(f"By type — {summary}")


def _render_table(slot, analytics: TrafficAnalytics, active_ids) -> None:
    rows = analytics.vehicle_table_rows(active_ids)
    with slot.container():
        st.subheader("Tracked vehicles")
        if rows:
            df = pd.DataFrame(rows)
            df = df.rename(
                columns={
                    "track_id": "ID",
                    "type": "Type",
                    "confidence": "Confidence (%)",
                    "current_speed_kmh": "Current Speed",
                    "average_speed_kmh": "Average Speed",
                    "max_speed_kmh": "Max Speed",
                    "direction": "Direction",
                    "time_in_scene_s": "Time in Scene (s)",
                    "violation": "Violation",
                }
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No vehicles currently tracked.")


def _render_charts(analytics: TrafficAnalytics) -> None:
    st.subheader("Traffic analytics")
    if not analytics.history:
        st.caption("Run a video to populate analytics charts.")
        return

    history_df = pd.DataFrame(
        [
            {
                "timestamp": s.timestamp,
                "active_vehicles": s.active_vehicles,
                "average_speed_kmh": s.average_speed_kmh,
                "violations": s.violations,
            }
            for s in analytics.history
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        type_counts = analytics.type_distribution()
        if type_counts:
            fig = px.bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                labels={"x": "Vehicle type", "y": "Count"},
                title="Vehicles by type",
            )
            st.plotly_chart(fig, use_container_width=True)

        fig_speed_dist = px.histogram(
            [t.current_speed_kmh for t in analytics.tracks.values() if t.current_speed_kmh > 0],
            nbins=20,
            labels={"value": "Speed (km/h)"},
            title="Speed distribution",
        )
        st.plotly_chart(fig_speed_dist, use_container_width=True)

    with col2:
        fig_count = px.line(
            history_df, x="timestamp", y="active_vehicles", title="Vehicle count over time"
        )
        st.plotly_chart(fig_count, use_container_width=True)

        fig_avg_speed = px.line(
            history_df, x="timestamp", y="average_speed_kmh", title="Average speed over time"
        )
        st.plotly_chart(fig_avg_speed, use_container_width=True)

    fig_violations = px.line(
        history_df, x="timestamp", y="violations", title="Violations over time"
    )
    st.plotly_chart(fig_violations, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AutoVision — Vehicle Intelligence", layout="wide")
    st.title("AutoVision — Real-Time Vehicle Intelligence & Traffic Analytics")
    st.caption(
        "Vehicle detection, multi-object tracking, approximate speed estimation, "
        "and traffic analytics. Not for law enforcement or facial recognition."
    )

    try:
        config = load_config(str(DEFAULT_CONFIG_PATH))
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Failed to load configuration: {exc}")
        return

    config = _sidebar_config(config)
    _init_session_state(config)

    st.sidebar.header("Input source")
    mode = st.sidebar.radio("Select input mode", ["Video file", "Webcam", "Sample / demo"])

    source: Optional[VideoSource] = None
    try:
        if mode == "Video file":
            uploaded = st.sidebar.file_uploader(
                "Upload a video", type=["mp4", "avi", "mov", "mkv"]
            )
            if uploaded is not None:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                source = VideoSource.from_file(tmp_path, config.video.default_fps)

        elif mode == "Webcam":
            st.sidebar.caption(
                "Webcam access depends on a camera being available to the "
                "machine running this Streamlit process. In hosted/containerized "
                "environments this usually is not available — see README "
                "'Webcam' section for the fallback (upload a short clip instead)."
            )
            camera_index = st.sidebar.number_input("Camera index", min_value=0, value=0, step=1)
            if st.sidebar.button("Start webcam"):
                source = VideoSource.from_camera(int(camera_index), config.video.default_fps)

        else:  # Sample / demo
            st.sidebar.info(
                "No video is bundled with this repository to avoid shipping "
                "copyrighted media. Download a short traffic clip (e.g. a "
                "Creative-Commons traffic video) and place it in `assets/`, "
                "then select 'Video file' and upload it. See README for links."
            )

        if source is not None:
            _process_video(source, config)

    except VideoSourceError as exc:
        st.error(str(exc))

    st.divider()
    _render_charts(st.session_state.analytics)


if __name__ == "__main__":
    main()
