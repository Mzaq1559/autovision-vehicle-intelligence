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


def _get_calibrator(
    config: AppConfig,
    source_w: int = 0,
    source_h: int = 0,
    display_w: int = 0,
    display_h: int = 0,
) -> Optional[PixelToMeterCalibrator]:
    try:
        raw_points = [tuple(p) for p in config.calibration.reference_points_px]
        if source_w > 0 and display_w > 0 and (source_w != display_w or source_h != display_h):
            scale_x = display_w / float(source_w)
            scale_y = display_h / float(source_h) if source_h > 0 else scale_x
            points = [(px * scale_x, py * scale_y) for (px, py) in raw_points]
        else:
            points = raw_points
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

    with st.sidebar.expander("Performance settings"):
        config.video.processing_width = st.sidebar.select_slider(
            "Inference width (px)",
            options=[0, 480, 640, 960, 1280],
            value=int(config.video.processing_width),
            help="Resolution used for YOLO inference. 0 = same as display width. "
                 "Lower values are faster; detection boxes are scaled back to "
                 "display resolution automatically.",
        )
        config.video.ui_update_interval = st.sidebar.number_input(
            "UI update every N frames",
            min_value=1,
            max_value=60,
            value=int(config.video.ui_update_interval),
            step=1,
            help="Metrics and the vehicle table refresh every N processed frames. "
                 "The annotated video frame always updates every processed frame.",
        )
        config.video.frame_skip = st.sidebar.number_input(
            "Frame skip",
            min_value=0,
            max_value=10,
            value=int(config.video.frame_skip),
            step=1,
            help="Drop N source frames between each processed frame. "
                 "0 = process every frame (default, safest for tracking). "
                 "1 = process every 2nd frame. Keep conservative to avoid losing IDs.",
        )

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
    st.session_state.processing_run_id = st.session_state.get("processing_run_id", 0) + 1

    analytics: TrafficAnalytics = st.session_state.analytics
    analytics.reset()
    analytics.speed_limit_kmh = config.speed.limit_kmh
    analytics.line_y_fraction = config.measurement_zone.line_y_fraction
    if "speed_estimators" in st.session_state:
        st.session_state.speed_estimators.clear()

    display_w = source.width
    display_h = source.height
    if config.video.max_width > 0 and source.width > config.video.max_width:
        display_w = config.video.max_width
        display_h = int(source.height * (config.video.max_width / float(source.width)))

    calibrator = _get_calibrator(
        config,
        source_w=source.width,
        source_h=source.height,
        display_w=display_w,
        display_h=display_h,
    )
    if calibrator is None:
        return

    tracker = VehicleTracker(config)

    # Pipeline configuration.
    ui_interval: int = max(1, int(config.video.ui_update_interval))
    frame_skip: int = max(0, int(config.video.frame_skip))
    # step = every (frame_skip+1)-th frame is processed; others are dropped.
    step: int = frame_skip + 1

    # Determine whether a separate inference resolution is requested.
    # processing_width == 0 means "use display resolution" (no extra resize).
    proc_width: int = max(0, int(config.video.processing_width))

    # Persistent Streamlit placeholders — created once, reused every frame.
    frame_slot = st.empty()
    stats_slot = st.empty()
    table_slot = st.empty()
    diag_slot = st.empty()   # lightweight diagnostics panel

    stop_button = st.button("Stop processing")

    # Performance diagnostics — updated cheaply, rendered only on UI-update frames.
    proc_start_wall = time.time()
    processed_frames = 0
    skipped_frames = 0
    ui_updates = 0
    last_snapshot: Optional[object] = None   # type: ignore[type-arg]

    for frame_index, frame in source.frames():
        if stop_button:
            break

        # ── Frame skipping ────────────────────────────────────────────────
        # Drop frames that are not on the processing cadence.  The original
        # frame_index is preserved so video-time calculations are always correct.
        if step > 1 and frame_index % step != 0:
            skipped_frames += 1
            continue

        # ── Display-resolution resize (for output / overlay) ──────────────
        display_frame = resize_keep_aspect(frame, config.video.max_width)
        display_h, display_w = display_frame.shape[:2]

        # ── Inference-resolution resize (optional) ────────────────────────
        # When processing_width is set and smaller than the display width,
        # run YOLO on a downscaled copy.  After tracking, all bounding-box
        # coordinates are scaled back to display resolution so that overlays,
        # speed estimation, and trajectory positions are all in display space.
        if proc_width > 0 and proc_width < display_w:
            infer_frame = resize_keep_aspect(display_frame, proc_width)
            infer_h, infer_w = infer_frame.shape[:2]
            coord_scale_x = display_w / infer_w
            coord_scale_y = display_h / infer_h
        else:
            infer_frame = display_frame
            coord_scale_x = 1.0
            coord_scale_y = 1.0

        # ── Video timestamp (not wall-clock) ──────────────────────────────
        # Using the original source frame_index ensures speed estimation uses
        # real video-time intervals regardless of CPU processing speed.
        video_ts: float = frame_index / source.fps if source.fps > 0 else float(frame_index) / 30.0

        try:
            detections = tracker.track_frame(infer_frame, frame_index, video_timestamp=video_ts)
        except RuntimeError as exc:
            st.error(str(exc))
            break

        # Scale detection boxes back to display resolution if we inferred on a
        # smaller frame.  Positions appended to VehicleTrack.positions below
        # will be in display space — consistent with what render_frame() draws.
        if coord_scale_x != 1.0 or coord_scale_y != 1.0:
            from app.models.vehicle import Detection as _Det
            scaled: list = []
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                scaled.append(
                    _Det(
                        track_id=det.track_id,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        bbox=(
                            x1 * coord_scale_x,
                            y1 * coord_scale_y,
                            x2 * coord_scale_x,
                            y2 * coord_scale_y,
                        ),
                        frame_index=det.frame_index,
                        timestamp=det.timestamp,
                    )
                )
            detections = scaled

        # ── Analytics & speed estimation ──────────────────────────────────
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

            analytics.maybe_count(track, display_frame.shape[0])

        # Staleness check uses video time so it reflects elapsed video seconds,
        # not wall-clock seconds (which would be wrong on a slow CPU).
        active_ids = analytics.active_track_ids(STALE_TRACK_SECONDS, now=video_ts)

        # Decide whether this is a UI-update frame.
        # Always update on the very first processed frame (processed_frames == 0
        # before the increment below) so the UI is never blank for the first
        # ui_interval frames.
        is_ui_frame = (processed_frames % ui_interval == 0)

        # Snapshot: always compute (for overlay text) but only record history
        # on UI-update frames to keep chart data meaningful.
        snapshot = analytics.snapshot(active_ids, record_history=is_ui_frame)
        last_snapshot = snapshot
        processed_frames += 1

        # ── Rendering ─────────────────────────────────────────────────────
        tracks_by_id = {tid: analytics.tracks[tid] for tid in active_ids}
        rendered = render_frame(
            display_frame,
            detections,
            tracks_by_id,
            config.measurement_zone.line_y_fraction,
            show_trajectories=st.session_state.get("show_trajectories", True),
            summary_lines=[
                f"Active: {snapshot.active_vehicles}  Total: {snapshot.total_counted}",
                f"Avg speed: {snapshot.average_speed_kmh} km/h  Violations: {snapshot.violations}",
            ],
        )

        # ── Video frame: update EVERY processed frame ──────────────────────
        # Using the persistent frame_slot placeholder means Streamlit updates
        # the existing browser element rather than appending a new one.
        frame_slot.image(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB), channels="RGB")

        # ── Metrics / table / diagnostics: update periodically ─────────────
        if is_ui_frame:
            ui_updates += 1
            _render_stats(stats_slot, snapshot)
            _render_table(table_slot, analytics, active_ids)
            _render_diagnostics(
                diag_slot,
                source_fps=source.fps,
                processed=processed_frames,
                skipped=skipped_frames,
                elapsed_wall=time.time() - proc_start_wall,
                ui_updates=ui_updates,
            )

    source.release()

    # ── Final UI flush after the loop ─────────────────────────────────────
    # Ensure metrics, table, and diagnostics reflect the final state even if
    # the last frame was not a UI-update frame.
    if last_snapshot is not None:
        final_active_ids = analytics.active_track_ids(
            STALE_TRACK_SECONDS,
            now=video_ts if 'video_ts' in dir() else time.time(),  # type: ignore[name-defined]
        )
        # Force a final history record for the charts.
        final_snapshot = analytics.snapshot(final_active_ids, record_history=True)
        _render_stats(stats_slot, final_snapshot)
        _render_table(table_slot, analytics, final_active_ids)
        _render_diagnostics(
            diag_slot,
            source_fps=source.fps,
            processed=processed_frames,
            skipped=skipped_frames,
            elapsed_wall=time.time() - proc_start_wall,
            ui_updates=ui_updates,
        )


def _render_diagnostics(
    slot,
    source_fps: float,
    processed: int,
    skipped: int,
    elapsed_wall: float,
    ui_updates: int,
) -> None:
    with slot.container():
        fps_proc = (processed / elapsed_wall) if elapsed_wall > 0 else 0.0
        st.caption(
            f"⚡ Processing Performance: **{fps_proc:.1f} FPS** "
            f"(Processed: {processed} frames | Skipped: {skipped} frames | UI refreshes: {ui_updates} | Source FPS: {source_fps:.1f})"
        )


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
