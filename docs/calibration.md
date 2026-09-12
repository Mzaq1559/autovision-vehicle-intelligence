# Calibration Guide

AutoVision estimates speed from **pixel displacement over time**. A single
camera has no inherent sense of real-world scale, so AutoVision cannot know
how many meters a pixel represents unless you tell it. This document explains
how to calibrate a scene and what the resulting speed numbers do and do not
mean.

## Why calibration is required

Without calibration, "the vehicle moved 40 pixels in 0.2 seconds" is not a
speed — it's just a pixel rate. To turn it into km/h, AutoVision needs a
**scale factor**: meters per pixel. That scale factor is derived from one
reference measurement that you provide: two pixel points in the frame that
you know the real-world distance between.

## Step 1 — Choose two reference points

Pick two points in your video frame whose real-world distance you can
measure or already know. Good choices:

- Two lane-marking dashes a known distance apart (many jurisdictions use
  standard dash/gap lengths).
- Two fixed objects (poles, curb markings, painted lines) you can physically
  measure with a tape measure or find on a site plan/satellite image.
- A road width or crosswalk width, if you can obtain an accurate figure.

The two points should sit close to the **measurement zone** (the line
vehicles cross for speed sampling — see `measurement_zone.line_y_fraction`
in the config), and ideally at roughly the same depth from the camera as
that zone. Calibration accuracy degrades the further a vehicle's actual
path is from the calibrated reference, because a single scale factor cannot
capture full camera perspective.

## Step 2 — Find their pixel coordinates

Open a representative frame (e.g. `cv2.imwrite` a frame, or a paused video
player) and read off the `(x, y)` pixel coordinates of your two reference
points. Coordinates start at `(0, 0)` in the top-left corner.

## Step 3 — Fill in the config

Edit `configs/default.yaml` (or your own copy):

```yaml
calibration:
  reference_distance_m: 20.0        # real-world distance between the two points, in meters
  reference_points_px:
    - [100.0, 600.0]                 # point 1 (x, y)
    - [100.0, 200.0]                 # point 2 (x, y)
  fps: null                          # null = read FPS from the source
```

`reference_distance_m` is the real-world distance between the two points
above. `reference_points_px` are their pixel coordinates in your video.

## Step 4 — Set the measurement/counting line

`measurement_zone.line_y_fraction` (0–1) places a horizontal line as a
fraction of frame height. Vehicles are counted, and speed samples are most
meaningful, near this line. Put it where your calibration reference is
most accurate — typically mid-frame, away from strong perspective distortion
at the very top or bottom of the frame.

## What the resulting speed numbers mean — and don't mean

- Speeds are computed from the vehicle's tracked ground-plane point (the
  bottom-center of its bounding box) moving between frames, converted to
  meters using your single calibrated scale factor.
- This assumes the road is roughly flat and the calibration reference is
  representative of the vehicle's actual path. It does **not** perform full
  perspective/homography correction.
- Accuracy is best near the calibrated region and near the measurement
  line, and degrades toward the edges of the frame or at steep camera
  angles.
- Detection/tracking noise (a box jittering by a few pixels) also
  translates into speed noise; `speed.smoothing_window` in the config
  averages recent samples to reduce this, at the cost of some
  responsiveness.
- Treat AutoVision's speeds as **approximate, for analytics and
  educational purposes** — not as calibrated, legally admissible
  enforcement measurements. Real enforcement-grade systems typically use
  radar/lidar or carefully surveyed, homography-corrected camera rigs.

## Improving accuracy

- Use a longer, more precisely measured reference distance rather than a
  short one — small pixel-measurement errors matter less as a fraction of
  a longer reference.
- Keep the measurement line close to the calibrated reference points.
- Where possible, calibrate along the same direction vehicles travel
  (e.g. two points along the lane, not across it), since foreshortening
  differs by direction.
- For higher accuracy, a future improvement could add a full perspective
  transform (see the README's "Future Improvements" section) using four or
  more reference points instead of two.
