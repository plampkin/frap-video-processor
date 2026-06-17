# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-06-17

### Changed
- `process_videos.py`: front detection is now restricted to the **middle 50% of
  the frame** (`MONITOR_BAND_TOP_FRACTION` to `MONITOR_BAND_BOTTOM_FRACTION`,
  default 0.25–0.75). Otsu thresholding, reacted-class detection, and front
  tracking all operate only inside this band, removing the top-of-tube jostling /
  soldering-iron initiation and the bottom-of-tube artifacts that were preventing
  the front from being spotted.
- `process_videos.py`: the front is tracked every frame and may enter the band at
  **any time** — before it arrives there is no reacted region (`NaN`), so the
  speed fit starts whenever the front actually appears rather than assuming motion
  from the start of the video.
- `process_videos.py`: speed is fit over the band transit only, excluding the
  pre-entry frames and the fully-reacted bottom plateau via `EDGE_MARGIN_FRACTION`.
- `process_videos.py`: annotated video now draws the monitoring-band boundaries.

### Removed
- `process_videos.py`: `MIDDLE_BAND_START_FRACTION` / `MIDDLE_BAND_END_FRACTION`
  (band of *total detected travel*), replaced by the fixed middle-50%-of-frame
  monitoring band above.

## [0.3.0] - 2026-06-17

### Changed
- Project layout: input videos and generated outputs are now sorted into
  subdirectories — `input_videos/` (`.mov` inputs), `output_videos/`
  (`*_annotated.mp4`), `output_images/` (`*_position_time.png`), and
  `output_data/` (`*_position_time.csv`, `front_speed_results.csv`).
- `process_videos.py`: reads inputs from `input_videos/` and writes each output
  type to its dedicated directory (created automatically if absent).
- `process_videos.py`: front direction is now fixed to **downward** instead of
  auto-detected. Fronts are top-initiated with a soldering iron and propagate
  down the tube; auto-detection was misclassifying direction and causing every
  analysis to fail.
- `process_videos.py`: speed is now fit over the **middle chunk of the tube**
  (the middle band of total front travel, `MIDDLE_BAND_START_FRACTION` to
  `MIDDLE_BAND_END_FRACTION`, default 0.25–0.75) rather than a fixed time window.
  This excludes the initial test-tube jostling and initiation transient as well
  as the end-of-tube plateau, isolating the steady-state propagation region.

### Removed
- `process_videos.py`: time-based `FIT_START_FRACTION` / `FIT_END_FRACTION`
  fit window, replaced by the spatial middle-band selection above.

## [0.2.0] - 2026-06-17

### Added
- `process_videos.py`: Full pipeline for automated front speed extraction from `.mov` files.
  - Phase 1: Video metadata inspection (dimensions, FPS, frame count)
  - Phase 2: Grayscale conversion, Gaussian blur, global Otsu thresholding with morphological open+close
  - Phase 3: Vectorized column-wise front position extraction with minimum-width filter; temporal outlier rejection (>2× median displacement) and rolling-median smoothing
  - Phase 4: Linear regression (scipy.stats.linregress) over configurable fit window (FIT_START/END_FRACTION); outputs speed in px/s (or mm/s when CALIBRATION_MM_PER_PX is set) and R²
  - Phase 5: Annotated output video with front-position overlay line and speed label
  - Phase 6: Batch processing of all `.mov` files; parses filename metadata fields (monomer, initiator, amount, replicate, sample)
  - Failure detection: FAILED_NO_STABLE_FRONT (R²<0.80), FAILED_FRONT_STALLED (second-half slope <20% of first-half), FAILED_FRONT_DID_NOT_COMPLETE (front traveled <80% of ROI height)
  - Outputs per video: `*_position_time.csv`, `*_position_time.png`; aggregated `front_speed_results.csv`

## [0.1.1] - 2026-06-17

### Changed
- `PROJECT_PLAN.md`: Updated front speed units to mm/s (was cm/min or mm/s); removed "luminescent" descriptor for the front boundary (now "visible boundary")
- `PROJECT_PLAN.md`: Clarified calibration fallback — program reports px/s when no calibration is provided, mm/s when `CALIBRATION_MM_PER_PX` is set
- `PROJECT_PLAN.md`: Added noise/bubble mitigation plan in Phase 2 (width filter, morphological opening, temporal outlier rejection, rolling median, column-wise median)
- `PROJECT_PLAN.md`: Added Section 9.1 — Reaction Failure Detection covering three failure modes: no stable front, front stalled, front did not reach tube bottom

## [0.1.0] - 2026-06-17

### Added
- `PROJECT_PLAN.md`: Detailed project plan for automated frontal polymerization front speed extraction from video. Covers background research on open-source tools (OpenCV, scikit-image, TrackPy, PyImageJ), core algorithm design, implementation phases, validation strategy, and risk mitigation.
