# Changelog

All notable changes to this project will be documented in this file.

## [0.11.0] - 2026-06-18

### Changed
- `process_videos.py`: **the front is now fit as the leading unreacted→reacted edge,
  not a thin line.** The kymograph response is built from the signed vertical **first
  derivative** (positive part of `dI/dy`, `EDGE_KSIZE`) instead of the polarity-agnostic
  second derivative `|d²I/dy²|`. Reacted is dark/above and unreacted is light/below, so a
  top→bottom column scan steps *up* in brightness at the front (`dI/dy > 0`); keeping only
  the positive part fires on that transition while the uniform dark interior behind the
  front and the horizontal bubble banding contribute ~zero. This fixes the bubbly-sample
  failure (`V70_800_1`), where the dark interior and bubble bands previously outvoted the
  real edge, and the shallow bias on faint fronts. Renamed `LINE_KSIZE` → `EDGE_KSIZE`.
- `process_videos.py`: **re-added a start-time cut** (`START_FRACTION = 1/3`). The first
  ~⅓ of the time axis (test-tube jostling + soldering-iron initiation) is excluded from
  the edge-threshold estimate, the Radon line search, the centroid trace, and the fit;
  the steady diagonal persists well past the cut, so the steady-state speed is unaffected.
  The ignored region is shaded on the position-time and kymograph diagnostics.
- `process_videos.py`: **the edge-presence threshold is computed on the retained region
  only** (`EDGE_PCTL` percentile over post-cut columns), so dense early initiation blobs
  cannot bias the percentile.
- `process_videos.py`: **accept/reject is now on geometry, not R².** R² is ~1 by
  construction here (points are selected for lying on a line), so the `R² < 0.80` gate is
  replaced by two geometric gates: **time coverage** (inliers span ≥ `MIN_COVERAGE_FRAC`
  of the post-cut time axis) and **fill / continuity** (≥ `MIN_FILL_FRAC` of the spanned
  columns carry a front pixel above the edge threshold). The downward-slope requirement
  (`slope > 0`) is kept; `FAILED_NO_STABLE_FRONT` if either gate fails. Both metrics are
  reported in the console, the plots, and `front_speed_results.csv` (the `r2` column is
  replaced by `coverage` and `fill`). Added `START_FRACTION`, `EDGE_PCTL`,
  `MIN_COVERAGE_FRAC`, `MIN_FILL_FRAC`; `MIN_SUPPORT_FRAC` is now relative to the
  retained frame count.

## [0.10.0] - 2026-06-18

### Changed
- `process_videos.py`: **replaced the Hough-on-partial-kymograph fit with a wide-range
  Radon line-integral search** so the fit is robust to wildly-varying front speeds,
  heavy bubble banding, and faint fronts. The diagonal is found by summing the
  kymograph response along every candidate **downward** line `y = m·t + b` over a wide
  slope range (`N_SLOPES = 240`, bounded by `SLOPE_MIN_TRAVEL_FRAC` /
  `MIN_TRANSIT_FRAC`): integrating the whole line lifts a faint front out of the noise,
  and near-horizontal lines are excluded so bubble bands / residual static features can
  never win. The winning line seeds a **per-column intensity-weighted centroid** trace
  (one sub-pixel front position per frame, `CENTROID_HALF_FRAC`, gated by
  `CENTROID_GATE`), which is then fit robustly (**Theil-Sen → OLS** on inliers). This
  fixes the bubbly-sample failures (`AIBN_300_1`, `V70_800_1`) and the shallow-slope
  bias on faint fronts (`V65_800_1`, `V70_1500_1`).
- `process_videos.py`: **dropped the fixed first-1/3 time cut** — the front may now
  enter the band at any time; weak columns simply stay `NaN`. Removed
  `FIT_START_FRACTION`, `RIDGE_PCTL`, `HOUGH_MIN_LINE_FRAC`, `HOUGH_MAX_GAP_FRAC`;
  added `N_SLOPES`, `SLOPE_MIN_TRAVEL_FRAC`, `MIN_TRANSIT_FRAC`, `MIN_SUPPORT_FRAC`,
  `CENTROID_HALF_FRAC`, `CENTROID_GATE`.
- `process_videos.py`: kymographs stay grayscale (`gray_r`, intense = black); the
  tracked front centroids are overlaid alongside the fitted diagonal.

## [0.9.0] - 2026-06-18

### Changed
- `process_videos.py`: **the diagonal fit no longer uses the whole plot.** The early
  part of the kymograph behaves differently from the steady diagonal (the test-tube
  jostling + soldering-iron initiation form a bright blob before the front enters the
  band), which skewed the whole-plot fit. The fit now (1) **ignores the first third**
  of the time axis (`FIT_START_FRACTION = 1/3`), (2) keeps only the **most-intense**
  ridge pixels (`RIDGE_PCTL = 92`), and (3) **returns to a Hough transform** on that
  partial kymograph: the longest downward Hough segment seeds the diagonal, then an
  OLS refit on its supporting ridge pixels gives the slope (front speed). Replaced
  the Theil-Sen per-column trace; added `FIT_START_FRACTION`, `RIDGE_PCTL`,
  `HOUGH_MIN_LINE_FRAC`, `HOUGH_MAX_GAP_FRAC`; removed `RIDGE_GATE`.
- `process_videos.py`: **kymograph images are now grayscale with the most-intense
  signal rendered black** (`cmap='gray_r'`) for readability, and the ignored first
  third is shaded on both the kymograph and the position-time plot.

## [0.8.0] - 2026-06-18

### Changed
- `process_videos.py`: **reinstated the hard-coded middle-50% analysis band** on
  both axes (`BAND_TOP/BOTTOM/LEFT/RIGHT_FRACTION = 0.25/0.75`). The kymograph is
  built only over this band, dropping the top jostling / initiation, the
  bottom-of-tube plateau, and the tube walls / background glints.
- `process_videos.py`: **replaced the Hough + OLS-on-inlier-pixels diagonal fit**
  (which let the fitted front fall behind / get lost when part of the diagonal
  faded or spurious bright clusters appeared) with a **robust per-column trace +
  Theil-Sen fit**. Each time-column contributes the row of strongest line
  response (`argmax`), columns whose peak is below `RIDGE_GATE * max` are gated
  out (front enters at any time), and the diagonal is fit with `scipy.stats.theilslopes`;
  the fit is then refined with OLS on the robust inliers and required to be
  downward (`slope > 0`). Removed `RIDGE_PCTL` / `MIN_LINE_FRAC`; added `RIDGE_GATE`.
- `process_videos.py`: diagnostics now show the band kymograph, the band shaded on
  the position-time plot, and the middle-50% rectangle on the annotated video.

## [0.7.0] - 2026-06-18

### Changed
- `process_videos.py`: front speed is now measured by **detecting and fitting the
  diagonal line on the kymograph** directly, replacing the per-frame
  Viterbi ridge tracker. Every run's kymograph shows one clear diagonal (the
  travelling front) regardless of bubbling or how faint the line is, so the front
  is no longer tracked frame-by-frame: the dominant diagonal is found with a
  probabilistic Hough transform (`cv2.HoughLinesP`) and least-squares fit to its
  supporting ridge pixels; the slope is the front speed.
- `process_videos.py`: the kymograph `L[y, t]` is built over the **whole frame**
  (polarity-agnostic vertical second derivative `|d^2 I/dy^2|`, `LINE_KSIZE`,
  averaged across the full width). Each row's temporal median is subtracted to
  cancel static horizontal features (tube bottom, meniscus, fixed markings),
  leaving the moving front as the dominant diagonal. Only downward (`slope > 0`)
  diagonals are accepted.
- `process_videos.py`: **removed all analysis-region machinery** — the
  hard-coded `MONITOR_BAND_*` middle-50% band, `ROI`, `EDGE_MARGIN_FRACTION`, the
  Viterbi `RIDGE_GATE` / `RIDGE_MAX_STEP_FRACTION`, and the temporal-smoothing /
  outlier-rejection pipeline. The whole video is analyzed. New tuning knobs:
  `RIDGE_PCTL`, `MIN_LINE_FRAC`, `INLIER_TOL_FRAC`.
- `process_videos.py`: diagnostics updated — the kymograph image overlays the
  fitted diagonal, the position-time plot shows the raw per-column ridge plus the
  fit, and the annotated video draws the fitted front line (no band rectangle).

## [0.6.0] - 2026-06-18

### Changed
- `process_videos.py`: front detection rebuilt as a **horizontal-line ridge
  tracker** for the refractive-index front, which has no significant brightness
  or color step. Each band frame is run through a polarity-agnostic vertical
  second-derivative line detector (`|d^2 I / dy^2|`, `LINE_KSIZE`) that responds
  to a thin horizontal line whether it reads slightly dark or bright; the
  response is averaged across the tube width (width-coherence collapse) so a
  full-width front survives while local bubbles/glints are diluted. The
  per-frame profiles are stacked into a **line-response kymograph** `L[y, t]`,
  replacing the previous intensity-gradient (`|dI/dy|`) kymograph that latched
  onto lighting/walls/bubbles.
- `process_videos.py`: the front is now tracked with a **continuity-constrained
  dynamic-programming (Viterbi) ridge** — a single connected, downward-only,
  bounded-speed trajectory maximising the summed line response — instead of an
  independent per-column `argmax`. The ridge may enter the band at any time:
  columns whose line strength falls below `RIDGE_GATE` stay `NaN`.
- `process_videos.py`: the monitoring band is the **middle 50% both vertically
  and horizontally** (`MONITOR_BAND_TOP/BOTTOM_FRACTION` = 0.25/0.75,
  `MONITOR_BAND_LEFT/RIGHT_FRACTION` = 0.25/0.75). The horizontal crop excludes
  tube walls, meniscus glints, and background so the width-collapse averages
  only over the clear inside of the tube.
- `process_videos.py`: the diagnostic kymograph now shows the line-response
  image (`magma`), and the annotated video draws the full middle-50% rectangle
  (both vertical and horizontal boundaries) plus the tracked front line.

### Added
- `process_videos.py`: new parameters `MONITOR_BAND_LEFT_FRACTION`,
  `MONITOR_BAND_RIGHT_FRACTION`, `LINE_KSIZE`, `RIDGE_GATE`, and
  `RIDGE_MAX_STEP_FRACTION` (max downward ridge step per frame).

### Removed
- `process_videos.py`: `KYMO_EDGE_GATE` and the intensity-gradient front tracker.

## [0.5.0] - 2026-06-17

### Changed
- `process_videos.py`: front detection now uses a **kymographic method**. Each
  frame is collapsed to a 1-D vertical intensity profile (median across the tube
  width) inside the monitoring band, and the per-frame profiles are stacked into
  a space-time kymograph (rows = position, columns = time). The reaction front is
  tracked as the strongest vertical-intensity-gradient (`|dI/dy|`) edge in each
  time column, replacing the previous per-frame Otsu thresholding + morphology +
  reacted-class detection. This is robust to the absolute brightness of either
  phase and removes the binary-class failure mode that produced
  `FAILED_NO_STABLE_FRONT` on every video.
- `process_videos.py`: the monitoring band is narrowed from the middle 50% to the
  **middle 33%** of the frame height (`MONITOR_BAND_TOP_FRACTION` = 1/3,
  `MONITOR_BAND_BOTTOM_FRACTION` = 2/3), excluding the top-third initiation/jostling
  and the bottom-third end-of-tube plateau more aggressively.
- `process_videos.py`: columns whose edge strength falls below `KYMO_EDGE_GATE`
  (relative to the peak) are left undetected (`NaN`), so the front is picked up
  whenever it enters the band rather than being assumed present from frame 0.

### Added
- `process_videos.py`: new parameters `KYMO_SMOOTH` (kymograph smoothing kernel)
  and `KYMO_EDGE_GATE` (relative edge-strength gate).
- `process_videos.py`: a per-video kymograph diagnostic image
  `output_images/<stem>_kymograph.png` showing the space-time kymograph with the
  monitoring band, the tracked front, and the fitted speed line overlaid.

### Removed
- `process_videos.py`: per-frame Otsu threshold sampling, morphological
  open/close, reacted-class (bright/dark) detection, and the
  `MIN_FRONT_WIDTH_FRACTION` width filter — superseded by the kymograph gradient
  tracker.

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
