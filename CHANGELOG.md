# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2026-06-17

### Changed
- `PROJECT_PLAN.md`: Updated front speed units to mm/s (was cm/min or mm/s); removed "luminescent" descriptor for the front boundary (now "visible boundary")
- `PROJECT_PLAN.md`: Clarified calibration fallback — program reports px/s when no calibration is provided, mm/s when `CALIBRATION_MM_PER_PX` is set
- `PROJECT_PLAN.md`: Added noise/bubble mitigation plan in Phase 2 (width filter, morphological opening, temporal outlier rejection, rolling median, column-wise median)
- `PROJECT_PLAN.md`: Added Section 9.1 — Reaction Failure Detection covering three failure modes: no stable front, front stalled, front did not reach tube bottom

## [0.1.0] - 2026-06-17

### Added
- `PROJECT_PLAN.md`: Detailed project plan for automated frontal polymerization front speed extraction from video. Covers background research on open-source tools (OpenCV, scikit-image, TrackPy, PyImageJ), core algorithm design, implementation phases, validation strategy, and risk mitigation.
