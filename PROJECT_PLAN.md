# Frontal Polymerization Video Processor — Project Plan

## 1. Problem Statement

Extract the **front speed** (cm/min or mm/s) from videos of frontal polymerization (FP) experiments in a deterministic, automated manner using Python. Each video shows a reaction front propagating through a monomer-filled test tube. The front appears as a visible luminescent/color boundary moving along the tube axis.

**Input:** `.mov` video files named `<monomer>_<initiator>_<amount>_<replicate>_<sample>.mov`
**Output:** front speed (with units) per video, plus a position-vs-time plot

---

## 2. Background

### 2.1 Frontal Polymerization

Frontal polymerization converts monomer to polymer via a self-sustaining propagating reaction zone. A triggering event (typically local heating) initiates exothermic polymerization at one end of the sample; the heat released sustains and propagates the front through the unreacted material. Front speed is the primary observable characterizing the process and depends on:

- Monomer type and concentration
- Initiator identity and loading (ppm)
- Sample tube geometry and ambient temperature

Front speeds in the literature range from ~0.5–20 cm/min for acrylate-based systems using azo initiators such as AIBN, V65 (2,2'-azobis(2,4-dimethyl valeronitrile)), and V70 (2,2'-azobis(4-methoxy-2,4-dimethyl valeronitrile)).

### 2.2 How Front Speed is Conventionally Measured

Researchers typically:
1. Record the experiment on video
2. Open the video in ImageJ/Fiji
3. Manually mark the front position at several time points
4. Fit a line to position vs. time data to extract speed

This process is slow, subjective, and difficult to reproduce across labs.

### 2.3 Relevant Open-Source Tools and Approaches

#### OpenCV (`opencv-python`)
The dominant library for video processing in Python. Relevant capabilities:
- `cv2.VideoCapture`: read `.mov` / `.mp4` frames with timestamps
- `cv2.cvtColor`: convert BGR → grayscale or HSV
- `cv2.GaussianBlur`: noise suppression before thresholding
- `cv2.threshold` / `cv2.adaptiveThreshold`: binary segmentation of reaction front
- `cv2.Canny`: edge detection at front boundary
- `cv2.findContours`: locate the front boundary as a contour
- `cv2.rectangle` / `cv2.line`: annotate output video

**Similar use cases:** oxidation front tracking in steel heating experiments, flame front propagation in combustion videos, solidification front in crystal growth videos. All use the same paradigm: segment → find boundary → measure position vs. time.

#### scikit-image (`scikit-image`)
Pythonic image analysis library, best for per-frame analysis:
- `skimage.filters.threshold_otsu`: automatic threshold selection without manual tuning
- `skimage.morphology.binary_closing/opening`: remove noise from binary mask
- `skimage.measure.regionprops`: centroid, bounding box of segmented front zone
- `skimage.feature.canny`: edge detection
- `skimage.segmentation.chan_vese`: active contour segmentation for complex fronts

#### TrackPy (`trackpy`)
Particle-tracking library (Crocker-Grier algorithm). Designed for discrete particles, not continuous fronts — **not recommended as the primary tool** for this application, though it could be used to track manually placed "seed points" along the front.

#### PyImageJ
Python wrapper for ImageJ/Fiji. Enables reuse of existing ImageJ macros that researchers may already have, but adds complexity and a Java runtime dependency. **Useful as a validation baseline** if researchers have existing ImageJ measurements to compare against.

#### Optical Flow (OpenCV)
`cv2.calcOpticalFlowFarneback` computes dense motion vectors between frames. Could be used to measure front velocity directly from pixel displacement, but is sensitive to noise and gives a field of velocities rather than a single front speed number. Best used as a cross-check.

#### ImageJ / FIJI (reference baseline)
While not a Python library, ImageJ is the most common tool researchers use manually. The "Manual Tracking" and "MTrack2" plugins are widely cited in FP literature. Understanding their workflow informed the algorithm below.

---

## 3. Core Algorithm

The proposed approach is a classical computer vision pipeline — deterministic, interpretable, and reproducible with no ML model required.

### 3.1 Pipeline Overview

```
Video frames
     │
     ▼
[1] Preprocessing
     │  Grayscale conversion
     │  Gaussian blur (denoise)
     │
     ▼
[2] ROI selection
     │  Crop to test tube region (manual or auto-detected)
     │
     ▼
[3] Binary thresholding (Otsu or fixed)
     │  Segment "reacted zone" from "unreacted zone"
     │  Morphological closing to fill holes
     │
     ▼
[4] Front position extraction
     │  For each frame: find the leading edge y-coordinate
     │  (bottommost row with reaction-zone pixels, or
     │   topmost, depending on front direction)
     │
     ▼
[5] Position-time data
     │  y[frame] → y[time_seconds]
     │  Convert pixel position to physical units (mm/cm)
     │  using known calibration (tube length / pixel count)
     │
     ▼
[6] Speed calculation
     │  Linear regression on y vs t (robust to noise)
     │  Speed = slope × unit_conversion_factor
     │
     ▼
[7] Output
       front_speed (cm/min), R², position-time plot, annotated video
```

### 3.2 Front Position Extraction Detail

For each preprocessed binary frame:
```
front_y = max row index where any pixel in ROI columns is "reacted"
```

Robustness improvements:
- Use the **median** column position across all non-zero rows rather than min/max, reducing sensitivity to noise at edges
- Apply a column-wise scan: for each column in the ROI, find the y at which the binary mask transitions; take the median across columns
- Optionally apply temporal smoothing (rolling median over N frames)

### 3.3 Calibration

Physical-unit conversion requires knowing the pixel-to-length ratio. Two options:
1. **Known tube length**: measure the tube in pixels from the video; divide by known physical length (e.g., 15 cm)
2. **Reference marker**: place a ruler or marker of known length in the frame

Without calibration, the code can report speed in **pixels/second**; calibration converts this to cm/min.

---

## 4. Implementation Plan

### Phase 1 — Environment and Video Inspection (1–2 hours)

**Tasks:**
- Set up Python environment with `uv`:
  ```bash
  uv pip install opencv-python scikit-image numpy scipy matplotlib pandas
  ```
- Write a short script to print metadata for each video:
  - Frame dimensions, FPS, total frame count, duration
- Visually inspect frames to understand:
  - Front direction (top-to-bottom or bottom-to-top)
  - Front appearance (bright band, dark band, color gradient)
  - Background clutter (labels, test tube holder, scale bar)
  - Whether ROI can be automatically detected or must be specified

**Deliverable:** confirmed video properties and sample frames

### Phase 2 — Preprocessing and Thresholding (2–4 hours)

**Tasks:**
- Implement grayscale conversion + Gaussian blur
- Test Otsu's thresholding (`cv2.threshold` with `THRESH_OTSU`) on sample frames
- If Otsu fails (e.g., front has similar intensity to background), try:
  - HSV-based segmentation (isolate luminescent hue)
  - Frame differencing (subtract an early "unreacted" reference frame)
  - Adaptive thresholding
- Apply morphological closing to clean binary mask
- Visualize binary mask overlay on original frame to verify quality

**Deliverable:** threshold function that reliably segments the front in all 4 videos

### Phase 3 — Front Position Extraction (2–3 hours)

**Tasks:**
- Implement column-wise front position scan
- Process all frames, record `(frame_index, front_y_pixels)`
- Convert `frame_index` to `time_seconds` using FPS
- Plot raw `y vs t` to confirm linear (constant-speed) behavior
- Identify and exclude non-linear segments (e.g., initiation transient, near end of tube)

**Deliverable:** `position_time.csv` with columns `[time_s, front_y_px]`

### Phase 4 — Speed Calculation (1 hour)

**Tasks:**
- Fit linear regression to the linear portion of `y vs t` (`scipy.stats.linregress`)
- Apply pixel-to-physical calibration (user-supplied or auto-detected)
- Report: slope (px/s), speed (cm/min), R², 95% confidence interval
- Produce publication-quality position-time plot with regression line

**Deliverable:** speed output per video, `front_speed_results.csv`

### Phase 5 — Annotated Output Video (1–2 hours)

**Tasks:**
- Overlay front position line on each frame
- Add speed annotation as text overlay
- Write annotated video with `cv2.VideoWriter`

**Deliverable:** `<video_name>_annotated.mp4` per input video

### Phase 6 — Multi-Video Batch Processing (1 hour)

**Tasks:**
- Wrap the single-video pipeline in a loop over all `.mov` files in the directory
- Parse filename metadata: monomer, initiator, amount, replicate from filename
- Append results to `front_speed_results.csv` with metadata columns

**Deliverable:** single `front_speed_results.csv` with one row per video

---

## 5. Code Architecture

Minimal flat-script design (per CLAUDE.md guidelines), with a single entry-point script:

```
frap-video-processor/
├── process_videos.py          # main script
├── PROJECT_PLAN.md            # this file
├── CHANGELOG.md               # version log
├── front_speed_results.csv    # output (generated)
├── *.mov                      # input videos
└── *_annotated.mp4            # annotated outputs (generated)
```

`process_videos.py` will be organized as sequential top-level code (no classes, no `if __name__ == "__main__"` guard per project conventions), operating on all `.mov` files in the working directory.

---

## 6. Key Parameters and Tuning Knobs

| Parameter | Default | Notes |
|---|---|---|
| `BLUR_KERNEL` | 5 | Gaussian blur kernel size (px) |
| `THRESHOLD_METHOD` | `'otsu'` | `'otsu'` or fixed integer (0–255) |
| `MIN_FRONT_WIDTH_FRACTION` | 0.3 | Min fraction of ROI width that must be active to count a row |
| `TEMPORAL_SMOOTH_FRAMES` | 5 | Rolling median window for position smoothing |
| `CALIBRATION_MM_PER_PX` | None | Set manually or auto-detected |
| `ROI` | `None` (full frame) | `(x, y, w, h)` in pixels |
| `FIT_START_FRACTION` | 0.1 | Fraction of video to skip at start (initiation) |
| `FIT_END_FRACTION` | 0.9 | Fraction of video to use for linear fit |

---

## 7. Validation Strategy

1. **Visual inspection:** confirm annotated video shows the overlay line tracking the visible front correctly
2. **R² check:** linear fit R² > 0.99 expected for well-behaved FP fronts
3. **Manual cross-check:** manually mark front position at 3–5 time points in ImageJ; compare to automated output
4. **Consistency check:** replicate videos from the same condition should give speeds within ~5%
5. **Literature comparison:** published front speeds for IBA/AIBN systems are ~1–10 cm/min; verify outputs are in range

---

## 8. Dependencies

```
opencv-python >= 4.8
scikit-image >= 0.21
numpy >= 1.25
scipy >= 1.11
matplotlib >= 3.7
pandas >= 2.0
```

Install with:
```bash
uv pip install opencv-python scikit-image numpy scipy matplotlib pandas
```

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Front not visible (low contrast) | Try HSV segmentation or frame differencing |
| Non-linear front speed (e.g., at initiation or near tube end) | Trim first/last 10% of frames before fitting |
| Different lighting across videos | Use Otsu's adaptive threshold per-video |
| Front direction varies by video | Auto-detect direction from first vs last frame |
| LFS-stored videos not fully available | Ensure `git lfs pull` is run before processing |

---

## 10. References

- Chechilo, N. M.; Enikolopyan, N. S. *Dokl. Akad. Nauk SSSR* **1974**, 214, 1131. (Original FP report)
- Pojman, J. A. *J. Am. Chem. Soc.* **1991**, 113, 6284. (Pojman's foundational FP work)
- Robertson, I. D. et al. *Nature* **2018**, 557, 223. (Frontal polymerization for manufacturing)
- Rouse, Z. et al. *ACS Macro Lett.* **2023**. (Front speed measurement methods)
- OpenCV documentation: https://docs.opencv.org/
- scikit-image documentation: https://scikit-image.org/docs/stable/
