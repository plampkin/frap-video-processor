import cv2
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# Directories
INPUT_DIR = 'input_videos'
OUTPUT_VIDEO_DIR = 'output_videos'
OUTPUT_IMAGE_DIR = 'output_images'
OUTPUT_DATA_DIR = 'output_data'
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

# Parameters
BLUR_KERNEL = 5
TEMPORAL_SMOOTH_FRAMES = 5
CALIBRATION_MM_PER_PX = None  # Set to e.g. 0.1 for 0.1 mm/px; None reports px/s
ROI = None  # (x, y, w, h) crop to tube region, or None for full frame
# Fronts are initiated at the top with a soldering iron and travel downward.
FRONT_DIRECTION = 'down'
# The front is only searched for within the middle 50% of the frame, both
# vertically and horizontally. Vertically, the top quarter (initial test-tube
# jostling + soldering-iron initiation) and the bottom quarter (end-of-tube
# plateau) are excluded. Horizontally, the left/right quarters (tube walls,
# meniscus glints, background) are excluded so the width-collapse averages only
# over the clear inside of the tube. The front is not moving at the start of the
# video and may enter this band at any time.
MONITOR_BAND_TOP_FRACTION = 0.25
MONITOR_BAND_BOTTOM_FRACTION = 0.75
MONITOR_BAND_LEFT_FRACTION = 0.25
MONITOR_BAND_RIGHT_FRACTION = 0.75
# Fraction of the band height treated as edge-pinning (front not yet entered, or
# fully reacted plateau) and excluded from the speed fit.
EDGE_MARGIN_FRACTION = 0.02
# Line-response kymograph smoothing kernel (odd) and the kernel size of the
# vertical second-derivative line detector used to find the thin horizontal
# refractive-index front line (polarity-agnostic via magnitude).
KYMO_SMOOTH = 5
LINE_KSIZE = 3
# Relative line-strength gate below which a time column is treated as
# "front not present" (NaN), so the front may enter the band at any time.
RIDGE_GATE = 0.30
# Max downward step of the tracked ridge between consecutive frames, as a
# fraction of the band height. Enforces a bounded, continuous, downward-only
# front trajectory during the dynamic-programming ridge track.
RIDGE_MAX_STEP_FRACTION = 0.06

results = []

for video_path in sorted(glob.glob(os.path.join(INPUT_DIR, '*.mov'))):
    print(f"Processing {video_path}")
    stem = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {frame_w}x{frame_h_orig} @ {fps:.1f} fps")

    frames_gray = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ROI is not None:
            x, y, w, h = ROI
            gray = gray[y:y + h, x:x + w]
        frames_gray.append(cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0))
    cap.release()

    if len(frames_gray) < 10:
        print(f"  Skipping: only {len(frames_gray)} frames readable")
        continue

    frames_gray = np.array(frames_gray)
    n_frames, roi_h, roi_w = frames_gray.shape

    # Restrict all analysis to the middle 50% band of the frame, both vertically
    # (drop initiation/jostling above, end-of-tube plateau below) and horizontally
    # (drop tube walls / glints / background, keep the clear inside of the tube).
    band_top = int(roi_h * MONITOR_BAND_TOP_FRACTION)
    band_bottom = int(roi_h * MONITOR_BAND_BOTTOM_FRACTION)
    band_left = int(roi_w * MONITOR_BAND_LEFT_FRACTION)
    band_right = int(roi_w * MONITOR_BAND_RIGHT_FRACTION)
    band_frames = frames_gray[:, band_top:band_bottom, band_left:band_right]
    band_h, band_w = band_frames.shape[1], band_frames.shape[2]

    # Fronts are top-initiated and travel downward; direction is fixed, not auto-detected.
    front_direction = FRONT_DIRECTION

    # --- Horizontal-line ridge tracking ---
    # The front is a refractive-index discontinuity with no significant brightness
    # or color step, so a mean-intensity-gradient tracker latches onto lighting /
    # walls / bubbles. The one invariant is that the front is a thin, near-horizontal
    # line spanning the full tube width (regardless of whether the reacted region
    # above is bubbly, textured, or clear). We detect that line directly.
    #
    # 1. Per-frame polarity-agnostic line response: the vertical second derivative
    #    |d^2 I / dy^2| peaks at a thin horizontal line whether it reads slightly
    #    dark or slightly bright, and is insensitive to a smooth region brightness
    #    step. 2. Width-coherence collapse: average the response across the tube
    #    width. A full-width front survives; a local bubble/glint at one x is
    #    diluted away. Stacking the per-frame width-collapsed profiles gives a
    #    line-response kymograph L[y, t] (rows = position in band, cols = time).
    L = np.empty((band_h, n_frames), dtype=np.float32)
    for t in range(n_frames):
        f = band_frames[t].astype(np.float32)
        d2y = cv2.Sobel(f, cv2.CV_32F, 0, 2, ksize=LINE_KSIZE)
        L[:, t] = np.abs(d2y).mean(axis=1)
    L = cv2.GaussianBlur(L, (KYMO_SMOOTH, KYMO_SMOOTH), 0)
    kymo = L  # diagnostic image is the line-response kymograph

    # 3. Track a single downward ridge with a continuity constraint, instead of an
    #    independent per-column argmax (which jumps between spurious responses).
    #    Forward Viterbi pass: the path may move down only and at most K rows per
    #    frame, maximising the summed line response. This yields one connected,
    #    monotonic, bounded-speed front trajectory.
    K = max(1, int(band_h * RIDGE_MAX_STEP_FRACTION))
    score = L[:, 0].astype(np.float64).copy()
    back = np.zeros((n_frames, band_h), dtype=np.int32)
    back[0] = np.arange(band_h)
    rows = np.arange(band_h)
    for t in range(1, n_frames):
        m = score.copy()                       # best reachable predecessor score
        src = rows.copy()                      # ...and the row it came from
        for k in range(1, K + 1):
            cand = score[:-k]                  # predecessor row y-k for outputs y>=k
            better = cand > m[k:]
            m[k:] = np.where(better, cand, m[k:])
            src[k:] = np.where(better, rows[:-k], src[k:])
        score = L[:, t].astype(np.float64) + m
        back[t] = src

    # Backtrack the optimal path from the strongest end state.
    path = np.empty(n_frames, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t][path[t]]

    # 4. The front may enter the band at any time: gate out columns whose line
    #    response along the path is weak (front not yet arrived / already gone).
    strength = L[path, np.arange(n_frames)]
    gate = RIDGE_GATE * float(np.nanmax(strength))
    front_band = path.astype(float)
    front_band[strength < gate] = np.nan

    # Convert band-relative rows to absolute frame-row coordinates.
    front_positions = band_top + front_band

    print(f"  Line-response kymograph {L.shape[0]}x{L.shape[1]}, direction: {front_direction}, "
          f"front detected in {int(np.sum(~np.isnan(front_positions)))}/{n_frames} frames")

    # Temporal outlier rejection: flag jumps > 2× median frame-to-frame displacement
    diffs = np.abs(np.diff(front_positions))
    median_diff = np.nanmedian(diffs)
    if median_diff > 0:
        outliers = np.concatenate([[False], diffs > 2 * median_diff])
        front_positions[outliers] = np.nan

    # Rolling median smoothing
    smooth = (pd.Series(front_positions)
              .rolling(TEMPORAL_SMOOTH_FRAMES, center=True, min_periods=1)
              .median()
              .values)

    times = np.arange(n_frames) / fps

    # Fit only while the front is in transit through the monitoring band. Before
    # the front enters there is no reacted region (NaN), and once the band is fully
    # reacted the leading edge pins to the bottom of the band (plateau). A small edge
    # margin drops both, leaving whenever-it-occurs steady-state propagation.
    margin = band_h * EDGE_MARGIN_FRACTION
    in_band = (~np.isnan(smooth)) & (smooth > band_top + margin) & (smooth < band_bottom - margin)

    fit_t = times[in_band]
    fit_p = smooth[in_band]
    valid = ~np.isnan(fit_p)

    slope = intercept = speed_px_s = r2 = np.nan
    status = 'FAILED_NO_STABLE_FRONT'

    if np.sum(valid) >= 5:
        res = stats.linregress(fit_t[valid], fit_p[valid])
        slope, intercept, r2 = res.slope, res.intercept, res.rvalue ** 2
        speed_px_s = abs(slope)

        if r2 < 0.80:
            status = 'FAILED_NO_STABLE_FRONT'
        else:
            vt, vp = fit_t[valid], fit_p[valid]
            mid = len(vt) // 2
            if mid >= 2:
                s1 = stats.linregress(vt[:mid], vp[:mid]).slope
                s2 = stats.linregress(vt[mid:], vp[mid:]).slope
                if abs(s1) > 0 and abs(s2) / abs(s1) < 0.2:
                    status = 'FAILED_FRONT_STALLED'
                else:
                    front_range = np.nanmax(smooth) - np.nanmin(smooth)
                    status = 'FAILED_FRONT_DID_NOT_COMPLETE' if front_range / band_h < 0.8 else 'OK'
            else:
                status = 'OK'

    if CALIBRATION_MM_PER_PX is not None:
        speed = speed_px_s * CALIBRATION_MM_PER_PX
        speed_unit = 'mm/s'
    else:
        speed = speed_px_s
        speed_unit = 'px/s'

    print(f"  Status: {status}, Speed: {speed:.3f} {speed_unit}, R²: {r2:.4f}")

    # Per-video position-time CSV
    pd.DataFrame({
        'time_s': times,
        'front_y_px_raw': front_positions,
        'front_y_px_smooth': smooth,
    }).to_csv(os.path.join(OUTPUT_DATA_DIR, f'{stem}_position_time.csv'), index=False)

    # Position-vs-time plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fit_t[valid], fit_p[valid], s=2, alpha=0.5, label='Front position (smoothed)')
    if not np.isnan(slope):
        ax.plot(fit_t[valid], slope * fit_t[valid] + intercept, 'r-',
                label=f'Fit: {speed:.2f} {speed_unit}, R²={r2:.3f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Front position (px)')
    ax.set_title(f'{stem}\n{speed:.2f} {speed_unit}  [{status}]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_IMAGE_DIR, f'{stem}_position_time.png'), dpi=150)
    plt.close(fig)

    # Diagnostic: line-response kymograph (x=time, y=absolute frame row) with the
    # monitoring band, the tracked ridge, and the fitted speed line overlaid.
    figk, axk = plt.subplots(figsize=(8, 5))
    axk.imshow(kymo, cmap='magma', aspect='auto',
               extent=[times[0], times[-1], band_bottom, band_top])
    axk.plot(times, smooth, '.', color='cyan', ms=2, label='Tracked front')
    if not np.isnan(slope):
        axk.plot(fit_t[valid], slope * fit_t[valid] + intercept, 'r-',
                 label=f'Fit: {speed:.2f} {speed_unit}, R²={r2:.3f}')
    axk.axhline(band_top, color='yellow', lw=1)
    axk.axhline(band_bottom, color='yellow', lw=1)
    axk.set_xlabel('Time (s)')
    axk.set_ylabel('Front position (px)')
    axk.set_title(f'{stem} line-response kymograph\n{speed:.2f} {speed_unit}  [{status}]')
    axk.legend()
    figk.tight_layout()
    figk.savefig(os.path.join(OUTPUT_IMAGE_DIR, f'{stem}_kymograph.png'), dpi=150)
    plt.close(figk)

    # Annotated video
    cap2 = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(OUTPUT_VIDEO_DIR, f'{stem}_annotated.mp4'), fourcc, fps, (frame_w, frame_h_orig))
    roi_x_offset = ROI[0] if ROI is not None else 0
    roi_y_offset = ROI[1] if ROI is not None else 0
    roi_width = ROI[2] if ROI is not None else frame_w

    band_top_abs = band_top + roi_y_offset
    band_bottom_abs = band_bottom + roi_y_offset
    band_left_abs = band_left + roi_x_offset
    band_right_abs = band_right + roi_x_offset

    for i, front_y in enumerate(smooth):
        ret, frame = cap2.read()
        if not ret:
            break
        # Monitoring band: middle 50% vertically and horizontally
        cv2.line(frame, (band_left_abs, band_top_abs), (band_right_abs, band_top_abs), (255, 255, 0), 1)
        cv2.line(frame, (band_left_abs, band_bottom_abs), (band_right_abs, band_bottom_abs), (255, 255, 0), 1)
        cv2.line(frame, (band_left_abs, band_top_abs), (band_left_abs, band_bottom_abs), (255, 255, 0), 1)
        cv2.line(frame, (band_right_abs, band_top_abs), (band_right_abs, band_bottom_abs), (255, 255, 0), 1)
        if not np.isnan(front_y):
            fy_abs = int(front_y) + roi_y_offset
            cv2.line(frame, (band_left_abs, fy_abs), (band_right_abs, fy_abs), (0, 0, 255), 2)
        label = f'{speed:.2f} {speed_unit}  [{status}]'
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out.write(frame)

    cap2.release()
    out.release()

    # Collect results row with parsed filename metadata
    parts = stem.split('_')
    field_names = ['monomer', 'initiator', 'amount', 'replicate', 'sample']
    meta = {field_names[i]: parts[i] if i < len(parts) else '' for i in range(len(field_names))}
    results.append({
        'video': video_path,
        'status': status,
        f'speed_{speed_unit.replace("/", "_per_")}': round(speed, 4) if not np.isnan(speed) else np.nan,
        'r2': round(r2, 4) if not np.isnan(r2) else np.nan,
        'direction': front_direction,
        **meta,
    })

results_path = os.path.join(OUTPUT_DATA_DIR, 'front_speed_results.csv')
pd.DataFrame(results).to_csv(results_path, index=False)
print(f'\nResults saved to {results_path}')
print(pd.DataFrame(results).to_string())
