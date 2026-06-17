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
# The front is only searched for within the middle 33% of the frame height:
# the top third (initial test-tube jostling + soldering-iron initiation) and the
# bottom third (end-of-tube plateau) are excluded. The front is not moving at the
# start of the video and may enter this band at any time.
MONITOR_BAND_TOP_FRACTION = 1.0 / 3.0
MONITOR_BAND_BOTTOM_FRACTION = 2.0 / 3.0
# Fraction of the band height treated as edge-pinning (front not yet entered, or
# fully reacted plateau) and excluded from the speed fit.
EDGE_MARGIN_FRACTION = 0.02
# Kymograph smoothing kernel (odd) and the relative edge-strength gate below which
# a time column is treated as "front not present" (NaN).
KYMO_SMOOTH = 5
KYMO_EDGE_GATE = 0.30

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

    # Restrict all analysis to the middle 33% band of the frame height.
    band_top = int(roi_h * MONITOR_BAND_TOP_FRACTION)
    band_bottom = int(roi_h * MONITOR_BAND_BOTTOM_FRACTION)
    band_frames = frames_gray[:, band_top:band_bottom, :]
    band_h = band_frames.shape[1]

    # Fronts are top-initiated and travel downward; direction is fixed, not auto-detected.
    front_direction = FRONT_DIRECTION

    # --- Kymographic front tracking ---
    # Collapse each frame to a 1-D vertical intensity profile (median across the
    # tube width) inside the monitoring band, then stack the per-frame profiles
    # column-by-column into a space-time kymograph: rows = vertical position
    # within the band, columns = frame index (time). The reaction front is the
    # moving boundary between reacted and unreacted material, which appears as a
    # single tilted edge sweeping across the kymograph; its slope is the speed.
    kymo = np.median(band_frames, axis=2).T.astype(np.float32)  # (band_h, n_frames)
    kymo = cv2.GaussianBlur(kymo, (KYMO_SMOOTH, KYMO_SMOOTH), 0)

    # Vertical intensity gradient: |dI/dy| peaks at the reacted/unreacted edge in
    # each time column. Tracking the gradient edge is robust to the absolute
    # brightness of either phase (no per-frame thresholding / morphology needed).
    grad_mag = np.abs(cv2.Sobel(kymo, cv2.CV_32F, 0, 1, ksize=3))

    # Per-column (per-frame) front = row of strongest edge, gated so columns the
    # front has not yet entered (weak edge) stay undetected (NaN). The front can
    # therefore appear at any time rather than being assumed present from frame 0.
    col_max = grad_mag.max(axis=0)
    gate = KYMO_EDGE_GATE * float(np.nanmax(col_max))
    front_band = np.argmax(grad_mag, axis=0).astype(float)
    front_band[col_max < gate] = np.nan

    # Convert band-relative rows to absolute frame-row coordinates.
    front_positions = band_top + front_band

    print(f"  Kymograph {kymo.shape[0]}x{kymo.shape[1]}, direction: {front_direction}, "
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

    # Kymograph diagnostic: space-time image (x=time, y=absolute frame row) with
    # the monitoring band, the tracked front, and the fitted speed line overlaid.
    figk, axk = plt.subplots(figsize=(8, 5))
    axk.imshow(kymo, cmap='gray', aspect='auto',
               extent=[times[0], times[-1], band_bottom, band_top])
    axk.plot(times, smooth, '.', color='cyan', ms=2, label='Tracked front')
    if not np.isnan(slope):
        axk.plot(fit_t[valid], slope * fit_t[valid] + intercept, 'r-',
                 label=f'Fit: {speed:.2f} {speed_unit}, R²={r2:.3f}')
    axk.axhline(band_top, color='yellow', lw=1)
    axk.axhline(band_bottom, color='yellow', lw=1)
    axk.set_xlabel('Time (s)')
    axk.set_ylabel('Front position (px)')
    axk.set_title(f'{stem} kymograph\n{speed:.2f} {speed_unit}  [{status}]')
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

    for i, front_y in enumerate(smooth):
        ret, frame = cap2.read()
        if not ret:
            break
        # Monitoring band boundaries (middle 50% of the frame)
        cv2.line(frame, (roi_x_offset, band_top_abs), (roi_x_offset + roi_width, band_top_abs), (255, 255, 0), 1)
        cv2.line(frame, (roi_x_offset, band_bottom_abs), (roi_x_offset + roi_width, band_bottom_abs), (255, 255, 0), 1)
        if not np.isnan(front_y):
            fy_abs = int(front_y) + roi_y_offset
            cv2.line(frame, (roi_x_offset, fy_abs), (roi_x_offset + roi_width, fy_abs), (0, 0, 255), 2)
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
