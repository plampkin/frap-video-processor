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
BLUR_KERNEL = 5                 # spatial blur applied to each frame
CALIBRATION_MM_PER_PX = None    # None -> report px/s; set (e.g. 0.1) to report mm/s
# Only the middle 50% of the frame is analyzed (hard-coded), horizontally and
# vertically. This drops the top jostling / soldering-iron initiation, the
# bottom-of-tube plateau, and the tube walls / background glints, leaving a clean
# band the front sweeps through.
BAND_TOP_FRACTION = 0.25
BAND_BOTTOM_FRACTION = 0.75
BAND_LEFT_FRACTION = 0.25
BAND_RIGHT_FRACTION = 0.75
# The front is a thin, near-horizontal refractive-index line that sweeps downward.
# On the space-time kymograph it shows up as one clear diagonal line whose slope is
# the front speed (rows per frame). The early part of the plot behaves differently
# (test-tube jostling + soldering-iron initiation produce a bright blob before the
# front enters the band), which skews any whole-plot fit. So we (1) drop the first
# third of the time axis, (2) keep only the most-intense ridge pixels, and (3) detect
# the dominant downward diagonal with a Hough transform on that partial kymograph,
# then least-squares refine on its inliers.
LINE_KSIZE = 3                  # vertical 2nd-derivative kernel (front-line detector)
KYMO_SMOOTH = 5                 # Gaussian smoothing of the kymograph (odd)
FIT_START_FRACTION = 1/3        # ignore the first third of the plot (jostling/initiation)
RIDGE_PCTL = 92                 # keep only ridge pixels >= this percentile (most intense)
HOUGH_MIN_LINE_FRAC = 0.25      # min diagonal length, as a fraction of the partial time span
HOUGH_MAX_GAP_FRAC = 0.10       # max gap to bridge along the diagonal (fraction of time span)
INLIER_TOL_FRAC = 0.05          # fit inlier tolerance, as a fraction of band height

results = []

for video_path in sorted(glob.glob(os.path.join(INPUT_DIR, '*.mov'))):
    print(f"Processing {video_path}")
    stem = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {frame_w}x{frame_h} @ {fps:.1f} fps")

    frames_gray = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray.append(cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0))
    cap.release()

    if len(frames_gray) < 10:
        print(f"  Skipping: only {len(frames_gray)} frames readable")
        continue

    frames_gray = np.array(frames_gray)
    n_frames, H, W = frames_gray.shape

    # --- Crop to the middle 50% of the frame (both axes) ---
    band_top = int(H * BAND_TOP_FRACTION)
    band_bottom = int(H * BAND_BOTTOM_FRACTION)
    band_left = int(W * BAND_LEFT_FRACTION)
    band_right = int(W * BAND_RIGHT_FRACTION)
    band_frames = frames_gray[:, band_top:band_bottom, band_left:band_right]
    band_h = band_frames.shape[1]

    # --- Build the space-time kymograph over the band ---
    # The front carries no brightness/color step (it is a refractive-index boundary),
    # but it is a thin horizontal line. The vertical second derivative |d^2 I / dy^2|
    # responds to that line regardless of its polarity (dark or bright hairline) and
    # ignores smooth region-brightness changes. Averaging across the band width
    # (the front spans the whole tube) keeps the full-width front and dilutes local
    # bubbles/glints. Stacking the per-frame profiles gives L[y, t].
    L = np.empty((band_h, n_frames), dtype=np.float32)
    for t in range(n_frames):
        d2y = cv2.Sobel(band_frames[t].astype(np.float32), cv2.CV_32F, 0, 2, ksize=LINE_KSIZE)
        L[:, t] = np.abs(d2y).mean(axis=1)
    L = cv2.GaussianBlur(L, (KYMO_SMOOTH, KYMO_SMOOTH), 0)
    # Remove static horizontal features (tube bottom, meniscus, fixed markings): they
    # are constant in time, so subtracting each row's temporal median cancels them
    # while the transient, moving front survives. The diagonal is then the dominant
    # feature on the kymograph.
    L = np.clip(L - np.median(L, axis=1, keepdims=True), 0, None)

    # --- Detect the diagonal with a Hough transform on the partial kymograph ---
    # The first third of the plot behaves differently from the steady diagonal (the
    # jostling/initiation blob lives there, before the front is in the band), so we
    # ignore it. On the remaining columns we keep only the most-intense ridge pixels
    # (a percentile threshold) so the diagonal is isolated, then run a probabilistic
    # Hough transform and take the longest downward segment as the front. Finally we
    # least-squares refine the slope on the ridge pixels that support that segment.
    t_start = int(n_frames * FIT_START_FRACTION)
    Lp = L[:, t_start:]                       # partial kymograph (time-cropped)
    n_cols = Lp.shape[1]

    thresh = np.percentile(Lp, RIDGE_PCTL)
    mask = (Lp >= thresh).astype(np.uint8) * 255

    slope = intercept = speed_px_s = r2 = np.nan
    n_inliers = 0
    status = 'FAILED_NO_STABLE_FRONT'

    min_len = max(10, int(HOUGH_MIN_LINE_FRAC * n_cols))
    max_gap = max(1, int(HOUGH_MAX_GAP_FRAC * n_cols))
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=min_len,
                            minLineLength=min_len, maxLineGap=max_gap)

    if lines is not None:
        # Longest segment that travels downward (row increases with time -> slope > 0).
        best = None
        best_len = 0.0
        for x1, y1, x2, y2 in lines[:, 0, :]:
            if x2 == x1:
                continue
            seg_slope = (y2 - y1) / (x2 - x1)
            if seg_slope <= 0:
                continue
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length > best_len:
                best_len = length
                best = (seg_slope, y1 - seg_slope * x1)   # (slope, intercept) in partial coords

        if best is not None:
            seed_slope, seed_inter = best
            # Ridge pixels supporting the seed line -> refine the slope on them (in
            # partial coords x measured from t_start, y = row within the band).
            ys_px, xs_px = np.nonzero(mask)
            pred = seed_slope * xs_px + seed_inter
            inl = np.abs(ys_px - pred) < INLIER_TOL_FRAC * band_h
            n_inliers = int(inl.sum())
            if n_inliers >= 10:
                res = stats.linregress(xs_px[inl].astype(float), ys_px[inl].astype(float))
                if res.slope > 0:
                    slope, r2 = res.slope, res.rvalue ** 2
                    # Shift intercept back onto the full (un-cropped) frame time axis.
                    intercept = res.intercept - res.slope * t_start
                    speed_px_s = abs(slope) * fps
                    status = 'OK' if r2 >= 0.80 else 'FAILED_NO_STABLE_FRONT'

    if CALIBRATION_MM_PER_PX is not None:
        speed = speed_px_s * CALIBRATION_MM_PER_PX
        speed_unit = 'mm/s'
    else:
        speed = speed_px_s
        speed_unit = 'px/s'

    print(f"  Band rows [{band_top}:{band_bottom}] cols [{band_left}:{band_right}], "
          f"kymograph {L.shape[0]}x{L.shape[1]}, fit inliers: {n_inliers}")
    print(f"  Status: {status}, Speed: {speed:.3f} {speed_unit}, R²: {r2:.4f}")

    # Per-column ridge (most-intense pixels, fit region only) and the fitted diagonal,
    # both in full-frame px coords.
    times = np.arange(n_frames) / fps
    col_peak = np.argmax(L, axis=0).astype(float)
    col_max = L.max(axis=0)
    show = (np.arange(n_frames) >= t_start) & (col_max >= thresh)
    front_raw = np.where(show, band_top + col_peak, np.nan)
    front_fit = (band_top + slope * np.arange(n_frames) + intercept
                 if not np.isnan(slope) else np.full(n_frames, np.nan))

    # Per-video position-time CSV
    pd.DataFrame({
        'time_s': times,
        'front_y_px_raw': front_raw,
        'front_y_px_fit': front_fit,
    }).to_csv(os.path.join(OUTPUT_DATA_DIR, f'{stem}_position_time.csv'), index=False)

    # Position-vs-time plot: gated ridge points + fitted diagonal
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(times, front_raw, s=2, alpha=0.3, label='Ridge (intense pixels)')
    if not np.isnan(slope):
        ax.plot(times, front_fit, 'r-', label=f'Fit: {speed:.2f} {speed_unit}, R²={r2:.3f}')
    ax.axhspan(band_top, band_bottom, color='gray', alpha=0.08, label='Band')
    ax.axvspan(times[0], times[t_start], color='red', alpha=0.08, label='Ignored (first 1/3)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Front position (px)')
    ax.set_ylim(H, 0)
    ax.set_title(f'{stem}\n{speed:.2f} {speed_unit}  [{status}]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_IMAGE_DIR, f'{stem}_position_time.png'), dpi=150)
    plt.close(fig)

    # Diagnostic: band kymograph (x=time, y=frame row) with the fitted diagonal overlaid.
    # Grayscale with the most-intense signal rendered black (gray_r) for readability.
    figk, axk = plt.subplots(figsize=(8, 5))
    axk.imshow(L, cmap='gray_r', aspect='auto', origin='upper',
               extent=[times[0], times[-1], band_bottom, band_top])
    # Shade the ignored first third (jostling/initiation) that is excluded from the fit.
    axk.axvspan(times[0], times[t_start], color='red', alpha=0.10,
                label='Ignored (first 1/3)')
    if not np.isnan(slope):
        axk.plot(times, front_fit, 'r-', lw=1.5,
                 label=f'Fit: {speed:.2f} {speed_unit}, R²={r2:.3f}')
    axk.legend()
    axk.set_xlabel('Time (s)')
    axk.set_ylabel('Front position (px)')
    axk.set_title(f'{stem} kymograph\n{speed:.2f} {speed_unit}  [{status}]')
    figk.tight_layout()
    figk.savefig(os.path.join(OUTPUT_IMAGE_DIR, f'{stem}_kymograph.png'), dpi=150)
    plt.close(figk)

    # Annotated video: draw the fitted front line on each frame.
    cap2 = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(OUTPUT_VIDEO_DIR, f'{stem}_annotated.mp4'), fourcc, fps, (frame_w, frame_h))
    for i in range(n_frames):
        ret, frame = cap2.read()
        if not ret:
            break
        # middle-50% analysis band
        cv2.rectangle(frame, (band_left, band_top), (band_right, band_bottom), (255, 255, 0), 1)
        fy = front_fit[i]
        if not np.isnan(fy) and 0 <= fy < frame_h:
            cv2.line(frame, (0, int(fy)), (frame_w, int(fy)), (0, 0, 255), 2)
        cv2.putText(frame, f'{speed:.2f} {speed_unit}  [{status}]', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
        'direction': 'down',
        **meta,
    })

results_path = os.path.join(OUTPUT_DATA_DIR, 'front_speed_results.csv')
pd.DataFrame(results).to_csv(results_path, index=False)
print(f'\nResults saved to {results_path}')
print(pd.DataFrame(results).to_string())
