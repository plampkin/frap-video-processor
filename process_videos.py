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
# the front speed (rows per frame). The front speed can vary wildly between runs, so
# we do NOT assume a narrow speed range. Instead we find the diagonal with a Radon-
# style line-integral search over a WIDE range of downward slopes: summing the
# kymograph response along each candidate line raises the SNR of a faint front, and
# near-horizontal lines (bubble bands / residual static features) are excluded by the
# slope range so they can never win. The maximizing line seeds a per-column centroid
# trace, which is then fit robustly (Theil-Sen -> OLS).
LINE_KSIZE = 3                  # vertical 2nd-derivative kernel (front-line detector)
KYMO_SMOOTH = 5                 # Gaussian smoothing of the kymograph (odd)
N_SLOPES = 240                  # number of candidate downward slopes in the Radon search
SLOPE_MIN_TRAVEL_FRAC = 0.10    # slowest front: covers >= this frac of band over the whole video
MIN_TRANSIT_FRAC = 0.05         # fastest front: can't cross the band in < this frac of frames
MIN_SUPPORT_FRAC = 0.15         # a candidate line must span >= this frac of frames in-band
CENTROID_HALF_FRAC = 0.06       # half-window (frac of band height) for the per-column centroid
CENTROID_GATE = 0.20            # keep columns whose centroid support >= this frac of the peak
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

    # --- Find the dominant downward diagonal with a wide-range Radon search ---
    # The front speed varies wildly between runs, so we search a WIDE band of downward
    # slopes. For each candidate line y = m*t + b we sum the kymograph response along
    # the line (normalized by its in-frame length). Summing the whole line lifts a
    # faint front out of the noise; near-horizontal lines are excluded by the slope
    # range, so bubble bands / residual static features can never accumulate. The
    # maximizing (m, b) seeds the per-column centroid trace below.
    slope = intercept = speed_px_s = r2 = np.nan
    n_inliers = 0
    status = 'FAILED_NO_STABLE_FRONT'

    t_idx = np.arange(n_frames)
    b_idx = np.arange(band_h)
    m_min = SLOPE_MIN_TRAVEL_FRAC * band_h / n_frames          # slowest plausible front
    m_max = band_h / max(1.0, MIN_TRANSIT_FRAC * n_frames)     # fastest plausible front
    slopes = np.linspace(m_min, m_max, N_SLOPES)
    min_support = max(10, int(MIN_SUPPORT_FRAC * n_frames))

    best_score = -1.0
    seed_slope = seed_inter = np.nan
    for m in slopes:
        shifts = np.round(m * t_idx).astype(int)              # row offset of the line per column
        target = b_idx[:, None] + shifts[None, :]            # row index hit at each (b, t)
        valid = (target >= 0) & (target < band_h)
        gathered = L[np.clip(target, 0, band_h - 1), t_idx[None, :]]
        gathered[~valid] = 0.0
        cnt = valid.sum(axis=1)
        score = np.where(cnt >= min_support, gathered.sum(axis=1) / np.maximum(cnt, 1), 0.0)
        bi = int(np.argmax(score))
        if score[bi] > best_score:
            best_score = float(score[bi])
            seed_slope, seed_inter = float(m), float(bi)

    # --- Per-column intensity-weighted centroid around the seed line ---
    # One sub-pixel front position per frame (not a pixel cloud) removes the shallow-
    # slope/intercept bias on faint fronts. Columns whose support is weak (front not yet
    # in the band) stay NaN, so the front may enter at any time.
    front_band = np.full(n_frames, np.nan)
    strength = np.zeros(n_frames)
    if not np.isnan(seed_slope):
        pred_rows = seed_slope * t_idx + seed_inter
        half = max(3, int(CENTROID_HALF_FRAC * band_h))
        for t in t_idx:
            lo = int(max(0, np.floor(pred_rows[t] - half)))
            hi = int(min(band_h, np.ceil(pred_rows[t] + half) + 1))
            if hi - lo < 3:
                continue
            seg = L[lo:hi, t]
            s = float(seg.sum())
            strength[t] = s
            if s > 0:
                front_band[t] = float((np.arange(lo, hi) * seg).sum() / s)
        gate = CENTROID_GATE * float(strength.max())
        front_band[strength < gate] = np.nan

    # --- Robust fit of the centroid trace: Theil-Sen seed -> OLS on inliers ---
    valid_cols = ~np.isnan(front_band)
    if valid_cols.sum() >= min_support:
        xs = t_idx[valid_cols].astype(float)
        ys = front_band[valid_cols]
        ts_slope, ts_inter, _, _ = stats.theilslopes(ys, xs)
        inl = np.abs(ys - (ts_slope * xs + ts_inter)) < INLIER_TOL_FRAC * band_h
        n_inliers = int(inl.sum())
        if n_inliers >= 10:
            res = stats.linregress(xs[inl], ys[inl])
            if res.slope > 0:
                slope, r2 = res.slope, res.rvalue ** 2
                intercept = res.intercept                    # band coords, full time axis
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

    # Per-column centroid trace (gated) and the fitted diagonal, both in full-frame px.
    times = np.arange(n_frames) / fps
    front_raw = np.where(~np.isnan(front_band), band_top + front_band, np.nan)
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
    ax.scatter(times, front_raw, s=4, alpha=0.5, label='Front centroid')
    if not np.isnan(slope):
        ax.plot(times, front_fit, 'r-', label=f'Fit: {speed:.2f} {speed_unit}, R²={r2:.3f}')
    ax.axhspan(band_top, band_bottom, color='gray', alpha=0.08, label='Band')
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
    axk.scatter(times, front_raw, s=4, c='tab:blue', alpha=0.5, label='Front centroid')
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
