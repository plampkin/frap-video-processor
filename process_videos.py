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
# The front is a thin horizontal feature (the unreacted->reacted boundary) with no
# reliable brightness polarity, so the kymograph response is the polarity-agnostic
# vertical SECOND derivative |d^2 I/dy^2| (LINE_KSIZE): it fires as a symmetric ridge
# on the front whether it reads slightly dark or bright, while the uniform interior
# behind the front gives ~zero. Averaging across the band width keeps the full-width
# line and dilutes local bubbles/glints; subtracting each row's temporal median then
# cancels the static horizontal banding. On the space-time kymograph the moving front
# is one clear diagonal whose slope is the front speed (rows per frame). The speed can vary
# wildly between runs, so we do NOT assume a narrow speed range. Instead we find the
# diagonal with a Radon-style line-integral search over a WIDE range of downward
# slopes: summing the kymograph response along each candidate line raises the SNR of a
# faint front, and near-horizontal lines (bubble bands / residual static features) are
# excluded by the slope range so they can never win. The maximizing line seeds a
# per-column leading-edge trace, which is then fit robustly (Theil-Sen -> OLS).
LINE_KSIZE = 3                  # vertical 2nd-derivative line kernel (polarity-agnostic)
KYMO_SMOOTH = 5                 # Gaussian smoothing of the kymograph (odd)
START_FRACTION = 1.0 / 3.0      # ignore the first ~1/3 (jostling + initiation transient)
EDGE_PCTL = 80                  # edge-presence threshold percentile (retained region only)
N_SLOPES = 240                  # number of candidate downward slopes in the Radon search
RIDGE_OFFSET_FRAC = 0.03        # ridge-contrast offset (frac of band height): wider than the
                                # front ridge, narrower than a bubble band
MIN_FRONT_SPEED_PX_S = 0.1      # real speed floor (px/s): drops near-horizontal bubble bands
MIN_TRANSIT_FRAC = 0.05         # fastest front: can't cross the band in < this frac of frames
MIN_SUPPORT_FRAC = 0.15         # a candidate line must span >= this frac of retained frames
EDGE_HALF_FRAC = 0.06           # half-window (frac of band height) for the per-column edge search
INLIER_TOL_FRAC = 0.05          # fit inlier tolerance, as a fraction of band height
MIN_COVERAGE_FRAC = 0.50        # inliers must span >= this frac of the post-cut time axis
MIN_FILL_FRAC = 0.50            # >= this frac of spanned columns must carry a front pixel

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
    # The front is a thin horizontal line (the unreacted->reacted boundary) with no
    # reliable brightness polarity, so each band frame is run through a polarity-agnostic
    # vertical second-derivative line detector |d^2 I/dy^2| (LINE_KSIZE): it responds as a
    # symmetric ridge on the front whether it reads slightly dark or bright, while the
    # uniform interior behind the front gives ~zero. Averaging across the band width (the
    # front spans the whole tube) keeps the full-width line and dilutes local
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

    # --- Ignore the first ~1/3 of the time axis ---
    # That region is test-tube jostling + soldering-iron initiation, not a steady front.
    # The threshold, the line search, and the fit all use only the retained columns; the
    # steady diagonal persists well past the cut, so the steady-state speed is unaffected.
    t_start = int(START_FRACTION * n_frames)
    n_ret = n_frames - t_start

    # Edge-presence threshold from the RETAINED region only, so the dense early
    # initiation blobs cannot bias the percentile.
    edge_thr = float(np.percentile(L[:, t_start:], EDGE_PCTL))

    # --- Find the dominant downward diagonal with a wide-range Radon search ---
    # The front speed varies wildly between runs, so we search a WIDE band of downward
    # slopes. For each candidate line y = m*t + b we score the RIDGE CONTRAST: the mean
    # response on the line minus the mean response at a parallel offset above and below it,
    # over the retained columns only. The front is a thin ridge -- dark on the line, light
    # just off it -- so it scores high; a thick dark bubble band is dark both on and off the
    # line, so it scores ~0. A plain mean-along-the-line score instead rewards any
    # near-horizontal line lying inside the thick band (the AIBN_300_1 failure), since the
    # band's interior is uniformly high. The offset (RIDGE_OFFSET_FRAC of the band height)
    # is wider than the front ridge but narrower than a band. Near-horizontal lines are also
    # excluded by the slope range, so static features can never win. The maximizing (m, b)
    # seeds the per-column leading-edge trace below.
    slope = intercept = speed_px_s = np.nan
    coverage_frac = fill_frac = np.nan
    n_inliers = 0
    status = 'FAILED_NO_STABLE_FRONT'

    t_idx = np.arange(n_frames)
    b_idx = np.arange(band_h)
    m_min = MIN_FRONT_SPEED_PX_S / fps                         # speed floor (rows/frame)
    m_max = band_h / max(1.0, MIN_TRANSIT_FRAC * n_frames)     # fastest plausible front
    slopes = np.linspace(m_min, m_max, N_SLOPES)
    min_support = max(10, int(MIN_SUPPORT_FRAC * n_ret))
    half = max(3, int(EDGE_HALF_FRAC * band_h))

    best_score = -1.0
    seed_slope = seed_inter = np.nan
    off = max(2, int(RIDGE_OFFSET_FRAC * band_h))            # ridge-contrast offset (rows)
    for m in slopes:
        shifts = np.round(m * t_idx).astype(int)              # row offset of the line per column
        target = b_idx[:, None] + shifts[None, :]            # row index hit at each (b, t)
        valid = (target >= 0) & (target < band_h)
        valid[:, :t_start] = False                           # retained columns only
        on_line = L[np.clip(target, 0, band_h - 1), t_idx[None, :]]
        up = L[np.clip(target - off, 0, band_h - 1), t_idx[None, :]]
        down = L[np.clip(target + off, 0, band_h - 1), t_idx[None, :]]
        on_line[~valid] = 0.0
        up[~valid] = 0.0
        down[~valid] = 0.0
        cnt = valid.sum(axis=1)
        denom = np.maximum(cnt, 1)
        contrast = on_line.sum(axis=1) / denom - 0.5 * (up.sum(axis=1) + down.sum(axis=1)) / denom
        score = np.where(cnt >= min_support, contrast, 0.0)
        bi = int(np.argmax(score))
        if score[bi] > best_score:
            best_score = float(score[bi])
            seed_slope, seed_inter = float(m), float(bi)

    # --- Per-column leading edge around the seed line ---
    # The front direction is fixed downward (reacted/dark above, unreacted/light below),
    # so per column we take the LEADING edge: within the window around the seed line,
    # scanning from the unreacted (advancing) side -- the bottom of the window -- toward
    # the reacted region, the FIRST row whose response crosses the edge threshold. This
    # is the deepest crossing in the window. The bubble-banded interior sits BEHIND the
    # front (above it) and is therefore excluded; an intensity-weighted centroid would
    # instead average that banding in and be pulled shallow off the true edge (the
    # V70_800 failure). Columns with no crossing in the window stay NaN, so the front may
    # enter the band at any time.
    front_band = np.full(n_frames, np.nan)
    if not np.isnan(seed_slope):
        pred_rows = seed_slope * t_idx + seed_inter
        for t in range(t_start, n_frames):
            lo = int(max(0, np.floor(pred_rows[t] - half)))
            hi = int(min(band_h, np.ceil(pred_rows[t] + half) + 1))
            if hi - lo < 3:
                continue
            crossings = np.nonzero(L[lo:hi, t] >= edge_thr)[0]
            if crossings.size:
                front_band[t] = float(lo + crossings[-1])

    # --- Robust fit of the leading-edge trace: Theil-Sen seed -> OLS on inliers ---
    valid_cols = ~np.isnan(front_band)
    if valid_cols.sum() >= min_support:
        xs = t_idx[valid_cols].astype(float)
        ys = front_band[valid_cols]
        ts_slope, ts_inter, _, _ = stats.theilslopes(ys, xs)
        inl = np.abs(ys - (ts_slope * xs + ts_inter)) < INLIER_TOL_FRAC * band_h
        n_inliers = int(inl.sum())
        if n_inliers >= 10:
            res = stats.linregress(xs[inl], ys[inl])
            if res.slope > 0:                                # downward front required
                slope = res.slope
                intercept = res.intercept                    # band coords, full time axis
                speed_px_s = abs(slope) * fps

                # --- Accept on GEOMETRY, not R^2 ---
                # R^2 is meaningless here (the points were selected for lying on a line,
                # so it is ~1 by construction). Instead require (1) the inliers to span a
                # large fraction of the post-cut time axis and (2) most of those spanned
                # columns to actually carry a front pixel (continuity / fill).
                xin = xs[inl]
                lo_t, hi_t = int(xin.min()), int(xin.max())
                span_cols = np.arange(lo_t, hi_t + 1)
                coverage_frac = len(span_cols) / max(1, n_ret)
                pred = slope * span_cols + intercept
                carries = 0
                for tt, pr in zip(span_cols, pred):
                    a = int(max(0, np.floor(pr - half)))
                    bb = int(min(band_h, np.ceil(pr + half) + 1))
                    if bb > a and L[a:bb, tt].max() >= edge_thr:
                        carries += 1
                fill_frac = carries / max(1, len(span_cols))
                # A near-horizontal bubble band is continuous and spans the time axis,
                # so coverage/fill cannot reject it -- only a real speed floor can. Require
                # the fitted speed to clear MIN_FRONT_SPEED_PX_S as well.
                status = ('OK' if coverage_frac >= MIN_COVERAGE_FRAC
                          and fill_frac >= MIN_FILL_FRAC
                          and speed_px_s >= MIN_FRONT_SPEED_PX_S
                          else 'FAILED_NO_STABLE_FRONT')

    if CALIBRATION_MM_PER_PX is not None:
        speed = speed_px_s * CALIBRATION_MM_PER_PX
        speed_unit = 'mm/s'
    else:
        speed = speed_px_s
        speed_unit = 'px/s'

    print(f"  Band rows [{band_top}:{band_bottom}] cols [{band_left}:{band_right}], "
          f"kymograph {L.shape[0]}x{L.shape[1]}, fit inliers: {n_inliers}")
    print(f"  Status: {status}, Speed: {speed:.3f} {speed_unit}, "
          f"coverage: {coverage_frac:.2f}, fill: {fill_frac:.2f}")

    # Per-column leading-edge trace and the fitted diagonal, both in full-frame px.
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

    # Position-vs-time plot: leading-edge points + fitted diagonal
    t_cut_s = t_start / fps
    fit_label = (f'Fit: {speed:.2f} {speed_unit}, '
                 f'cov={coverage_frac:.2f}, fill={fill_frac:.2f}')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(times, front_raw, s=4, alpha=0.5, label='Front leading edge')
    if not np.isnan(slope):
        ax.plot(times, front_fit, 'r-', label=fit_label)
    ax.axvspan(times[0], t_cut_s, color='gray', alpha=0.15, label='Ignored start')
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
    axk.scatter(times, front_raw, s=4, c='tab:blue', alpha=0.5, label='Front leading edge')
    if not np.isnan(slope):
        axk.plot(times, front_fit, 'r-', lw=1.5, label=fit_label)
    axk.axvspan(times[0], t_cut_s, color='tab:red', alpha=0.10, label='Ignored start')
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
        'coverage': round(coverage_frac, 4) if not np.isnan(coverage_frac) else np.nan,
        'fill': round(fill_frac, 4) if not np.isnan(fill_frac) else np.nan,
        'direction': 'down',
        **meta,
    })

results_path = os.path.join(OUTPUT_DATA_DIR, 'front_speed_results.csv')
pd.DataFrame(results).to_csv(results_path, index=False)
print(f'\nResults saved to {results_path}')
print(pd.DataFrame(results).to_string())
