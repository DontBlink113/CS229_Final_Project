import numpy as np
from scipy.ndimage import gaussian_filter


N_DIRS = 8
GRID   = 8
SIGMA  = 1.0


def resample_stroke(stroke, spacing=2.0):
    stroke = np.array(stroke, dtype=float)
    if len(stroke) < 2:
        return stroke
    diffs = np.diff(stroke, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len == 0:
        return stroke[[0]]
    num_points = max(2, int(round(total_len / spacing)) + 1)
    t_new = np.linspace(0, total_len, num_points)
    xs = np.interp(t_new, cum_len, stroke[:, 0])
    ys = np.interp(t_new, cum_len, stroke[:, 1])
    return np.stack([xs, ys], axis=1)


def normalize_P2DBMN(strokes, grid=4):
    all_pts = np.concatenate([np.array(s) for s in strokes if len(s) > 0])
    x_edges = np.percentile(all_pts[:, 0], np.linspace(0, 100, grid + 1))
    x_edges[0] -= 1e-6; x_edges[-1] += 1e-6

    def bimom_norm(vals):
        pivot = vals.mean()
        lo = vals[vals <= pivot]
        hi = vals[vals > pivot]
        s_lo = abs(pivot - lo.min()) if len(lo) > 1 and lo.min() < pivot else 1.0
        s_hi = abs(hi.max() - pivot) if len(hi) > 1 and hi.max() > pivot else 1.0
        return np.where(vals <= pivot, (vals - pivot) / s_lo, (vals - pivot) / s_hi)

    lengths = [len(s) for s in strokes]
    flat = all_pts.copy()

    for i in range(grid):
        mask = (flat[:, 0] >= x_edges[i]) & (flat[:, 0] < x_edges[i + 1])
        if mask.sum() >= 2:
            flat[mask, 1] = bimom_norm(flat[mask, 1])

    y_edges = np.percentile(flat[:, 1], np.linspace(0, 100, grid + 1))
    y_edges[0] -= 1e-6; y_edges[-1] += 1e-6
    for i in range(grid):
        mask = (flat[:, 1] >= y_edges[i]) & (flat[:, 1] < y_edges[i + 1])
        if mask.sum() >= 2:
            flat[mask, 0] = bimom_norm(flat[mask, 0])

    out, idx = [], 0
    for l in lengths:
        out.append(flat[idx:idx + l])
        idx += l
    return out


def extract_online_features(strokes):
    strokes = [np.array(s, dtype=float) for s in strokes if len(s) >= 2]
    if not strokes:
        return np.zeros(512)

    strokes = [resample_stroke(s) for s in strokes]
    strokes = [s for s in strokes if len(s) >= 2]
    if not strokes:
        return np.zeros(512)

    strokes = normalize_P2DBMN(strokes)

    all_pts = np.concatenate(strokes)
    xmin, ymin = all_pts[:, 0].min(), all_pts[:, 1].min()
    xrange = (all_pts[:, 0].max() - xmin) or 1.0
    yrange = (all_pts[:, 1].max() - ymin) or 1.0

    maps = np.zeros((N_DIRS, GRID, GRID))
    for stroke in strokes:
        stroke = np.array(stroke)
        for i in range(len(stroke) - 1):
            dx, dy = stroke[i+1][0] - stroke[i][0], stroke[i+1][1] - stroke[i][1]
            if dx == 0 and dy == 0:
                continue
            angle_norm = (np.arctan2(dy, dx) / (2 * np.pi) * N_DIRS) % N_DIRS
            d_lo, d_hi = int(angle_norm) % N_DIRS, (int(angle_norm) + 1) % N_DIRS
            w_hi = angle_norm - int(angle_norm)
            mx = np.clip(int((((stroke[i][0]+stroke[i+1][0])/2) - xmin) / xrange * (GRID-1e-9)), 0, GRID-1)
            my = np.clip(int((((stroke[i][1]+stroke[i+1][1])/2) - ymin) / yrange * (GRID-1e-9)), 0, GRID-1)
            maps[d_lo, my, mx] += 1 - w_hi
            maps[d_hi, my, mx] += w_hi

    blurred = np.stack([gaussian_filter(maps[d], sigma=SIGMA) for d in range(N_DIRS)])
    return np.sqrt(np.clip(blurred.flatten(), 0, None))


def extract_features_batch(chars_dict):
    X, y = [], []
    for label, samples in chars_dict.items():
        for strokes in samples:
            X.append(extract_online_features(strokes))
            y.append(label)
        print(f"label {label} complete")
    return np.array(X), y



def plot_feature_vector(feat, strokes=None, title='Feature vector'):
    
    
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    DIR_NAMES = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
    DIR_ANGLES_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
    GRID = 8

    # Reshape: (512,) → (8, 8, 8)  [dir, row, col]
    maps = feat.reshape(8, GRID, GRID)
    global_max = maps.max() or 1.0

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title, fontsize=13, y=0.98)

    # --- top row ---
    if strokes is not None:
        ax_strokes = fig.add_axes([0.02, 0.45, 0.18, 0.48])
        ax_strokes.set_title('Strokes', fontsize=10)
        ax_strokes.set_aspect('equal')
        ax_strokes.invert_yaxis()
        ax_strokes.axis('off')
        for stroke in strokes:
            pts = np.array(stroke)
            ax_strokes.plot(pts[:, 0], pts[:, 1], 'b-', lw=1.5)
            ax_strokes.plot(pts[0, 0], pts[0, 1], 'go', ms=4)
        feat_left = 0.23
    else:
        feat_left = 0.05

    # Full feature heatmap (8 dir columns × 8 rows, each cell = one 8×8 map)
    ax_feat = fig.add_axes([feat_left, 0.45, 0.94 - feat_left, 0.48])
    ax_feat.set_title('512-d feature vector  (8 directions × 8×8 spatial grid)', fontsize=10)

    # Tile all 8 direction maps side by side: shape (8, 8×8) → (8, 64)
    tiled = np.concatenate([maps[d] for d in range(8)], axis=1)  # (8, 64)
    ax_feat.imshow(tiled, aspect='auto', cmap='YlGn', vmin=0, vmax=global_max,
                   interpolation='nearest', origin='upper')

    # Dividers between direction blocks
    for d in range(1, 8):
        ax_feat.axvline(d * GRID - 0.5, color='white', lw=1.2)

    # Direction labels along top
    for d, name in enumerate(DIR_NAMES):
        ax_feat.text(d * GRID + GRID / 2 - 0.5, -0.8, name,
                     ha='center', va='bottom', fontsize=11)

    ax_feat.set_xticks([])
    ax_feat.set_yticks(range(GRID))
    ax_feat.set_yticklabels([f'r{i}' for i in range(GRID)], fontsize=7)

    # --- bottom row: 8 individual direction maps ---
    for d in range(8):
        left = 0.03 + d * 0.121
        ax = fig.add_axes([left, 0.04, 0.105, 0.35])
        ax.imshow(maps[d], cmap='YlGn', vmin=0, vmax=global_max,
                  interpolation='nearest', origin='upper')
        ax.set_title(DIR_NAMES[d], fontsize=12, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

        # Overlay direction arrows on active cells
        ang = np.radians(DIR_ANGLES_DEG[d])
        dx_arr = np.cos(ang) * 0.3
        dy_arr = np.sin(ang) * 0.3
        threshold = global_max * 0.05
        for row in range(GRID):
            for col in range(GRID):
                v = maps[d, row, col]
                if v > threshold:
                    scale = 0.2 + 0.5 * (v / global_max)
                    ax.annotate('', xy=(col + dx_arr * scale, row + dy_arr * scale),
                                xytext=(col - dx_arr * scale, row - dy_arr * scale),
                                arrowprops=dict(arrowstyle='->', color='white',
                                                lw=0.8 + v / global_max))

    plt.colorbar(
        plt.cm.ScalarMappable(cmap='YlGn',
                              norm=plt.Normalize(vmin=0, vmax=global_max)),
        ax=fig.axes, shrink=0.6, pad=0.01, label='activation'
    )
    plt.show()






