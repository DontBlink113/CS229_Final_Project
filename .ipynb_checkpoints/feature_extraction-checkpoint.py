import numpy as np
from scipy.ndimage import gaussian_filter
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def resample_stroke(stroke, num_points=None, spacing=1.0):
    """
    Resample a stroke to equally-spaced points by arc length.
    If num_points is None, resample at the given pixel spacing.
    """
    stroke = np.array(stroke, dtype=float)
    if len(stroke) < 2:
        return stroke
    diffs = np.diff(stroke, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len == 0:
        return stroke[[0]]
    if num_points is None:
        num_points = max(2, int(round(total_len / spacing)) + 1)
    t_new = np.linspace(0, total_len, num_points)
    xs = np.interp(t_new, cum_len, stroke[:, 0])
    ys = np.interp(t_new, cum_len, stroke[:, 1])
    return np.stack([xs, ys], axis=1)
 
 
def get_bounding_box(strokes):
    """Return (xmin, ymin, xmax, ymax) over all strokes."""
    all_pts = np.concatenate([np.array(s) for s in strokes if len(s) > 0])
    return all_pts[:, 0].min(), all_pts[:, 1].min(), \
           all_pts[:, 0].max(), all_pts[:, 1].max()
 
 
def get_moments(strokes):
    """Compute 1st and 2nd moments (mean and std) of all points, per axis."""
    all_pts = np.concatenate([np.array(s) for s in strokes if len(s) > 0])
    mx, my = all_pts[:, 0].mean(), all_pts[:, 1].mean()
    sx, sy = all_pts[:, 0].std(), all_pts[:, 1].std()
    sx = sx if sx > 1e-6 else 1.0
    sy = sy if sy > 1e-6 else 1.0
    return mx, my, sx, sy
 
 
# ---------------------------------------------------------------------------
# 1.  Normalization methods  (Section 3.2 of the paper)
# ---------------------------------------------------------------------------
 
def normalize_LN(strokes):
    """
    Linear Normalization: scale bounding box to [0, 1] x [0, 1].
    Uniform scale (preserve aspect ratio) centred in the unit square.
    """
    xmin, ymin, xmax, ymax = get_bounding_box(strokes)
    w = xmax - xmin or 1.0
    h = ymax - ymin or 1.0
    scale = max(w, h)
    out = []
    for s in strokes:
        s = np.array(s, dtype=float)
        s[:, 0] = (s[:, 0] - xmin) / scale
        s[:, 1] = (s[:, 1] - ymin) / scale
        out.append(s)
    return out
 
 
def normalize_MN(strokes):
    """
    Moment Normalization: shift to zero mean, scale to unit std, per axis.
    """
    mx, my, sx, sy = get_moments(strokes)
    out = []
    for s in strokes:
        s = np.array(s, dtype=float)
        s[:, 0] = (s[:, 0] - mx) / sx
        s[:, 1] = (s[:, 1] - my) / sy
        out.append(s)
    return out
 
 
def normalize_BMN(strokes):
    """
    Bi-Moment Normalization: separately normalise upper/lower halves (y-axis)
    and left/right halves (x-axis) to equalise stroke density.
    Based on Liu et al. ICDAR 2003.
    """
    all_pts = np.concatenate([np.array(s) for s in strokes if len(s) > 0])
    mx = all_pts[:, 0].mean()
    my = all_pts[:, 1].mean()
 
    def half_scale(vals, pivot):
        lo = vals[vals <= pivot]
        hi = vals[vals > pivot]
        s_lo = (pivot - lo.min()) if len(lo) > 1 and lo.min() < pivot else 1.0
        s_hi = (hi.max() - pivot) if len(hi) > 1 and hi.max() > pivot else 1.0
        return s_lo, s_hi
 
    sx_lo, sx_hi = half_scale(all_pts[:, 0], mx)
    sy_lo, sy_hi = half_scale(all_pts[:, 1], my)
 
    out = []
    for s in strokes:
        s = np.array(s, dtype=float)
        xs, ys = s[:, 0].copy(), s[:, 1].copy()
        xs = np.where(xs <= mx, (xs - mx) / sx_lo, (xs - mx) / sx_hi)
        ys = np.where(ys <= my, (ys - my) / sy_lo, (ys - my) / sy_hi)
        out.append(np.stack([xs, ys], axis=1))
    return out
 
 
def normalize_P2DMN(strokes, grid=4):
    """
    Pseudo-2D Moment Normalization: divide into `grid` vertical slabs,
    apply independent MN per slab on y; then divide into `grid` horizontal
    slabs and apply independent MN per slab on x.
    """
    all_pts = np.concatenate([np.array(s) for s in strokes if len(s) > 0])
 
    # --- pass 1: vertical slabs, normalise y ---
    x_edges = np.percentile(all_pts[:, 0], np.linspace(0, 100, grid + 1))
    x_edges[0] -= 1e-6; x_edges[-1] += 1e-6
 
    def slab_norm_y(pts):
        out_pts = pts.copy()
        for i in range(grid):
            mask = (pts[:, 0] >= x_edges[i]) & (pts[:, 0] < x_edges[i + 1])
            if mask.sum() < 2:
                continue
            m = pts[mask, 1].mean()
            s = pts[mask, 1].std() or 1.0
            out_pts[mask, 1] = (pts[mask, 1] - m) / s
        return out_pts
 
    # --- pass 2: horizontal slabs, normalise x ---
    def slab_norm_x(pts):
        y_edges = np.percentile(pts[:, 1], np.linspace(0, 100, grid + 1))
        y_edges[0] -= 1e-6; y_edges[-1] += 1e-6
        out_pts = pts.copy()
        for i in range(grid):
            mask = (pts[:, 1] >= y_edges[i]) & (pts[:, 1] < y_edges[i + 1])
            if mask.sum() < 2:
                continue
            m = pts[mask, 0].mean()
            s = pts[mask, 0].std() or 1.0
            out_pts[mask, 0] = (pts[mask, 0] - m) / s
        return out_pts
 
    # Apply to concatenated point cloud, then split back into strokes
    lengths = [len(s) for s in strokes]
    flat = all_pts.copy()
    flat = slab_norm_y(flat)
    flat = slab_norm_x(flat)
 
    out = []
    idx = 0
    for l in lengths:
        out.append(flat[idx:idx + l])
        idx += l
    return out
 
 
def normalize_P2DBMN(strokes, grid=4):
    """
    Pseudo-2D Bi-Moment Normalization: like P2DMN but uses bi-moment
    (BMN) within each slab instead of simple mean/std.
    This is the best-performing method in the paper for online data.
    """
    all_pts = np.concatenate([np.array(s) for s in strokes if len(s) > 0])
    x_edges = np.percentile(all_pts[:, 0], np.linspace(0, 100, grid + 1))
    x_edges[0] -= 1e-6; x_edges[-1] += 1e-6
 
    def bimom_norm(vals):
        pivot = vals.mean()
        lo = vals[vals <= pivot]
        hi = vals[vals > pivot]
        s_lo = abs(pivot - lo.min()) if len(lo) > 1 and lo.min() < pivot else 1.0
        s_hi = abs(hi.max() - pivot) if len(hi) > 1 and hi.max() > pivot else 1.0
        return np.where(vals <= pivot,
                        (vals - pivot) / s_lo,
                        (vals - pivot) / s_hi)
 
    lengths = [len(s) for s in strokes]
    flat = all_pts.copy()
 
    # pass 1: normalise y within vertical slabs
    for i in range(grid):
        mask = (flat[:, 0] >= x_edges[i]) & (flat[:, 0] < x_edges[i + 1])
        if mask.sum() < 2:
            continue
        flat[mask, 1] = bimom_norm(flat[mask, 1])
 
    # pass 2: normalise x within horizontal slabs
    y_edges = np.percentile(flat[:, 1], np.linspace(0, 100, grid + 1))
    y_edges[0] -= 1e-6; y_edges[-1] += 1e-6
    for i in range(grid):
        mask = (flat[:, 1] >= y_edges[i]) & (flat[:, 1] < y_edges[i + 1])
        if mask.sum() < 2:
            continue
        flat[mask, 0] = bimom_norm(flat[mask, 0])
 
    out = []
    idx = 0
    for l in lengths:
        out.append(flat[idx:idx + l])
        idx += l
    return out
 
 
NORMALIZERS = {
    'LN':     normalize_LN,
    'MN':     normalize_MN,
    'BMN':    normalize_BMN,
    'P2DMN':  normalize_P2DMN,
    'P2DBMN': normalize_P2DBMN,
}
 
 
# ---------------------------------------------------------------------------
# 2.  Direction feature extraction  (Section 3.2)
# ---------------------------------------------------------------------------
 
N_DIRS = 8      # 8 chaincode directions
GRID   = 8      # 8x8 spatial grid
SIGMA  = 1.0    # Gaussian blur sigma (in grid cells)
 
 
def point_to_direction_weights(dx, dy):
    """
    Decompose a (dx, dy) vector into weights over 8 directions using the
    parallelogram rule (bilinear decomposition into two adjacent directions).
    Returns array of shape (8,) summing to 1.
    """
    if dx == 0 and dy == 0:
        return np.zeros(N_DIRS)
    angle = np.arctan2(dy, dx)          # [-pi, pi]
    angle_norm = angle / (2 * np.pi) * N_DIRS  # [−4, 4]
    angle_norm = angle_norm % N_DIRS           # [0, 8)
    d_lo = int(angle_norm) % N_DIRS
    d_hi = (d_lo + 1) % N_DIRS
    w_hi = angle_norm - int(angle_norm)
    w_lo = 1.0 - w_hi
    weights = np.zeros(N_DIRS)
    weights[d_lo] += w_lo
    weights[d_hi] += w_hi
    return weights
 
 
def extract_direction_maps(strokes, use_pen_lifts=False, pen_lift_weight=0.5):
    """
    Build 8 direction accumulation maps of size (GRID, GRID).
 
    For each pair of adjacent points in a stroke, compute the direction
    of the line segment and accumulate into the spatial bin.
 
    If use_pen_lifts=True, also add imaginary strokes (pen-lift segments
    connecting end of one stroke to start of next) with the given weight.
 
    Returns array of shape (N_DIRS, GRID, GRID).
    """
    # Find coordinate range for grid mapping
    all_pts = np.concatenate([np.array(s) for s in strokes if len(s) > 0])
    xmin, ymin = all_pts[:, 0].min(), all_pts[:, 1].min()
    xmax, ymax = all_pts[:, 0].max(), all_pts[:, 1].max()
    xrange = xmax - xmin or 1.0
    yrange = ymax - ymin or 1.0
 
    direction_maps = np.zeros((N_DIRS, GRID, GRID))
 
    def accumulate_segment(p1, p2, weight=1.0):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dir_weights = point_to_direction_weights(dx, dy) * weight
        # midpoint position → grid cell
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        gx = int((mx - xmin) / xrange * (GRID - 1e-9))
        gy = int((my - ymin) / yrange * (GRID - 1e-9))
        gx = np.clip(gx, 0, GRID - 1)
        gy = np.clip(gy, 0, GRID - 1)
        direction_maps[:, gy, gx] += dir_weights
 
    # Real strokes
    for stroke in strokes:
        stroke = np.array(stroke)
        for i in range(len(stroke) - 1):
            accumulate_segment(stroke[i], stroke[i + 1])
 
    # Imaginary strokes (pen lifts)
    if use_pen_lifts:
        for i in range(len(strokes) - 1):
            s_end = np.array(strokes[i])[-1]
            s_start = np.array(strokes[i + 1])[0]
            accumulate_segment(s_end, s_start, weight=pen_lift_weight)
 
    return direction_maps
 
 
# ---------------------------------------------------------------------------
# 3.  Gaussian blur + flatten → 512-dim vector
# ---------------------------------------------------------------------------
 
def direction_maps_to_feature(direction_maps):
    """
    Apply per-direction 2D Gaussian blur, then flatten to 512-dim vector.
    shape: (8, 8, 8) → (512,)
    """
    blurred = np.stack([
        gaussian_filter(direction_maps[d], sigma=SIGMA)
        for d in range(N_DIRS)
    ])  # (8, 8, 8)
    return blurred.flatten()   # 512
 
 
# ---------------------------------------------------------------------------
# 4.  Box-Cox transform  (Section 3.3, pre-classification step)
# ---------------------------------------------------------------------------
 
def box_cox_transform(features):
    """y = x^0.5  (Box-Cox with lambda=0.5, as stated in the paper)."""
    return np.sqrt(np.clip(features, 0, None))
 
 
# ---------------------------------------------------------------------------
# 5.  Full pipeline
# ---------------------------------------------------------------------------
 
def extract_online_features(
    strokes,
    normalization='P2DBMN',
    direction_type='original_enhanced',
    apply_box_cox=True,
    resample=True,
    resample_spacing=2.0,
):
    """
    Full online feature extraction pipeline from Liu et al. 2012.
 
    Parameters
    ----------
    strokes : list of array-like, each shape (N, 2)
        Raw stroke data for one character.
    normalization : str
        One of 'LN', 'MN', 'BMN', 'P2DMN', 'P2DBMN'.
    direction_type : str
        'original'          – stroke direction only
        'normalized'        – direction of resampled/normalized trajectory
        'original_enhanced' – original + pen lifts (best result in paper)
    apply_box_cox : bool
        Apply x^0.5 transform before returning (used before classification).
    resample : bool
        Resample strokes to uniform arc-length spacing before processing.
    resample_spacing : float
        Arc-length spacing for resampling (in original coordinate units).
 
    Returns
    -------
    features : np.ndarray, shape (512,)
    """
    # Filter out empty/degenerate strokes
    strokes = [np.array(s, dtype=float) for s in strokes if len(s) >= 2]
    if len(strokes) == 0:
        return np.zeros(512)
 
    # Optional resampling to uniform spacing
    if resample:
        strokes = [resample_stroke(s, spacing=resample_spacing) for s in strokes]
        strokes = [s for s in strokes if len(s) >= 2]
    if len(strokes) == 0:
        return np.zeros(512)
 
    # Step 1: coordinate normalization
    norm_fn = NORMALIZERS.get(normalization)
    if norm_fn is None:
        raise ValueError(f"Unknown normalization '{normalization}'. "
                         f"Choose from {list(NORMALIZERS)}")
    strokes = norm_fn(strokes)
 
    # Step 2: direction feature extraction
    use_pen_lifts = (direction_type == 'original_enhanced')
    direction_maps = extract_direction_maps(strokes, use_pen_lifts=use_pen_lifts)
 
    # Step 3: Gaussian blur + flatten → 512-dim
    features = direction_maps_to_feature(direction_maps)
 
    # Step 4: Box-Cox transform (optional, apply before classification)
    if apply_box_cox:
        features = box_cox_transform(features)
 
    return features
 
 
def extract_features_batch(chars_dict, normalization='P2DBMN',
                           direction_type='original_enhanced'):
    """
    Extract features for all samples in a chars_dict.
 
    Returns
    -------
    X : np.ndarray, shape (N, 512)
    y : list of str, length N  — character labels
    """
    X, y = [], []
    for label, samples in chars_dict.items():
        for strokes in samples:
            feat = extract_online_features(
                strokes,
                normalization=normalization,
                direction_type=direction_type,
            )
            X.append(feat)
            y.append(label)
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






