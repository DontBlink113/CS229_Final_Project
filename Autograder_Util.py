import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_ref_dataset(file_path, char_list):

    with open(file_path, "r", encoding="utf-8") as f:
        raw_dataset = f.read()

    ref_chars = []
    for char in char_list:
        ref_stroke_dict = get_true_char(char, raw_dataset)

        # Convert dict to list of strokes (no substrokes)
        ref_strokes = [ref_stroke_dict[i] for i in sorted(ref_stroke_dict.keys())]
        ref_chars.append(ref_strokes)

    return ref_chars


def get_true_char(target_char, raw_dataset):

    lines = raw_dataset.strip().split("\n")

    for line in lines:
        if not line.strip():
            continue

        temp_data = json.loads(line)
        if temp_data.get("character") == target_char:

            orig = get_stroke_dict_from_json(line)
            return orig
    print(f"Character '{target_char}' not found in the dataset.")
    return None


def get_stroke_dict_from_json(json_line):

    data = json.loads(json_line)
    medians_list = data.get("medians", [])

    return {i: np.array(stroke_points) for i, stroke_points in enumerate(medians_list)}


def load_character_data(csv_path):
    """
    Load CSV data and convert to nested structure:
    chars[character_idx][stroke_idx] = array of points (x, y)
    """
    # Load the CSV - it's comma-separated, not tab-separated
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Get unique characters (in order of appearance)
    unique_chars = df["character"].unique()

    chars = []

    for char_name in unique_chars:
        char_df = df[df["character"] == char_name]
        unique_strokes = sorted(char_df["stroke"].unique())  # sort to ensure order
        strokes = []

        for stroke_idx in unique_strokes:
            stroke_df = char_df[char_df["stroke"] == stroke_idx]
            points = stroke_df[["x", "y"]].values
            strokes.append(points)
        chars.append(strokes)

    return chars, unique_chars


def resample_stroke(points, target_num=50):
    """Helper function to resample a single stroke."""
    if len(points) < 2:
        return points
    diff = np.diff(points, axis=0)
    dist = np.sqrt((diff**2).sum(axis=1))
    cumulative_dist = np.concatenate(([0], np.cumsum(dist)))
    target_dist = np.linspace(0, cumulative_dist[-1], target_num)
    resampled_x = np.interp(target_dist, cumulative_dist, points[:, 0])
    resampled_y = np.interp(target_dist, cumulative_dist, points[:, 1])
    return np.column_stack((resampled_x, resampled_y))


def resample_characters(chars, num_pts=50):
    """
    Resample all strokes in all characters to have num_pts points.

    Args:
        chars: list of chars, each char is a list of strokes (numpy arrays)
        num_pts: target number of points per stroke

    Returns:
        resampled_chars: same structure with resampled strokes
    """
    resampled_chars = []

    for char_strokes in chars:
        resampled_strokes = []
        for stroke in char_strokes:
            resampled = resample_stroke(stroke, target_num=num_pts)
            resampled_strokes.append(resampled)
        resampled_chars.append(resampled_strokes)

    return resampled_chars


def invert_y_axis(chars):
    """
    Invert y-axis for all characters.

    Args:
        chars: list of chars, each char is a list of strokes (numpy arrays)

    Returns:
        inverted_chars: same structure with inverted y coordinates
    """
    inverted_chars = []

    for char_strokes in chars:
        # Find global y bounds across all strokes in this character
        all_points = np.vstack(char_strokes)
        max_y = np.max(all_points[:, 1])
        min_y = np.min(all_points[:, 1])

        inverted_strokes = []
        for stroke in char_strokes:
            new_stroke = stroke.copy()
            new_stroke[:, 1] = max_y + min_y - new_stroke[:, 1]
            inverted_strokes.append(new_stroke)

        inverted_chars.append(inverted_strokes)

    return inverted_chars


import numpy as np


def normalize_character(chars, target_scale=1.0):

    normalized_chars = []

    for char_strokes in chars:
        # Find bounds across all strokes in this character
        all_points = np.vstack(char_strokes)
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)

        size = max_coords - min_coords
        max_dim = np.max(size)
        if max_dim == 0:
            max_dim = 1

        char_midpoint = min_coords + (size / 2)
        target_midpoint = np.array([target_scale / 2, target_scale / 2])

        normalized_strokes = []
        for stroke in char_strokes:
            centered_at_zero = stroke - char_midpoint
            norm_stroke = (centered_at_zero / max_dim * target_scale) + target_midpoint
            normalized_strokes.append(norm_stroke)

        normalized_chars.append(normalized_strokes)

    return normalized_chars


def preprocess_characters(chars, num_pts=50, target_scale=1.0, flip_y=False):
    processed = resample_characters(chars, num_pts=num_pts)
    processed = normalize_character(processed, target_scale=target_scale)

    if flip_y:
        processed = invert_y_axis(processed)

    return processed


def plot_character(strokes, flip=True, padding_ratio=0.1):
    """
    Plot a character from a list of strokes.

    Args:
        strokes: list of strokes, where each stroke is a numpy array of shape (n_points, 2)
        flip: whether to invert y-axis
        padding_ratio: padding around the character as ratio of max dimension
    """
    plt.figure(figsize=(7, 7))
    colors = plt.cm.jet(np.linspace(0, 1, len(strokes)))

    # Get all points to determine bounds
    all_points = np.vstack(strokes)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    # Plot each stroke
    for i, points in enumerate(strokes):
        plt.plot(
            points[:, 0],
            points[:, 1],
            color=colors[i],
            marker="o",
            markersize=3,
            label=f"Stroke {i}",
        )

    # Calculate bounds with padding
    width = x_max - x_min
    height = y_max - y_min
    max_dim = max(width, height)

    if max_dim == 0:
        max_dim = 1

    padding = max_dim * padding_ratio

    mid_x, mid_y = (x_max + x_min) / 2, (y_max + y_min) / 2
    half_size = (max_dim / 2) + padding

    plt.xlim(mid_x - half_size, mid_x + half_size)
    plt.ylim(mid_y - half_size, mid_y + half_size)

    plt.gca().set_aspect("equal", adjustable="box")
    if flip:
        plt.gca().invert_yaxis()

    plt.title(f"Character Visualization")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
