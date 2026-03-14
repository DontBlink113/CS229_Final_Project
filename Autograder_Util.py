import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# def load_ref_dataset(file_path, char_list):

#     with open(file_path, "r", encoding="utf-8") as f:
#         raw_dataset = f.read()

#     ref_chars = []
#     for char in char_list:
#         ref_stroke_dict = get_true_char(char, raw_dataset)

#         # Convert dict to list of strokes (no substrokes)
#         ref_strokes = [ref_stroke_dict[i] for i in sorted(ref_stroke_dict.keys())]
#         ref_chars.append(ref_strokes)
#         chars.append(char)

#     return ref_chars


#loads the graphics.txt file as a dictionary of characters and their median strokes
#inverts the y-axis
#returns a dictionary with the character as the key and the points as the value
#PRIMARY WAY TO LOAD graphics.txt
def parse_medians(filepath, num_entries=10000):
    results = {}
    char_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            results[obj['character']] = [np.array(stroke) for stroke in obj['medians']]  # convert here
            char_list.append(obj['character'])
            if len(results) >= num_entries:
                break
    return results, char_list


def load_character_data(csv_path):

    df = pd.read_csv(csv_path, encoding="utf-8")

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
    if len(points) < 2:
        return points
    diff = np.diff(points, axis=0)
    dist = np.sqrt((diff**2).sum(axis=1))
    cumulative_dist = np.concatenate(([0], np.cumsum(dist)))
    target_dist = np.linspace(0, cumulative_dist[-1], target_num)
    resampled_x = np.interp(target_dist, cumulative_dist, points[:, 0])
    resampled_y = np.interp(target_dist, cumulative_dist, points[:, 1])
    return np.column_stack((resampled_x, resampled_y))


def resample_characters(chars_dict, num_pts=50):
    return {
        char: [resample_stroke(stroke, target_num=num_pts) for stroke in strokes]
        for char, strokes in chars_dict.items()
    }

def invert_y_axis(chars_dict):
    inverted = {}
    for char, strokes in chars_dict.items():
        all_points = np.vstack(strokes)
        max_y, min_y = np.max(all_points[:, 1]), np.min(all_points[:, 1])
        inverted[char] = [
            (lambda s: (s.__setitem__(slice(None), s.copy()) or s.copy()))(stroke.copy())
            for stroke in strokes
        ]
        for stroke in inverted[char]:
            stroke[:, 1] = max_y + min_y - stroke[:, 1]
    return inverted

def normalize_characters(chars_dict, target_scale=1.0):
    normalized = {}
    for char, strokes in chars_dict.items():
        all_points = np.vstack(strokes)
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        size = max_coords - min_coords
        max_dim = np.max(size) or 1
        char_midpoint = min_coords + (size / 2)
        target_midpoint = np.array([target_scale / 2, target_scale / 2])
        normalized[char] = [
            (stroke - char_midpoint) / max_dim * target_scale + target_midpoint
            for stroke in strokes
        ]
    return normalized

def preprocess_characters(chars_dict, num_pts=50, target_scale=1.0, flip_y=False):
    processed = resample_characters(chars_dict, num_pts=num_pts)
    processed = normalize_characters(processed, target_scale=target_scale)
    if flip_y:
        processed = invert_y_axis(processed)
    return processed


def plot_character(strokes, flip=False, padding_ratio=0.1):

    plt.figure(figsize=(7, 7))
    colors = plt.cm.jet(np.linspace(0, 1, len(strokes)))

    # Get all points to determine bounds
    all_points = np.vstack(strokes)
    x_min, y_min = (0, 0) #this is hard-coded 
    x_max, y_max = (1, 1) #^^ this too 

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


def load_dictionary_characters(filepath):
    chars = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    chars[obj['character']] = obj
                except json.JSONDecodeError:
                    pass  # skip malformed lines
    return chars

