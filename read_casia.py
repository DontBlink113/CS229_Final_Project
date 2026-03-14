
import struct
import matplotlib.pyplot as plt
import os
import struct


def read_casia_pot(filepath):
    samples = []
    with open(filepath, 'rb') as f:
        while True:
            if len(f.read(2)) < 2:  # sample size
                break
            f.read(4)               # tag code (skip)
            stroke_count_bytes = f.read(2)
            if len(stroke_count_bytes) < 2:
                break

            strokes = []
            current_stroke = []

            while True:
                xy = f.read(4)
                if len(xy) < 4:
                    break
                x, y = struct.unpack('<hh', xy)

                if x == -1 and y == -1:
                    if current_stroke:
                        strokes.append(current_stroke)
                    break
                elif x == -1 and y == 0:
                    if current_stroke:
                        strokes.append(current_stroke)
                    current_stroke = []
                else:
                    current_stroke.append((x, y))

            samples.append({'strokes': strokes})
    return samples


def display_pot_samples(samples, max_samples=9):
    n = min(max_samples, len(samples))
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3.5))
    axes = axes.flatten()
    for i, sample in enumerate(samples[:n]):
        ax = axes[i]
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        for stroke in sample:
            if len(stroke) >= 2:
                xs, ys = zip(*stroke)
                ax.plot(xs, ys, 'b-', linewidth=1.5)
    for j in range(n, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()



def extract_samples_by_user(folder_path):
    """
    Returns a list of lists, where each inner list contains
    all samples (strokes) for one user/file.
    """
    all_users = []

    pot_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith('.pot')
    ])

    for filename in pot_files:
        filepath = os.path.join(folder_path, filename)
        samples = read_casia_pot(filepath)

        # Each user's samples: list of stroke-sets
        user_samples = [sample['strokes'] for sample in samples]
        all_users.append(user_samples)
        print(f"file {filename} loaded")

    return all_users



def extract_chars_dict(folder_path):
    """
    Returns a dict where:
        key   = character tag (e.g. 'A', '你', etc.)
        value = list of all handwritten versions (each a list of strokes)
    """
    chars_dict = {}
    pot_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith('.pot')
    ])
    for filename in pot_files:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'rb') as f:
            while True:
                sample_start = f.tell()
                size_bytes = f.read(2)
                if len(size_bytes) < 2:
                    break
                sample_size = struct.unpack('<H', size_bytes)[0]

                tag_bytes = f.read(4)
                if len(tag_bytes) < 4:
                    break
                tag = bytes([tag_bytes[1], tag_bytes[0]]).decode('gbk', errors='replace').strip('\x00')

                stroke_count_bytes = f.read(2)
                if len(stroke_count_bytes) < 2:
                    break

                strokes = []
                current_stroke = []
                while True:
                    xy = f.read(4)
                    if len(xy) < 4:
                        break
                    x, y = struct.unpack('<hh', xy)
                    if x == -1 and y == -1:
                        if current_stroke:
                            strokes.append(current_stroke)
                        break
                    elif x == -1 and y == 0:
                        if current_stroke:
                            strokes.append(current_stroke)
                        current_stroke = []
                    else:
                        current_stroke.append((x, y))

                if tag not in chars_dict:
                    chars_dict[tag] = []
                chars_dict[tag].append(strokes)

                f.seek(sample_start + sample_size)

        print(f"loaded {filepath}")
    return chars_dict

def merge_dict(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged:
            merged[key].extend(value)
        else:
            merged[key] = value
    return merged
