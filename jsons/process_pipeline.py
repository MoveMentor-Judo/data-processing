import json
import numpy as np
from pathlib import Path
from movenet_pre_processor.normalizer import normalize_all_frames
from movenet_pre_processor.pre_processor import build_sequence


def build_label_maps(metadata):
    labels = sorted(set(item["label"] for item in metadata))
    angles = sorted(set(item["angle"] for item in metadata))
    return {label: i for i, label in enumerate(labels)}, {angle: i for i, angle in enumerate(angles)}


def build_dataset(labels_path, clip_dir, output_npz_path, save_maps=True):
    with open(labels_path, "r") as f:
        metadata = json.load(f)

    label_map, angle_map = build_label_maps(metadata)
    X, y, a = [], [], []

    for item in metadata:
        path = Path(clip_dir) / item["file"]
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        with open(path, "r") as f:
            raw_frames = json.load(f)

        try:
            normalized = normalize_all_frames(raw_frames)
            sequence = build_sequence(normalized, max_people=2, target_len=60, fill_mode="last")

            X.append(sequence)
            y.append(label_map[item["label"]])
            a.append(angle_map[item["angle"]])
        except Exception as e:
            print(f"Error processing {path.name}: {e}")

    X = np.stack(X)
    y = np.array(y)
    a = np.array(a)
    np.savez(output_npz_path, X=X, y=y, angles=a)

    if save_maps:
        with open(Path(output_npz_path).with_name("label_map.json"), "w") as f:
            json.dump(label_map, f, indent=2)
        with open(Path(output_npz_path).with_name("angle_map.json"), "w") as f:
            json.dump(angle_map, f, indent=2)

    print(f"✅ Saved dataset to {output_npz_path}")
    print(f"   X: {X.shape}, y: {y.shape}, angles: {a.shape}")


build_dataset(
    labels_path="labels/labels.json",
    clip_dir="training",
    output_npz_path="dataset/dataset-v0-1.npz"
)