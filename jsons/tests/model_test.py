import json
import os
import time

import numpy as np
import tensorflow as tf
from pathlib import Path
from movenet_pre_processor.normalizer import normalize_all_frames
from movenet_pre_processor.pre_processor import build_sequence
from jsons.tools.custom_layers import SqueezeLayer


# === Load maps once ===
def load_maps(label_map_path, angle_map_path):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    with open(angle_map_path, "r") as f:
        angle_map = json.load(f)
    # Invert label map for decoding
    idx_to_label = {v: k for k, v in label_map.items()}
    idx_to_angle = {v: k for k, v in angle_map.items()}
    return label_map, angle_map, idx_to_label, idx_to_angle


# === Inference function ===
def predict_throw(
        model_path: str,
        json_path: str,
        angle_label: str,
        label_map_path: str = "../dataset/label_map.json",
        angle_map_path: str = "../dataset/angle_map.json"
):
    # Load model and maps
    model = tf.keras.models.load_model(model_path, custom_objects={"SqueezeLayer": SqueezeLayer})
    label_map, angle_map, idx_to_label, _ = load_maps(label_map_path, angle_map_path)

    # Load and preprocess clip
    with open(json_path, "r") as f:
        raw_frames = json.load(f)

    normalized = normalize_all_frames(raw_frames)
    sequence = build_sequence(
        frames=normalized,
        max_people=2,
        target_len=60,
        fill_mode="last"
    )

    # Prepare model inputs
    X = np.expand_dims(sequence, axis=0)  # shape: (1, 60, 68)
    angle_index = np.array([[angle_map[angle_label]]])  # shape: (1, 1)

    # Run prediction
    probs = model.predict({"pose_input": X, "angle_input": angle_index})
    predicted_class = int(np.argmax(probs))

    return {
        "predicted_label": idx_to_label[predicted_class],
        "confidence": float(np.max(probs)),
        "all_probs": probs.tolist()[0],
        "class_index": predicted_class
    }


result = predict_throw(
    model_path="../models/throw_detection_v0-1.keras",
    json_path="test_data/clip-003(4).json",
    angle_label="side"
)
print("🎯 Predicted throw:", result["predicted_label"])
print("🔢 Confidence:", f"{result['confidence']:.2%}")
print("📊 Raw class probabilities:", result["all_probs"])

'''for file in os.listdir("test_data"):
    file_path = os.path.join("test_data", file)
    result = predict_throw(
        model_path="../models/lstm_with_angle-balanced-weights-epoch-2000.keras",
        json_path=file_path,
        angle_label="side"
    )
    print("\n\n" + str(file))
    print("🎯 Predicted throw:", result["predicted_label"])
    print("🔢 Confidence:", f"{result['confidence']:.2%}")
    print("📊 Raw class probabilities:", result["all_probs"])
    time.sleep(1)'''








'''
import numpy as np
import json

y = np.load("../dataset/complete_dataset.npz")["y"]
with open("../dataset/label_map.json") as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}
counts = np.bincount(y)

for idx, count in enumerate(counts):
    print(f"{idx_to_label[idx]}: {count}")
    
'''