import os
import json
import re


def label_jsons():
    vid_labels = "labels/vid-labels.json"
    files = os.listdir("training")
    result_list = []  # List to store the final output format

    try:
        with open(vid_labels, 'r') as file:
            vid_label_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found: {vid_labels}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in: {vid_labels}")
        return None

    # Extract the clip identifiers from vid_label_data and create a mapping
    clip_map = {}
    for entry in vid_label_data:
        # Extract clip_XXX part from the vid-label field (assumes format like "clip_001.mov")
        clip_id = entry["vid-label"].split(".")[0]  # Get "clip_001" part
        clip_map[clip_id] = entry

    # Process each JSON file in the training directory
    json_files = [f for f in files if f.endswith(".json")]

    # Custom sorting function to handle filenames like 'clip-001.json', 'clip-001(1).json', etc.
    def sort_key(filename):
        # Extract the base number (e.g., '001' from 'clip-001.json')
        base_match = re.search(r'clip-(\d+)', filename)
        base_num = int(base_match.group(1)) if base_match else 999999

        # Extract the suffix number if it exists (e.g., '1' from 'clip-001(1).json')
        suffix_match = re.search(r'\((\d+)\)', filename)
        suffix_num = int(suffix_match.group(1)) if suffix_match else 0

        # Return a tuple for sorting: (base_number, suffix_number)
        return (base_num, suffix_num)

    # Sort the files based on our custom sorting function
    sorted_files = sorted(json_files, key=sort_key)

    # Now process the sorted files
    for filename in sorted_files:
        # Create a new entry for this file
        file_entry = {
            "file": filename,
            "label": "",
            "angle": ""
        }

        # Extract clip name pattern (like "clip-001" from "clip-001.json" or "clip-001(1).json")
        match = re.match(r"(clip-\d+)", filename)
        if match:
            clip_name = match.group(1)

            # Convert "clip-001" to "clip_001" format to match vid_labels.json
            clip_id = clip_name.replace("-", "_")

            # If we have label data for this clip, fill in the label and angle
            if clip_id in clip_map:
                file_entry["label"] = clip_map[clip_id]["label"]
                file_entry["angle"] = clip_map[clip_id]["angle"]

        # Add this entry to our result list
        result_list.append(file_entry)

    # Write the organized data to a new labels.json file
    with open("labels.json", 'w') as outfile:
        json.dump(result_list, outfile, indent=4)

    print("Successfully created labels.json with organized label data.")
    return result_list


if __name__ == "__main__":
    label_jsons()










