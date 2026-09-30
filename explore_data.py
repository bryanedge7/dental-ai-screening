import os
import json


def explore_training_annotations(base_folder=os.path.join("data", "raw", "training_data")):
    """
    Search data/raw/training_data for the first available JSON annotation file
    and inspect its top-level keys and structure.
    """
    print(f"Searching for JSON annotation files in: {base_folder}")

    if not os.path.exists(base_folder):
        print(f"Error: Folder '{base_folder}' does not exist.")
        return

    found_json = None

    # Walk through the folder and subdirectories to locate the first JSON file
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(".json"):
                found_json = os.path.join(root, file)
                break
        if found_json:
            break

    if not found_json:
        print("No JSON annotation files found in the specified folder.")
        return

    print(f"\nFound annotation file: {os.path.basename(found_json)}")
    print(f"Full path: {found_json}\n")

    # Load and inspect the JSON file
    with open(found_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Top-level keys ({len(keys)}): {keys}\n")
        print("Structure overview:")
        for key in keys:
            val = data[key]
            if isinstance(val, list):
                sample_item = val[0] if val else None
                sample_info = f" (sample item fields: {list(sample_item.keys())})" if isinstance(sample_item, dict) else ""
                print(f"  - '{key}': list with {len(val):,} items{sample_info}")
            elif isinstance(val, dict):
                print(f"  - '{key}': dict with keys {list(val.keys())}")
            else:
                print(f"  - '{key}': {type(val).__name__} = {val}")
    elif isinstance(data, list):
        print(f"Root object is a list of {len(data)} items.")
        if data and isinstance(data[0], dict):
            print(f"Sample item keys: {list(data[0].keys())}")
    else:
        print(f"Root object type: {type(data).__name__}")


if __name__ == "__main__":
    explore_training_annotations()
