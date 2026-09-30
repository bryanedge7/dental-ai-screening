import os
import json
import shutil
import argparse
from pathlib import Path


def find_json_files(raw_dir):
    """Scan directory recursively for all DENTEX JSON annotation files."""
    json_files = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(".json"):
                json_files.append(os.path.join(root, f))
    return sorted(json_files)


def resolve_category_mapping(data):
    """
    Resolve category IDs and names from COCO-style dictionaries.
    Supports standard COCO ('categories') and DENTEX multi-task ('categories_1', 'categories_2', 'categories_3').
    """
    if "categories" in data and data["categories"]:
        cat_list = data["categories"]
        id_to_name = {c["id"]: str(c["name"]) for c in cat_list}
        sorted_ids = sorted(id_to_name.keys())
        orig_to_yolo = {orig_id: idx for idx, orig_id in enumerate(sorted_ids)}
        names = {orig_to_yolo[orig_id]: id_to_name[orig_id] for orig_id in sorted_ids}

        def get_cat_id(ann):
            orig = ann.get("category_id")
            return orig_to_yolo.get(orig)

        return get_cat_id, names

    elif "categories_3" in data and data["categories_3"]:
        # Disease categories (Impacted, Caries, Periapical Lesion, Deep Caries)
        cat_list = data["categories_3"]
        id_to_name = {c["id"]: str(c["name"]) for c in cat_list}
        sorted_ids = sorted(id_to_name.keys())
        orig_to_yolo = {orig_id: idx for idx, orig_id in enumerate(sorted_ids)}
        names = {orig_to_yolo[orig_id]: id_to_name[orig_id] for orig_id in sorted_ids}

        def get_cat_id(ann):
            orig = ann.get("category_id_3")
            return orig_to_yolo.get(orig)

        return get_cat_id, names

    elif "categories_2" in data and data["categories_2"]:
        # Tooth enumeration categories (1 through 8)
        cat_list = data["categories_2"]
        id_to_name = {c["id"]: str(c["name"]) for c in cat_list}
        sorted_ids = sorted(id_to_name.keys())
        orig_to_yolo = {orig_id: idx for idx, orig_id in enumerate(sorted_ids)}
        names = {orig_to_yolo[orig_id]: id_to_name[orig_id] for orig_id in sorted_ids}

        def get_cat_id(ann):
            orig = ann.get("category_id_2")
            return orig_to_yolo.get(orig)

        return get_cat_id, names

    else:
        # Fallback for generic datasets
        def get_cat_id(ann):
            return ann.get("category_id", 0)

        return get_cat_id, {0: "object"}


def convert_coco_to_yolo(
    json_path,
    output_labels_dir=os.path.join("data", "processed", "labels", "train"),
    output_images_dir=os.path.join("data", "processed", "images", "train"),
    raw_images_dir=None,
    copy_images=True,
):
    """
    Convert a COCO instance segmentation JSON file to YOLOv8-seg text format.
    """
    print(f"Loading annotation file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    print(f"Loaded {len(images):,} images and {len(annotations):,} annotations.")

    get_cat_id, class_names = resolve_category_mapping(coco_data)
    print(f"Detected classes ({len(class_names)}): {class_names}")

    # Create target directories
    os.makedirs(output_labels_dir, exist_ok=True)
    if copy_images:
        os.makedirs(output_images_dir, exist_ok=True)

    # Determine candidate source directories for X-ray images
    possible_xray_dirs = []
    json_parent = os.path.dirname(json_path)
    possible_xray_dirs.append(os.path.join(json_parent, "xrays"))
    possible_xray_dirs.append(os.path.join(json_parent, "images"))
    if raw_images_dir:
        possible_xray_dirs.insert(0, raw_images_dir)

    # Build image lookup table: image_id -> image_dict
    image_dict = {img["id"]: img for img in images}

    # Group annotations by image_id
    ann_by_image = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        ann_by_image.setdefault(img_id, []).append(ann)

    # Conversion counters
    total_images_processed = 0
    total_polygons_saved = 0
    class_counts = {cid: 0 for cid in class_names.keys()}
    images_copied = 0
    images_skipped = 0
    images_missing = 0

    print("\nConverting annotations and processing images...")

    for img_id, img_info in image_dict.items():
        file_name = img_info["file_name"]
        stem = Path(file_name).stem
        img_w = float(img_info.get("width", 0))
        img_h = float(img_info.get("height", 0))

        if img_w <= 0 or img_h <= 0:
            print(f"Warning: Invalid dimensions for image {file_name}: {img_w}x{img_h}")
            continue

        label_file = os.path.join(output_labels_dir, f"{stem}.txt")
        yolo_lines = []

        img_annotations = ann_by_image.get(img_id, [])
        for ann in img_annotations:
            cat_id = get_cat_id(ann)
            if cat_id is None:
                continue

            segmentation = ann.get("segmentation", [])
            # COCO segmentation is a list of polygons: [[x1, y1, x2, y2, ...]]
            if isinstance(segmentation, list):
                for poly in segmentation:
                    if not isinstance(poly, (list, tuple)) or len(poly) < 6:
                        # Skip if fewer than 3 points (6 coords)
                        continue

                    normalized_coords = []
                    for i in range(0, len(poly), 2):
                        px = min(max(float(poly[i]) / img_w, 0.0), 1.0)
                        py = min(max(float(poly[i + 1]) / img_h, 0.0), 1.0)
                        normalized_coords.extend([f"{px:.6f}", f"{py:.6f}"])

                    yolo_line = f"{cat_id} " + " ".join(normalized_coords)
                    yolo_lines.append(yolo_line)
                    total_polygons_saved += 1
                    class_counts[cat_id] = class_counts.get(cat_id, 0) + 1

        # Write YOLO label file (even if empty, representing background)
        with open(label_file, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        # Copy matching X-ray image if requested
        if copy_images:
            src_image_path = None
            for pdir in possible_xray_dirs:
                candidate = os.path.join(pdir, file_name)
                if os.path.exists(candidate):
                    src_image_path = candidate
                    break

            if src_image_path:
                dest_image_path = os.path.join(output_images_dir, file_name)
                # Avoid re-copying if identical file already exists
                if not os.path.exists(dest_image_path) or os.path.getsize(dest_image_path) != os.path.getsize(src_image_path):
                    shutil.copy2(src_image_path, dest_image_path)
                    images_copied += 1
                else:
                    images_skipped += 1
            else:
                images_missing += 1

        total_images_processed += 1
        if total_images_processed % 150 == 0 or total_images_processed == len(images):
            print(f"  Processed {total_images_processed}/{len(images)} images...")

    # Write data.yaml for YOLOv8 segmentation training
    processed_root = Path(output_labels_dir).parents[1]
    yaml_path = os.path.join(processed_root, "data.yaml")
    yaml_content = f"""# YOLOv8 Instance Segmentation Dataset Configuration
path: {Path(processed_root).resolve().as_posix()}
train: images/train
val: images/train  # Update with images/val once validation split is prepared

names:
"""
    for cid, cname in sorted(class_names.items()):
        yaml_content += f"  {cid}: \"{cname}\"\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\n" + "=" * 55)
    print("           CONVERSION SUMMARY")
    print("=" * 55)
    print(f"Images Processed:     {total_images_processed:,}")
    print(f"Polygons Converted:   {total_polygons_saved:,}")
    print(f"Labels Output:        {output_labels_dir}")
    if copy_images:
        print(f"Images Copied:        {images_copied:,} (Skipped existing: {images_skipped:,}, Missing: {images_missing})")
        print(f"Images Output:        {output_images_dir}")
    print(f"YOLO Config Created:  {yaml_path}")
    print("\nClass distribution:")
    for cid, count in sorted(class_counts.items()):
        print(f"  - Class {cid} ({class_names.get(cid, 'Unknown')}): {count:,} instances")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Convert COCO instance segmentation annotations to YOLOv8-seg format.")
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to specific COCO JSON file. Defaults to the first JSON found or chosen by --task.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="quadrant",
        choices=["quadrant", "disease", "enumeration"],
        help="Target DENTEX task to convert (quadrant, disease, enumeration). Default is quadrant.",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=os.path.join("data", "raw", "training_data"),
        help="Root folder containing raw dataset.",
    )
    parser.add_argument(
        "--output_labels",
        type=str,
        default=os.path.join("data", "processed", "labels", "train"),
        help="Directory to save YOLO .txt labels.",
    )
    parser.add_argument(
        "--output_images",
        type=str,
        default=os.path.join("data", "processed", "images", "train"),
        help="Directory to copy X-ray images.",
    )
    parser.add_argument(
        "--no_copy_images",
        action="store_true",
        help="Skip copying image files.",
    )
    args = parser.parse_args()

    # Determine target JSON file
    target_json = args.json_path
    if not target_json:
        available_jsons = find_json_files(args.raw_dir)
        if not available_jsons:
            print(f"Error: No JSON files found in {args.raw_dir}")
            return

        task_matches = {
            "quadrant": "train_quadrant.json",
            "disease": "train_quadrant_enumeration_disease.json",
            "enumeration": "train_quadrant_enumeration.json",
        }
        target_filename = task_matches.get(args.task, "train_quadrant.json")

        for j in available_jsons:
            if os.path.basename(j).lower() == target_filename.lower():
                target_json = j
                break

        if not target_json:
            # Fall back to first available JSON
            target_json = available_jsons[0]

    convert_coco_to_yolo(
        json_path=target_json,
        output_labels_dir=args.output_labels,
        output_images_dir=args.output_images,
        copy_images=not args.no_copy_images,
    )


if __name__ == "__main__":
    main()
