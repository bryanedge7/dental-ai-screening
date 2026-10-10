"""
train_stage2a.py - Stage 2A: Dental Pathology & Restoration Multi-Label Classifier
---------------------------------------------------------------------------------
Architecture: EfficientNet-B3 (PyTorch)
Input: Localized tooth patches (Regions of Interest / ROIs) extracted from preprocessed
       panoramic radiographs (OPG) using DENTEX clinical bounding box annotations.
Targets: Multi-label classification across 4 primary clinical targets:
         0: Caries (radiolucent enamel/dentin decay)
         1: Periapical Lesions (radiolucent rounded apical infections)
         2: Existing Crowns (radiopaque prosthetic coronal restorations)
         3: RCT Presence (radiopaque endodontic obturation material in root canals)

Optimized for lightweight, thread-efficient execution on AMD Ryzen CPU environments.
"""

import os
import sys

# Ensure UTF-8 output encoding across Windows consoles and background logs
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import math
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


# ---------------------------------------------------------------------------
# Global Constants & Target Definitions
# ---------------------------------------------------------------------------
PATHOLOGY_CLASSES = [
    "Caries",
    "Periapical_Lesion",
    "Existing_Crown",
    "RCT"
]
NUM_CLASSES = len(PATHOLOGY_CLASSES)


def parse_args():
    """Command-line argument parser with lightweight defaults for CPU execution."""
    parser = argparse.ArgumentParser(
        description="Stage 2A: EfficientNet-B3 Multi-Label Tooth Pathology Training"
    )
    parser.add_argument(
        "--annotations-json",
        type=str,
        default=os.path.join(
            "data", "raw", "training_data", "training_data",
            "quadrant-enumeration-disease", "train_quadrant_enumeration_disease.json"
        ),
        help="Path to DENTEX disease annotation JSON file."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=os.path.join("data", "processed", "images", "train"),
        help="Directory containing preprocessed panoramic X-ray images."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("runs", "stage2a_train"),
        help="Directory to save checkpoints, metrics, and logs."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5 for initial CPU baseline)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for DataLoader (default: 16)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate for AdamW optimizer (default: 5e-4)."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay for regularization (default: 0.01)."
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input resolution for tooth patches (default: 224x224)."
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.20,
        help="Proportion of panoramic scans reserved for validation (default: 0.20)."
    )
    parser.add_argument(
        "--roi-padding",
        type=float,
        default=0.10,
        help="Relative padding applied around tooth bounding boxes (default: 0.10)."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (default: 0 for Windows CPU stability)."
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=6,
        help="Number of intra-op CPU threads for PyTorch operations (default: 6 physical cores)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target execution device ('cpu' or 'cuda')."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 quick batch verification pass without full epoch training."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Clinical Label Extraction & Heuristics
# ---------------------------------------------------------------------------
def extract_multilabel_vector(ann, patch_bgr, quadrant_id):
    """
    Extracts a 4-dimensional binary ground-truth float vector:
    [Caries, Periapical_Lesion, Existing_Crown, RCT].

    1. Caries: Category ID 1 ('Caries') or 3 ('Deep Caries') in DENTEX annotations.
    2. Periapical Lesion: Category ID 2 ('Periapical Lesion') in DENTEX annotations.
    3. Existing Crown: Severe radio-opacity signature in coronal section (metal / PFM crown).
    4. RCT: Dense linear radio-opacity within root canal axis (gutta-percha obturation).
    """
    labels = np.zeros(NUM_CLASSES, dtype=np.float32)

    # 1. Caries & Deep Caries
    c3 = ann.get("category_id_3")
    if c3 in (1, 3):
        labels[0] = 1.0

    # 2. Periapical Lesion
    if c3 == 2:
        labels[1] = 1.0

    # 3 & 4. Radiopacity analysis for Crown & RCT Restorations
    if patch_bgr is not None and patch_bgr.size > 0:
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY) if len(patch_bgr.shape) == 3 else patch_bgr
        ph, pw = gray.shape

        # Quadrant 0/1 (Maxilla, FDI 1x/2x): roots point up (top), crowns point down (bottom)
        # Quadrant 2/3 (Mandible, FDI 3x/4x): roots point down (bottom), crowns point up (top)
        if quadrant_id in (0, 1):
            root_zone = gray[:int(ph * 0.5), :]
            crown_zone = gray[int(ph * 0.5):, :]
        else:
            crown_zone = gray[:int(ph * 0.5), :]
            root_zone = gray[int(ph * 0.5):, :]

        # Existing Crown: High radiopacity saturation covering > 8% of the coronal zone
        if crown_zone.size > 0:
            crown_radiopaque_ratio = np.mean(crown_zone >= 242)
            if crown_radiopaque_ratio > 0.08:
                labels[2] = 1.0

        # RCT: Gutta-percha obturation in radicular zone (> 4% high radiopacity along canal)
        if root_zone.size > 0:
            rct_radiopaque_ratio = np.mean(root_zone >= 238)
            if rct_radiopaque_ratio > 0.04:
                labels[3] = 1.0

    return labels


# ---------------------------------------------------------------------------
# Dataset & ROI Extraction
# ---------------------------------------------------------------------------
class DentexToothROIDataset(Dataset):
    """
    Dataset that holds pre-cropped, pre-resized tooth ROI patches in memory
    for zero-overhead CPU data loading.
    """
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.is_train = is_train

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pil_img = sample["pil_image"]
        labels = torch.from_numpy(sample["labels"])
        tensor_img = self.transform(pil_img)
        return tensor_img, labels


def load_and_extract_tooth_rois(
    annotations_json,
    images_dir,
    img_size=224,
    val_split=0.20,
    roi_padding=0.10,
    seed=42
):
    """
    Loads DENTEX disease annotations, extracts localized tooth patches,
    pre-resizes them with OpenCV once upfront, and performs patient-aware train/val splitting.
    """
    print(f"\n[Data Prep] Loading annotations from: {annotations_json}")
    if not os.path.exists(annotations_json):
        raise FileNotFoundError(f"Annotation file not found: {annotations_json}")

    with open(annotations_json, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    print(f"[Data Prep] Total images in JSON: {len(images):,}, total annotations: {len(annotations):,}")

    img_info_lookup = {img["id"]: img for img in images}

    # Group annotations by image_id to process each panoramic scan once
    anns_by_img = defaultdict(list)
    for ann in annotations:
        anns_by_img[ann["image_id"]].append(ann)

    # Filter to scans that exist in images_dir
    valid_img_ids = []
    for img_id in anns_by_img.keys():
        info = img_info_lookup.get(img_id)
        if not info:
            continue
        p = os.path.join(images_dir, info["file_name"])
        if os.path.exists(p):
            valid_img_ids.append(img_id)

    print(f"[Data Prep] Matched {len(valid_img_ids):,} panoramic scans available in '{images_dir}'")

    # Patient-isolated train/val split based on image_ids
    rng = np.random.RandomState(seed)
    shuffled_ids = rng.permutation(valid_img_ids)
    num_val = int(math.ceil(len(shuffled_ids) * val_split))
    val_set_ids = set(shuffled_ids[:num_val])
    train_set_ids = set(shuffled_ids[num_val:])

    print(f"[Data Prep] Image Partitioning: {len(train_set_ids)} Train scans, {len(val_set_ids)} Val scans")

    train_samples = []
    val_samples = []
    class_totals = np.zeros(NUM_CLASSES, dtype=np.int64)

    t0 = time.time()
    total_scans = len(valid_img_ids)
    processed_scans = 0

    print(f"[Data Prep] Extracting localized tooth ROI patches and pre-resizing to {img_size}x{img_size}...")

    for img_id in valid_img_ids:
        info = img_info_lookup[img_id]
        img_path = os.path.join(images_dir, info["file_name"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        ih, iw, _ = img_bgr.shape
        ow, oh = float(info["width"]), float(info["height"])
        scale_x = iw / ow
        scale_y = ih / oh

        is_val = img_id in val_set_ids
        target_list = val_samples if is_val else train_samples

        for ann in anns_by_img[img_id]:
            bx, by, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue

            x1 = bx * scale_x
            y1 = by * scale_y
            x2 = (bx + bw) * scale_x
            y2 = (by + bh) * scale_y

            # Apply proportional ROI padding
            pw = (x2 - x1) * roi_padding
            ph = (y2 - y1) * roi_padding
            px1 = max(0, int(x1 - pw))
            py1 = max(0, int(y1 - ph))
            px2 = min(iw, int(x2 + pw))
            py2 = min(ih, int(y2 + ph))

            if px2 <= px1 or py2 <= py1:
                continue

            patch = img_bgr[py1:py2, px1:px2]
            if patch.size == 0:
                continue

            q_id = ann.get("category_id_1", 0)
            labels = extract_multilabel_vector(ann, patch, q_id)
            class_totals += labels.astype(np.int64)

            # Pre-convert to RGB and pre-resize once with OpenCV
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch_resized = cv2.resize(patch_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
            pil_image = Image.fromarray(patch_resized)

            sample_item = {
                "pil_image": pil_image,
                "labels": labels,
                "tooth_id": ann.get("id"),
                "image_id": img_id,
            }
            target_list.append(sample_item)

        processed_scans += 1
        if processed_scans % 150 == 0 or processed_scans == total_scans:
            print(f"  Processed {processed_scans}/{total_scans} scans ({time.time() - t0:.1f}s)...")

    print(f"\n[Data Prep] ROI Extraction Complete in {time.time() - t0:.2f}s!")
    print(f"  Total Extracted Patches: {len(train_samples) + len(val_samples):,}")
    print(f"  - Training Patches:      {len(train_samples):,}")
    print(f"  - Validation Patches:    {len(val_samples):,}")
    print("\n[Data Prep] Pathology Distribution across extracted patches:")
    for i, cname in enumerate(PATHOLOGY_CLASSES):
        print(f"  - {cname:20s}: {class_totals[i]:,}")

    return train_samples, val_samples, class_totals


# ---------------------------------------------------------------------------
# EfficientNet-B3 Multi-Label Architecture
# ---------------------------------------------------------------------------
class EfficientNetB3MultiLabel(nn.Module):
    """
    EfficientNet-B3 model with a custom multi-label linear projection head
    for concurrent dental pathology and restoration classification.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dropout=0.30):
        super().__init__()
        if pretrained:
            try:
                print("[Model] Loading pre-trained EfficientNet-B3 weights (ImageNet-1K)...")
                weights = EfficientNet_B3_Weights.DEFAULT
                self.backbone = efficientnet_b3(weights=weights)
            except Exception as e:
                print(f"[Model] Pretrained weight download failed ({e}). Initializing random weights.")
                self.backbone = efficientnet_b3(weights=None)
        else:
            self.backbone = efficientnet_b3(weights=None)

        in_features = self.backbone.classifier[1].in_features  # 1536 for B3
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Metrics Evaluation
# ---------------------------------------------------------------------------
def compute_multilabel_metrics(y_true, y_logits, threshold=0.5):
    """
    Pure NumPy multi-label evaluation metrics:
    - Binary cross-entropy loss
    - Exact match (subset) accuracy
    - Hamming accuracy (mean binary label accuracy)
    - Macro & per-class Precision, Recall, and F1-score
    """
    probs = 1.0 / (1.0 + np.exp(-np.clip(y_logits, -30.0, 30.0)))
    preds = (probs >= threshold).astype(np.float32)

    # Subset exact match accuracy
    exact_match = np.mean(np.all(preds == y_true, axis=1))

    # Hamming accuracy
    hamming_acc = np.mean(preds == y_true)

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = preds[:, i]

        tp = np.sum((yp == 1) & (yt == 1))
        fp = np.sum((yp == 1) & (yt == 0))
        fn = np.sum((yp == 0) & (yt == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_precision.append(prec)
        per_class_recall.append(rec)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))
    macro_precision = float(np.mean(per_class_precision))
    macro_recall = float(np.mean(per_class_recall))

    return {
        "exact_match": float(exact_match),
        "hamming_acc": float(hamming_acc),
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def train_stage2a():
    args = parse_args()

    # Configure hardware thread limits for optimal Ryzen CPU execution
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cpu" and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(max(1, args.cpu_threads // 2))

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"\n{'=' * 65}")
    print("      DENTAL AI SCREENING -- STAGE 2A PATHOLOGY TRAINING")
    print(f"{'=' * 65}")
    print(f"  Target Architecture: EfficientNet-B3")
    print(f"  Compute Device:      {device} (Threads: {torch.get_num_threads()})")
    print(f"  Input Resolution:    {args.img_size}x{args.img_size}")
    print(f"  Batch Size:          {args.batch_size}")
    print(f"  Target Pathologies:  {', '.join(PATHOLOGY_CLASSES)}")
    print(f"  Epochs:              {args.epochs}")
    print(f"  Output Directory:    {args.output_dir}")
    print(f"{'=' * 65}")

    # Create run output directory
    weights_dir = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 1. Load and extract tooth ROI patches
    train_samples, val_samples, class_totals = load_and_extract_tooth_rois(
        annotations_json=args.annotations_json,
        images_dir=args.images_dir,
        img_size=args.img_size,
        val_split=args.val_split,
        roi_padding=args.roi_padding,
        seed=args.seed
    )

    train_dataset = DentexToothROIDataset(train_samples, is_train=True)
    val_dataset = DentexToothROIDataset(val_samples, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(args.batch_size, 32),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 2. Build EfficientNet-B3 Model
    model = EfficientNetB3MultiLabel(num_classes=NUM_CLASSES, pretrained=True, dropout=0.30)
    model.to(device)

    # 3. Calculate positive weights for multi-label BCE loss to handle class imbalance
    num_train = max(1, len(train_samples))
    pos_counts = np.sum([s["labels"] for s in train_samples], axis=0) if train_samples else np.ones(NUM_CLASSES)
    neg_counts = num_train - pos_counts
    # Clamp pos_weight between 1.0 and 4.0 for stable CPU convergence
    pos_weights = np.clip(neg_counts / np.maximum(pos_counts, 1.0), 1.0, 4.0)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    print(f"\n[Loss] BCE Pos Weights: {dict(zip(PATHOLOGY_CLASSES, np.round(pos_weights, 2)))}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)

    # 4. Dry Run check if requested
    if args.dry_run:
        print("\n[Dry Run] Executing single-batch verification step...")
        model.eval()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            print(f"[Dry Run] Forward pass successful! Output: {out.shape}, Loss: {loss.item():.4f}")
            break
        print("Dry run completed successfully.")
        return

    # 5. Training Loop
    results_csv = os.path.join(args.output_dir, "results.csv")
    with open(results_csv, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,val_loss,macro_f1,macro_prec,macro_rec,hamming_acc,exact_match,"
            + ",".join([f"f1_{c}" for c in PATHOLOGY_CLASSES])
            + "\n"
        )

    best_macro_f1 = -1.0
    print(f"\n{'=' * 65}")
    print("                STARTING STAGE 2A TRAINING LOOP")
    print(f"{'=' * 65}")

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        # Training Phase
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            train_batches += 1

            if batch_idx % 25 == 0 or batch_idx == len(train_loader):
                cur_avg = train_loss_accum / train_batches
                sys.stdout.write(
                    f"\rEpoch [{epoch:02d}/{args.epochs:02d}] Batch {batch_idx:03d}/{len(train_loader):03d} "
                    f"- Train Loss: {cur_avg:.4f}"
                )
                sys.stdout.flush()

        scheduler.step()
        train_avg_loss = train_loss_accum / max(1, train_batches)
        print()

        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        all_targets = []
        all_logits = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                logits = model(images)
                loss = criterion(logits, targets)

                val_loss_accum += loss.item()
                val_batches += 1

                all_targets.append(targets.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        val_avg_loss = val_loss_accum / max(1, val_batches)
        y_true_val = np.concatenate(all_targets, axis=0)
        y_logits_val = np.concatenate(all_logits, axis=0)

        metrics = compute_multilabel_metrics(y_true_val, y_logits_val, threshold=0.5)
        epoch_sec = time.time() - epoch_t0

        # Safe ASCII Console Summary
        print(
            f"Epoch {epoch:02d} Summary ({epoch_sec:.1f}s) | "
            f"Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | Hamming Acc: {metrics['hamming_acc']:.4f}"
        )
        for i, cname in enumerate(PATHOLOGY_CLASSES):
            print(
                f"   |-- {cname:18s} -> F1: {metrics['per_class_f1'][i]:.4f} "
                f"(Prec: {metrics['per_class_precision'][i]:.4f}, Rec: {metrics['per_class_recall'][i]:.4f})"
            )

        # Append to CSV
        with open(results_csv, "a", encoding="utf-8") as f:
            f1_str = ",".join([f"{v:.4f}" for v in metrics["per_class_f1"]])
            f.write(
                f"{epoch},{train_avg_loss:.4f},{val_avg_loss:.4f},"
                f"{metrics['macro_f1']:.4f},{metrics['macro_precision']:.4f},{metrics['macro_recall']:.4f},"
                f"{metrics['hamming_acc']:.4f},{metrics['exact_match']:.4f},{f1_str}\n"
            )

        # Checkpointing
        last_ckpt = os.path.join(weights_dir, "last_stage2a.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "classes": PATHOLOGY_CLASSES,
            },
            last_ckpt
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_ckpt = os.path.join(weights_dir, "best_stage2a.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "classes": PATHOLOGY_CLASSES,
                },
                best_ckpt
            )
            print(f"   [BEST] Model checkpoint saved to: {best_ckpt} (Macro F1: {best_macro_f1:.4f})")

    print(f"\n{'=' * 65}")
    print("[COMPLETE] STAGE 2A MULTI-LABEL CLASSIFICATION TRAINING COMPLETE!")
    print(f"  Best Validation Macro F1: {best_macro_f1:.4f}")
    print(f"  Results saved to:         {results_csv}")
    print(f"  Best weights:             {os.path.join(weights_dir, 'best_stage2a.pt')}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    train_stage2a()
