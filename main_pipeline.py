"""
main_pipeline.py - Master Orchestration Pipeline for Dental AI Screening
-------------------------------------------------------------------------
Combines the multi-stage vision and diagnostic architecture into a unified workflow:
  1. Stage 1: YOLOv8 Quadrant Instance Segmentation & FDI Missing Teeth Gap Analysis
  2. Stage 2A: EfficientNet-B3 Multi-Label Tooth Pathology & Restoration Classifier
               (Caries, Periapical Lesions, Existing Crowns, and RCT Presence)
  3. Stage 2B: U-Net Anatomical Landmark Regression & Section 6 Implant Site Assessment
               (Maxillary Sinus Clearance & Mandibular Nerve Canal 2mm Safety Buffer)
  4. Clinical Report Generator: Structured JSON Findings and Treatment Planning Engine
"""

import os
import sys

# Ensure UTF-8 output encoding across Windows shells and background log pipes
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO

# Import Stage 2 architectures
from train_stage2a import (
    EfficientNetB3MultiLabel,
    PATHOLOGY_CLASSES,
    NUM_CLASSES as NUM_PATHOLOGY_CLASSES
)
from train_stage2b_landmark import (
    LandmarkUNetRegression,
    evaluate_upper_jaw_site,
    evaluate_lower_jaw_site,
    LANDMARK_CHANNELS,
    DEFAULT_PIXEL_SPACING_MM
)
from format_output import (
    format_clean_report,
    generate_call_center_text_briefing,
    MANDATORY_LEGAL_DISCLAIMER
)

# ---------------------------------------------------------------------------
# Global Constants & FDI Dentition Definitions
# ---------------------------------------------------------------------------
STANDARD_ADULT_FDI_TEETH = {
    1: [18, 17, 16, 15, 14, 13, 12, 11],  # Quadrant 1: Upper Right
    2: [21, 22, 23, 24, 25, 26, 27, 28],  # Quadrant 2: Upper Left
    3: [31, 32, 33, 34, 35, 36, 37, 38],  # Quadrant 3: Lower Left
    4: [41, 42, 43, 44, 45, 46, 47, 48],  # Quadrant 4: Lower Right
}
ALL_32_FDI_NUMBERS = (
        STANDARD_ADULT_FDI_TEETH[1] +
        STANDARD_ADULT_FDI_TEETH[2] +
        STANDARD_ADULT_FDI_TEETH[3] +
        STANDARD_ADULT_FDI_TEETH[4]
)

QUADRANT_NAMES = {
    0: "Quadrant 1 (Maxillary Right)",
    1: "Quadrant 2 (Maxillary Left)",
    2: "Quadrant 3 (Mandibular Left)",
    3: "Quadrant 4 (Mandibular Right)",
}

TOOTH_NAMES_MAP = {
    1: "Central Incisor",
    2: "Lateral Incisor",
    3: "Canine",
    4: "First Premolar",
    5: "Second Premolar",
    6: "First Molar",
    7: "Second Molar",
    8: "Third Molar (Wisdom Tooth)"
}


# ---------------------------------------------------------------------------
# Helper: FDI Tooth Name Resolver
# ---------------------------------------------------------------------------
def get_fdi_description(fdi_num: int) -> str:
    """Returns anatomical description for standard FDI number (e.g. 16 -> Upper Right First Molar)."""
    quad = fdi_num // 10
    pos = fdi_num % 10
    quad_desc = {
        1: "Upper Right",
        2: "Upper Left",
        3: "Lower Left",
        4: "Lower Right"
    }.get(quad, "Unknown")
    pos_desc = TOOTH_NAMES_MAP.get(pos, "Tooth")
    return f"{quad_desc} {pos_desc}"


# ---------------------------------------------------------------------------
# Stage 1: Quadrant Localization & FDI Inventory Manager
# ---------------------------------------------------------------------------
class Stage1Detector:
    """Encapsulates YOLOv8 instance segmentation and FDI tooth inventory mapping."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = device
        self.weights_path = weights_path
        print(f"[Stage 1] Loading YOLOv8 quadrant segmenter from: {weights_path}")
        if os.path.exists(weights_path):
            self.model = YOLO(weights_path)
        else:
            print(f"[Stage 1] Warning: Checkpoint not found at {weights_path}. Falling back to yolov8n-seg.pt")
            self.model = YOLO("yolov8n-seg.pt")

    def detect_quadrants(self, image_bgr: np.ndarray, conf: float = 0.25) -> List[Dict[str, Any]]:
        """Runs instance segmentation to localize the 4 clinical quadrants."""
        results = self.model(image_bgr, conf=conf, verbose=False)
        detections = []
        if not results or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf_val = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].tolist()
            detections.append({
                "quadrant_id": cls_id,
                "quadrant_name": QUADRANT_NAMES.get(cls_id, f"Quadrant_{cls_id + 1}"),
                "confidence": conf_val,
                "box": [round(c, 1) for c in xyxy]
            })
        return detections


class FDIInventoryManager:
    """
    Correlates detected teeth or annotation records to adult FDI numbering
    and identifies missing teeth candidate implant sites.
    """

    def __init__(self, annotation_file: Optional[str] = None):
        self.annotations_by_filename = defaultdict(list)
        self.image_meta_by_filename = {}
        if annotation_file and os.path.exists(annotation_file):
            self._load_annotations(annotation_file)

    def _load_annotations(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            id_to_file = {img["id"]: img for img in data.get("images", [])}
            for ann in data.get("annotations", []):
                img_info = id_to_file.get(ann.get("image_id"))
                if img_info:
                    fn = img_info["file_name"]
                    self.image_meta_by_filename[fn] = img_info
                    self.annotations_by_filename[fn].append(ann)
        except Exception as e:
            print(f"[Inventory] Warning: Failed to load annotation database: {e}")

    def get_tooth_inventory(
            self,
            image_path: str,
            image_shape: Tuple[int, int],
            detected_quadrants: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns:
            present_teeth: List of localized tooth regions with assigned FDI numbers.
            missing_teeth: List of missing teeth identified via arch gap analysis.
        """
        filename = os.path.basename(image_path)
        ih, iw = image_shape[:2]
        present_teeth = []

        # 1. Check if ground-truth tooth enumerations are available in the dataset
        if filename in self.annotations_by_filename:
            img_info = self.image_meta_by_filename[filename]
            ow, oh = float(img_info.get("width", iw)), float(img_info.get("height", ih))
            sx, sy = iw / ow, ih / oh

            for ann in self.annotations_by_filename[filename]:
                q_id = ann.get("category_id_1", 0)  # 0..3
                pos_id = ann.get("category_id_2", 0)  # 0..7
                fdi_num = (q_id + 1) * 10 + (pos_id + 1)

                bx, by, bw, bh = ann["bbox"]
                x1 = int(bx * sx)
                y1 = int(by * sy)
                x2 = int((bx + bw) * sx)
                y2 = int((by + bh) * sy)

                # Clamp to image frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                present_teeth.append({
                    "fdi_number": fdi_num,
                    "description": get_fdi_description(fdi_num),
                    "quadrant_id": q_id,
                    "box": [x1, y1, x2, y2],
                    "center_x": (x1 + x2) // 2,
                    "center_y": (y1 + y2) // 2,
                })
        else:
            # 2. Automated Partitioning fallback across detected quadrants
            for quad in detected_quadrants:
                qid = quad["quadrant_id"]
                x1, y1, x2, y2 = quad["box"]
                qw = max(1, x2 - x1)
                qh = max(1, y2 - y1)
                # Partition quadrant horizontally into standard 8 dental position intervals
                num_slots = 8
                slot_w = qw / num_slots

                # In Q1 & Q4 (Patient Right), central incisor is at the midline (left or right depending on quadrant)
                # Standard OPG orientation: Midline is at center (~X = 512 in a 1024-wide scan)
                # Q1 & Q4 are on the patient's right -> displayed on the left half of the OPG
                # Teeth 8..1 proceed toward center: position 8 (lateral) to position 1 (midline)
                is_right_side = qid in (0, 3)

                for slot_idx in range(num_slots):
                    pos = (num_slots - slot_idx) if is_right_side else (slot_idx + 1)
                    fdi_num = (qid + 1) * 10 + pos

                    tx1 = int(x1 + slot_idx * slot_w)
                    tx2 = int(tx1 + slot_w)
                    ty1 = int(y1)
                    ty2 = int(y2)

                    present_teeth.append({
                        "fdi_number": fdi_num,
                        "description": get_fdi_description(fdi_num),
                        "quadrant_id": qid,
                        "box": [tx1, ty1, tx2, ty2],
                        "center_x": (tx1 + tx2) // 2,
                        "center_y": (ty1 + ty2) // 2,
                    })

        # 3. Gap Analysis for Missing Teeth
        present_fdis = {t["fdi_number"] for t in present_teeth}
        missing_teeth = []

        for fdi_num in ALL_32_FDI_NUMBERS:
            if fdi_num not in present_fdis:
                quad = fdi_num // 10
                is_upper = quad in (1, 2)
                missing_teeth.append({
                    "fdi_number": fdi_num,
                    "description": get_fdi_description(fdi_num),
                    "jaw": "Upper Jaw (Maxilla)" if is_upper else "Lower Jaw (Mandible)",
                    "quadrant": quad,
                    "is_implant_candidate": (fdi_num % 10) != 8,  # Wisdom teeth excluded by default
                })

        return present_teeth, missing_teeth


# ---------------------------------------------------------------------------
# Stage 2A: Multi-Label Pathology Classifier
# ---------------------------------------------------------------------------
class Stage2APathologyService:
    """Evaluates cropped tooth patches for Caries, Periapical Lesions, Crowns, and RCTs."""

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = EfficientNetB3MultiLabel(
            num_classes=NUM_PATHOLOGY_CLASSES,
            pretrained=False,
            dropout=0.0
        )
        print(f"[Stage 2A] Loading EfficientNet-B3 pathology weights from: {weights_path}")
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            print("[Stage 2A] Checkpoint loaded successfully!")
        else:
            print(f"[Stage 2A] Warning: Weights file not found at {weights_path}. Using base initialized model.")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def classify_tooth_patch(self, patch_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs multi-label forward pass on a single cropped tooth ROI."""
        if patch_bgr is None or patch_bgr.size == 0:
            return {c: {"detected": False, "confidence": 0.0} for c in PATHOLOGY_CLASSES}

        # Safe radiological channel check: Force 3-channel compliance dynamically
        if len(patch_bgr.shape) == 2:
            # For pure 2D arrays, merge it into 3 identical color layers cleanly
            patch_bgr = cv2.merge([patch_bgr, patch_bgr, patch_bgr])
        elif len(patch_bgr.shape) == 3 and patch_bgr.shape[2] == 1:
            # For 3D arrays with a single channel, expand it to BGR standard
            patch_bgr = cv2.cvtColor(patch_bgr, cv2.COLOR_GRAY2BGR)

        # Standard conversion to RGB for standard PIL/EfficientNet parsing
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(patch_rgb)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        results = {}
        thresholds = {
            "Caries": 0.85,
            "Periapical_Lesion": 0.55,
            "Existing_Crown": 0.75,
            "RCT": 0.75
        }
        for i, cname in enumerate(PATHOLOGY_CLASSES):
            prob = float(probs[i])
            th = thresholds.get(cname, 0.50)
            results[cname] = {
                "detected": bool(prob >= th),
                "confidence": round(prob, 4)
            }
        return results


# ---------------------------------------------------------------------------
# Stage 2B: Landmark Regression & Section 6 Decision Engine
# ---------------------------------------------------------------------------
class Stage2BLandmarkService:
    """Predicts anatomical boundary curves and evaluates Section 6 implant site clearances."""

    def __init__(self, device: str = "cpu", pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM):
        self.device = torch.device(device)
        self.pixel_spacing_mm = pixel_spacing_mm
        self.model = LandmarkUNetRegression(
            in_channels=1,
            num_landmarks=len(LANDMARK_CHANNELS),
            base_features=32
        ).to(self.device)
        self.model.eval()
        print("[Stage 2B] LandmarkUNetRegression initialized with Section 6 decision rules.")

    def predict_landmarks(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Runs U-Net forward pass and returns continuous coordinate curves (3, W):
          Row 0: Bone Crest y(x)
          Row 1: Maxillary Sinus Floor y(x)
          Row 2: Mandibular Nerve Canal y(x)
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        orig_h, orig_w = gray.shape
        model_h, model_w = 256, 512

        # Resize for model forward pass
        resized_gray = cv2.resize(gray, (model_w, model_h), interpolation=cv2.INTER_AREA)
        norm_tensor = torch.from_numpy(resized_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(
            self.device)

        with torch.no_grad():
            _, coords_y_downscaled = self.model(norm_tensor)
            coords_np = coords_y_downscaled[0].cpu().numpy()  # (3, model_w)

        # Scale coordinates back to original panoramic resolution
        scale_y = orig_h / model_h
        scale_x = orig_w / model_w

        # Resample along width to match orig_w
        resampled_coords = np.zeros((3, orig_w), dtype=np.float32)
        x_orig = np.arange(orig_w)
        x_down = np.linspace(0, orig_w - 1, model_w)

        for c in range(3):
            curve_scaled = coords_np[c] * scale_y
            resampled_coords[c] = np.interp(x_orig, x_down, curve_scaled)

        return resampled_coords

    def evaluate_site_clearance(
            self,
            col_x: int,
            is_upper_jaw: bool,
            landmark_coords: np.ndarray
    ) -> Dict[str, Any]:
        """Applies Section 6 decision rules to compute residual bone height and surgical clearance."""
        col_x = min(max(0, col_x), landmark_coords.shape[1] - 1)
        crest_y = float(landmark_coords[0, col_x])

        if is_upper_jaw:
            sinus_y = float(landmark_coords[1, col_x])
            dist_px = abs(crest_y - sinus_y)
            dist_mm = dist_px * self.pixel_spacing_mm
            return evaluate_upper_jaw_site(dist_mm)
        else:
            nerve_y = float(landmark_coords[2, col_x])
            dist_px = abs(nerve_y - crest_y)
            dist_mm = dist_px * self.pixel_spacing_mm
            return evaluate_lower_jaw_site(dist_mm, safety_buffer_mm=2.0)


# ---------------------------------------------------------------------------
# Dental AI Screening Master Pipeline
# ---------------------------------------------------------------------------
class DentalAIScreeningPipeline:
    """Master orchestrator integrating Stage 1, Stage 2A, and Stage 2B."""

    def __init__(
            self,
            stage1_weights: str = "runs/segment/runs/stage1_train/quadrant_segmentation/weights/best.pt",
            stage2a_weights: str = "runs/stage2a_train/weights/best_stage2a.pt",
            annotation_file: Optional[str] = os.path.join(
                "data", "raw", "training_data", "training_data",
                "quadrant_enumeration", "train_quadrant_enumeration.json"
            ),
            pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM,
            device: str = "cpu"
    ):
        self.device = device
        self.pixel_spacing_mm = pixel_spacing_mm

        print("\n" + "=" * 70)
        print("      INITIALIZING DENTAL AI SCREENING MASTER PIPELINE")
        print("=" * 70)
        self.stage1 = Stage1Detector(stage1_weights, device=device)
        self.inventory_mgr = FDIInventoryManager(annotation_file)
        self.stage2a = Stage2APathologyService(stage2a_weights, device=device)
        self.stage2b = Stage2BLandmarkService(device=device, pixel_spacing_mm=pixel_spacing_mm)
        print("[Pipeline] All vision modules successfully loaded and ready for inference.\n")

    def process_panoramic_scan(
            self,
            image_path: str,
            output_dir: str = os.path.join("runs", "pipeline_output"),
            save_visual: bool = True
    ) -> Dict[str, Any]:
        """Executes full diagnostic pipeline on a single panoramic radiograph."""
        t_start = time.time()
        print(f"[Pipeline] Processing OPG radiograph: {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input radiograph not found at: {image_path}")

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Failed to decode image at: {image_path}")

        ih, iw, _ = image_bgr.shape
        os.makedirs(output_dir, exist_ok=True)

        # -------------------------------------------------------------------
        # 1. Stage 1: Quadrant Localization & FDI Inventory
        # -------------------------------------------------------------------
        t1 = time.time()
        quadrants = self.stage1.detect_quadrants(image_bgr)
        present_teeth, missing_teeth = self.inventory_mgr.get_tooth_inventory(
            image_path=image_path,
            image_shape=(ih, iw),
            detected_quadrants=quadrants
        )
        stage1_time = time.time() - t1

        # -------------------------------------------------------------------
        # 2. Stage 2A: Per-Tooth Pathology & Restoration Classification
        # -------------------------------------------------------------------
        t2 = time.time()
        tooth_findings = []
        pathology_summary = {"Caries": 0, "Periapical_Lesion": 0, "Existing_Crown": 0, "RCT": 0}

        for tooth in present_teeth:
            x1, y1, x2, y2 = tooth["box"]
            # Apply slight padding for apical & coronal margin capture
            pw, ph = int((x2 - x1) * 0.10), int((y2 - y1) * 0.10)
            cx1 = max(0, x1 - pw)
            cy1 = max(0, y1 - ph)
            cx2 = min(iw, x2 + pw)
            cy2 = min(ih, y2 + ph)

            patch = image_bgr[cy1:cy2, cx1:cx2]
            analysis = self.stage2a.classify_tooth_patch(patch)

            # Record detections
            detected_conditions = [c for c, val in analysis.items() if val["detected"]]
            for c in detected_conditions:
                pathology_summary[c] += 1

            tooth_findings.append({
                "fdi_number": tooth["fdi_number"],
                "description": tooth["description"],
                "quadrant_id": tooth["quadrant_id"],
                "box": [x1, y1, x2, y2],
                "conditions_detected": detected_conditions,
                "detailed_probabilities": analysis
            })
        stage2a_time = time.time() - t2

        # -------------------------------------------------------------------
        # 3. Stage 2B: Landmark Regression & Implant Site Clearance
        # -------------------------------------------------------------------
        t3 = time.time()
        landmark_coords = self.stage2b.predict_landmarks(image_bgr)

        implant_site_evaluations = []
        # Evaluate missing teeth sites as primary implant candidates
        candidate_missing = [m for m in missing_teeth if m.get("is_implant_candidate", True)]

        # If no missing teeth were detected, evaluate posterior molar probe zones
        if not candidate_missing:
            candidate_missing = [
                {"fdi_number": 16, "description": "Upper Right First Molar", "jaw": "Upper Jaw (Maxilla)",
                 "quadrant": 1},
                {"fdi_number": 26, "description": "Upper Left First Molar", "jaw": "Upper Jaw (Maxilla)",
                 "quadrant": 2},
                {"fdi_number": 36, "description": "Lower Left First Molar", "jaw": "Lower Jaw (Mandible)",
                 "quadrant": 3},
                {"fdi_number": 46, "description": "Lower Right First Molar", "jaw": "Lower Jaw (Mandible)",
                 "quadrant": 4},
            ]

        # Map candidate sites to horizontal column positions
        col_mappings = {
            1: int(iw * 0.25),  # Upper right posterior
            2: int(iw * 0.75),  # Upper left posterior
            3: int(iw * 0.72),  # Lower left posterior
            4: int(iw * 0.28),  # Lower right posterior
        }

        for site in candidate_missing:
            fdi = site["fdi_number"]
            quad = site["quadrant"]
            is_upper = quad in (1, 2)
            col_x = col_mappings.get(quad, iw // 2)

            clearance = self.stage2b.evaluate_site_clearance(
                col_x=col_x,
                is_upper_jaw=is_upper,
                landmark_coords=landmark_coords
            )
            clearance["fdi_number"] = fdi
            clearance["description"] = site["description"]
            clearance["column_x"] = col_x
            implant_site_evaluations.append(clearance)
        stage2b_time = time.time() - t3

        # -------------------------------------------------------------------
        # 4. Treatment Planning Engine & Report Synthesis
        # -------------------------------------------------------------------
        treatment_plan = self._generate_treatment_plan(
            tooth_findings=tooth_findings,
            missing_teeth=missing_teeth,
            implant_evaluations=implant_site_evaluations
        )

        total_runtime = time.time() - t_start

        # Consolidated Report Structure
        report = {
            "meta": {
                "scan_file": os.path.basename(image_path),
                "resolution": f"{iw}x{ih}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_inference_time_sec": round(total_runtime, 3),
                "timing_breakdown": {
                    "stage1_quadrants_sec": round(stage1_time, 3),
                    "stage2a_pathology_sec": round(stage2a_time, 3),
                    "stage2b_landmarks_sec": round(stage2b_time, 3),
                }
            },
            "dentition_inventory": {
                "total_present_teeth": len(present_teeth),
                "total_missing_teeth": len(missing_teeth),
                "present_fdi_list": sorted([t["fdi_number"] for t in present_teeth]),
                "missing_fdi_list": sorted([m["fdi_number"] for m in missing_teeth]),
            },
            "pathology_summary": pathology_summary,
            "tooth_level_findings": tooth_findings,
            "implant_site_evaluations": implant_site_evaluations,
            "preliminary_treatment_plan": treatment_plan
        }

        # Generate Section 7.2 clean report (stripping raw decimals, mapping plain flags, attaching legal disclaimer)
        clean_report = format_clean_report(report)

        # Save Clean Structured JSON Report (Section 7.2 Specification)
        json_path = os.path.join(output_dir, "report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clean_report, f, indent=2)
        print(f"[Pipeline] Section 7.2 clean report written to: {json_path}")

        # Also save technical raw debug report (continuous probabilities)
        raw_json_path = os.path.join(output_dir, "report_raw.json")
        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Optional Annotated Visual Output
        if save_visual:
            visual_path = os.path.join(output_dir, "annotated_screening.jpg")
            self._render_annotated_image(image_bgr, tooth_findings, landmark_coords, visual_path)
            print(f"[Pipeline] Annotated visual screening overlay saved to: {visual_path}")

        return clean_report


    def _generate_treatment_plan(
            self,
            tooth_findings: List[Dict[str, Any]],
            missing_teeth: List[Dict[str, Any]],
            implant_evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Synthesizes prioritized actionable dental treatment suggestions."""
        plan = []

        # Priority 1: Periapical Lesions (Infection & Endodontic Emergency)
        for t in tooth_findings:
            if "Periapical_Lesion" in t["conditions_detected"]:
                plan.append({
                    "priority": "HIGH (Urgent Infection)",
                    "target": f"Tooth #{t['fdi_number']} ({t['description']})",
                    "clinical_finding": "Periapical apical radiolucency / endodontic infection",
                    "action_item": "Perform CBCT / vitality testing. Schedule endodontic treatment or apicoectomy."
                })

        # Priority 2: Caries Lesions (Demineralization & Restorative Care)
        for t in tooth_findings:
            if "Caries" in t["conditions_detected"]:
                plan.append({
                    "priority": "MEDIUM (Restorative)",
                    "target": f"Tooth #{t['fdi_number']} ({t['description']})",
                    "clinical_finding": "Coronal radiolucent caries decay",
                    "action_item": "Composite resin restoration or inlay/onlay preparation."
                })

        # Priority 3: Implant Site Recommendations
        for imp in implant_evaluations:
            status = imp["status"]
            fdi = imp["fdi_number"]
            site_desc = imp.get("description", f"Site #{fdi}")
            h_val = imp["residual_bone_height_mm"]

            if "SINUS LIFT" in status:
                plan.append({
                    "priority": "PLANNED (Surgical Augmentation)",
                    "target": f"Implant Site #{fdi} ({site_desc})",
                    "clinical_finding": f"Maxillary sinus proximity (Residual Bone: {h_val}mm, Status: {status})",
                    "action_item": imp["recommendation"]
                })
            elif "NERVE" in status or "HIGH CAUTION" in status:
                plan.append({
                    "priority": "HIGH CAUTION (Nerve Proximity)",
                    "target": f"Implant Site #{fdi} ({site_desc})",
                    "clinical_finding": f"Mandibular canal safety buffer alert (Residual Bone: {h_val}mm, Usable: {imp.get('usable_bone_height_mm', 'N/A')}mm)",
                    "action_item": imp["recommendation"]
                })
            elif status == "CLEAR":
                plan.append({
                    "priority": "ELECTIVE (Standard Implant Placement)",
                    "target": f"Implant Site #{fdi} ({site_desc})",
                    "clinical_finding": f"Adequate vertical bone volume ({h_val}mm)",
                    "action_item": imp["recommendation"]
                })

        return plan


    def _render_annotated_image(
            self,
            image_bgr: np.ndarray,
            tooth_findings: List[Dict[str, Any]],
            landmarks: np.ndarray,
            output_path: str
    ):
        """Renders diagnostic overlay with tooth bounding boxes, findings, and landmark curves."""
        vis = image_bgr.copy()
        H, W = vis.shape[:2]

        # Draw Landmark Curves
        # Channel 0: Bone Crest (Green)
        # Channel 1: Maxillary Sinus Floor (Cyan)
        # Channel 2: Mandibular Nerve Canal (Yellow)
        colors = [(0, 255, 0), (255, 255, 0), (0, 215, 255)]
        names = ["Bone Crest", "Sinus Floor", "IAN Canal"]

        for c, color in enumerate(colors):
            pts = []
            for x in range(0, W, 4):
                y = int(np.clip(landmarks[c, x], 0, H - 1))
                pts.append([x, y])
            pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Draw Tooth Findings
        for t in tooth_findings:
            x1, y1, x2, y2 = t["box"]
            conds = t["conditions_detected"]

            # Color code based on severity
            box_color = (0, 200, 0)
            if "Periapical_Lesion" in conds:
                box_color = (0, 0, 255)  # Red for infections
            elif "Caries" in conds:
                box_color = (0, 140, 255)  # Orange for cavities
            elif "Existing_Crown" in conds or "RCT" in conds:
                box_color = (255, 100, 0)  # Blue for restorations

            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 1)

            label = f"#{t['fdi_number']}"
            if conds:
                label += f": {','.join([c[:4] for c in conds])}"
            cv2.putText(vis, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color, 1, cv2.LINE_AA)

        # Legend Box
        cv2.putText(vis, "DENTAL AI SCREENING PIPELINE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(vis, "Green: Bone Crest | Cyan: Sinus Floor | Yellow: IAN Canal", (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (200, 200, 200), 1)

        # Mandatory Section 7.2 Legal Disclaimer Banner
        cv2.putText(
            vis,
            MANDATORY_LEGAL_DISCLAIMER,
            (15, H - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imwrite(output_path, vis)


# ---------------------------------------------------------------------------
# CLI Execution & Formatted Console Reporting
# ---------------------------------------------------------------------------
def print_console_summary(clean_report: Dict[str, Any]):
    """Prints Section 7.2 call center briefing and clinical summary in terminal."""
    print(generate_call_center_text_briefing(clean_report))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dental AI Screening Master Orchestration Pipeline"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join("data", "processed", "images", "train", "train_0.png"),
        help="Path to panoramic radiograph (OPG) image."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("runs", "pipeline_output"),
        help="Directory to save structured JSON report and diagnostic overlay."
    )
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=DEFAULT_PIXEL_SPACING_MM,
        help="Panoramic spatial calibration (mm per pixel)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device ('cpu' or 'cuda')."
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Disable saving annotated diagnostic visual image."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = DentalAIScreeningPipeline(
        stage1_weights="runs/segment/runs/stage1_train/quadrant_segmentation/weights/best.pt",
        stage2a_weights="runs/stage2a_train/weights/best_stage2a.pt",
        pixel_spacing_mm=args.pixel_spacing,
        device=args.device
    )

    report = pipeline.process_panoramic_scan(
        image_path=args.image,
        output_dir=args.output_dir,
        save_visual=not args.no_visual
    )

    print_console_summary(report)


if __name__ == "__main__":
    main()
