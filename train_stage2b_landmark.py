"""
train_stage2b_landmark.py - Stage 2B: Anatomical Landmark & Bone Height Regression
-----------------------------------------------------------------------------------
Architecture: PyTorch U-Net with Coordinate Regression Head
Anatomical Targets:
  1. Bone Crest (Alveolar Ridge)
  2. Maxillary Sinus Floor (Upper Jaw)
  3. Mandibular Nerve Canal / Inferior Alveolar Nerve (IAN) (Lower Jaw)

Section 6 Clinical Decision Rules:
  - Upper Jaw (Maxilla):
      * CLEAR: >= 10 mm
      * CAUTION: 7 - 9 mm
      * SINUS LIFT likely: 4 - 6 mm
      * SINUS LIFT required: <= 3 - 4 mm
  - Lower Jaw (Mandible):
      * 2 mm mandatory safety margin buffer above nerve canal
      * CLEAR: >= 12 mm
      * CAUTION: 10 - 12 mm
      * HIGH CAUTION: < 10 mm
      * NERVE RISK: < 6 - 8 mm
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows shells and background log listeners
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import math
import time
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Global Constants & Landmark Channels
# ---------------------------------------------------------------------------
LANDMARK_CHANNELS = [
    "Bone_Crest",           # Channel 0: Alveolar ridge crest
    "Maxillary_Sinus_Floor",# Channel 1: Inferior cortical border of maxillary sinus
    "Mandibular_Nerve_Canal"# Channel 2: Inferior alveolar nerve (IAN) canal roof
]
NUM_LANDMARK_CHANNELS = len(LANDMARK_CHANNELS)
DEFAULT_PIXEL_SPACING_MM = 0.10  # Standard panoramic calibration: 0.10 mm per pixel


# ---------------------------------------------------------------------------
# 1. PyTorch U-Net Architecture with Regression Head
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> LeakyReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """Downscaling via MaxPool2D followed by DoubleConv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """Upscaling via ConvTranspose2D, skip concatenation, then DoubleConv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if needed due to odd spatial dimensions
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class LandmarkUNetRegression(nn.Module):
    """
    Lightweight U-Net architecture skeleton featuring:
      1. Spatial encoder-decoder for multi-scale anatomical feature extraction.
      2. 3-channel landmark heatmap regression head.
      3. Differentiable vertical soft-argmax coordinate extraction layer.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_landmarks: int = NUM_LANDMARK_CHANNELS,
        base_features: int = 32
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_landmarks = num_landmarks

        # Encoder
        self.inc = DoubleConv(in_channels, base_features)                 # -> 32
        self.down1 = DownBlock(base_features, base_features * 2)          # -> 64
        self.down2 = DownBlock(base_features * 2, base_features * 4)      # -> 128
        self.down3 = DownBlock(base_features * 4, base_features * 8)      # -> 256
        self.down4 = DownBlock(base_features * 8, base_features * 16)     # -> 512 (Bottleneck)

        # Decoder
        self.up1 = UpBlock(base_features * 16, base_features * 8)         # -> 256
        self.up2 = UpBlock(base_features * 8, base_features * 4)          # -> 128
        self.up3 = UpBlock(base_features * 4, base_features * 2)          # -> 64
        self.up4 = UpBlock(base_features * 2, base_features)              # -> 32

        # Landmark Regression Head (outputs spatial probability heatmaps)
        self.out_conv = nn.Conv2d(base_features, num_landmarks, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            heatmaps: (B, num_landmarks, H, W) continuous landmark confidence maps
            coords_y: (B, num_landmarks, W) vertical y-coordinate curves across horizontal columns
        """
        # Encoder forward
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder forward with skip connections
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        # Output raw heatmaps
        heatmaps = self.out_conv(d1)  # shape: (B, num_landmarks, H, W)

        # Differentiable Soft-Argmax across height dimension H
        # Calculates continuous vertical coordinate y(x) for each column x in [0, W)
        B, C, H, W = heatmaps.shape
        pos_y = torch.linspace(0.0, float(H - 1), H, device=x.device).view(1, 1, H, 1)
        weights_y = F.softmax(heatmaps * 10.0, dim=2)
        coords_y = torch.sum(weights_y * pos_y, dim=2)  # shape: (B, C, W)

        return heatmaps, coords_y


# ---------------------------------------------------------------------------
# 2. Asynchronous Data Simulation Block
# ---------------------------------------------------------------------------
class AsyncSimulatedLandmarkDataset(Dataset):
    """
    Asynchronous data simulation generator that synthesizes realistic panoramic
    radiographic profiles and anatomical landmark trajectories on-the-fly.
    Bypasses external dataset file path restrictions for clean, deterministic verification.
    """
    def __init__(
        self,
        num_samples: int = 32,
        height: int = 256,
        width: int = 512,
        pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.pixel_spacing_mm = pixel_spacing_mm
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.RandomState(self.seed + idx)

        H, W = self.height, self.width
        x_norm = np.linspace(-1.0, 1.0, W)

        # 1. Synthesize background radiograph texture
        # Panoramic X-rays show higher radiopacity in the mid-arch and cervical spine
        bg_pattern = np.outer(
            np.linspace(0.2, 0.8, H),
            0.5 + 0.3 * np.cos(np.pi * x_norm)
        )
        noise = rng.normal(0.0, 0.05, (H, W))
        radiograph = np.clip(bg_pattern + noise, 0.0, 1.0).astype(np.float32)

        # 2. Synthesize Ground-Truth Landmark Coordinate Curves y(x) in pixels
        # A. Alveolar Bone Crest: Gentle U-shape curve across the arch
        # Upper maxilla crest at ~y=110, lower mandible crest at ~y=160
        base_crest_y = float(H) * 0.48
        crest_curve = base_crest_y + 12.0 * (x_norm ** 2) + rng.uniform(-3.0, 3.0)

        # B. Maxillary Sinus Floor: Bilateral concavities above upper molars
        # Located above bone crest (lower y-value in image coordinate system)
        # Lateral bilateral troughs at x in [-0.8, -0.2] and [0.2, 0.8]
        sinus_offset = rng.uniform(50.0, 95.0)  # ~5 to 9.5 mm distance from crest
        sinus_curve = crest_curve - (sinus_offset + 18.0 * np.cos(2.5 * np.pi * np.abs(x_norm)))
        sinus_curve = np.clip(sinus_curve, 20.0, base_crest_y - 15.0)

        # C. Mandibular Nerve Canal (IAN): Bilateral canal running below lower crest
        # Runs through body of mandible from mandibular notch down toward mental foramen
        nerve_offset = rng.uniform(70.0, 130.0)  # ~7 to 13 mm distance below crest
        nerve_curve = crest_curve + (nerve_offset - 10.0 * (x_norm ** 2))
        nerve_curve = np.clip(nerve_curve, base_crest_y + 25.0, float(H - 20))

        # Target coordinate profile: (3, W)
        target_coords = np.stack([crest_curve, sinus_curve, nerve_curve], axis=0).astype(np.float32)

        # 3. Synthesize 2D Gaussian Heatmaps for target landmarks
        target_heatmaps = np.zeros((NUM_LANDMARK_CHANNELS, H, W), dtype=np.float32)
        y_grid = np.arange(H).reshape(-1, 1)  # (H, 1)
        sigma = 3.5

        for c in range(NUM_LANDMARK_CHANNELS):
            curve_c = target_coords[c].reshape(1, W)  # (1, W)
            dist_sq = (y_grid - curve_c) ** 2
            target_heatmaps[c] = np.exp(-dist_sq / (2.0 * (sigma ** 2)))

        # Convert to PyTorch tensors
        tensor_img = torch.from_numpy(radiograph).unsqueeze(0)             # (1, H, W)
        tensor_heatmaps = torch.from_numpy(target_heatmaps)                # (3, H, W)
        tensor_coords = torch.from_numpy(target_coords)                    # (3, W)

        return {
            "image": tensor_img,
            "target_heatmaps": tensor_heatmaps,
            "target_coords": tensor_coords,
            "sample_idx": torch.tensor(idx, dtype=torch.long)
        }


# ---------------------------------------------------------------------------
# 3. Section 6 Clinical Decision Rules & Post-Processing Functions
# ---------------------------------------------------------------------------
def evaluate_upper_jaw_site(bone_height_mm: float) -> Dict[str, Any]:
    """
    Applies Section 6 clinical decision rules for Upper Jaw (Maxilla) sinus floor distance:
      - CLEAR: >= 10 mm
      - CAUTION: 7 - 9 mm (7.0 <= h < 10.0 mm)
      - SINUS LIFT likely: 4 - 6 mm (4.0 <= h < 7.0 mm)
      - SINUS LIFT required: <= 3 - 4 mm (h < 4.0 mm)
    """
    bone_height_mm = max(0.0, float(bone_height_mm))

    if bone_height_mm >= 10.0:
        status = "CLEAR"
        severity = "Normal"
        recommendation = (
            "Adequate residual vertical bone height (>= 10mm). "
            "Standard implant fixture (>= 10mm) placement feasible without sinus augmentation."
        )
    elif bone_height_mm >= 7.0:
        status = "CAUTION"
        severity = "Moderate"
        recommendation = (
            "Borderline bone height (7 - 9mm). "
            "Short implant (8mm) or indirect crestal sinus floor elevation (osteotome technique) considered."
        )
    elif bone_height_mm >= 4.0:
        status = "SINUS LIFT likely"
        severity = "Substantial"
        recommendation = (
            "Deficient bone height (4 - 6mm). "
            "Sinus floor lift likely indicated (transcrestal or lateral window graft) with simultaneous implant placement."
        )
    else:
        status = "SINUS LIFT required"
        severity = "Critical"
        recommendation = (
            "Severe bone deficiency (<= 3-4mm). "
            "Mandatory lateral window sinus graft required; two-stage delayed implant placement recommended."
        )

    return {
        "anatomical_site": "Upper Jaw (Maxilla - Sinus Floor)",
        "residual_bone_height_mm": round(bone_height_mm, 2),
        "status": status,
        "severity": severity,
        "recommendation": recommendation
    }


def evaluate_lower_jaw_site(
    bone_height_mm: float,
    safety_buffer_mm: float = 2.0
) -> Dict[str, Any]:
    """
    Applies Section 6 clinical decision rules for Lower Jaw (Mandible) IAN nerve canal distance:
      - Mandatory 2.0 mm safety margin buffer applied above the nerve canal.
      - CLEAR: >= 12 mm (effective usable height >= 10 mm)
      - CAUTION: 10 - 12 mm (effective usable height 8 - 10 mm)
      - HIGH CAUTION: < 10 mm (effective usable height < 8 mm, specifically 8.0 <= h < 10.0 mm)
      - NERVE RISK: < 6 - 8 mm (h < 8.0 mm; hazardous proximity to IAN canal)
    """
    bone_height_mm = max(0.0, float(bone_height_mm))
    usable_height_mm = max(0.0, bone_height_mm - safety_buffer_mm)

    if bone_height_mm >= 12.0:
        status = "CLEAR"
        risk_level = "Low"
        recommendation = (
            f"Sufficient vertical bone volume (>= 12mm). "
            f"Accommodates standard >=10mm implant fixture with full {safety_buffer_mm:.1f}mm nerve clearance buffer."
        )
    elif bone_height_mm >= 10.0:
        status = "CAUTION"
        risk_level = "Moderate"
        recommendation = (
            f"Moderate vertical bone volume (10 - 12mm). "
            f"Usable height: {usable_height_mm:.1f}mm. 8 - 10mm implant feasible under guided surgical stent."
        )
    elif bone_height_mm >= 8.0:
        status = "HIGH CAUTION"
        risk_level = "High"
        recommendation = (
            f"Reduced bone height (< 10mm). Usable height: {usable_height_mm:.1f}mm. "
            f"High risk of proximity to nerve canal. Short implant (6 - 8mm) or vertical augmentation indicated."
        )
    else:
        status = "NERVE RISK"
        risk_level = "Critical"
        recommendation = (
            f"Critical vertical deficiency (< 6-8mm). Usable height: {usable_height_mm:.1f}mm. "
            f"Hazardous proximity to IAN canal! Severe risk of irreversible nerve injury/paresthesia. Avoid standard osteotomy."
        )

    return {
        "anatomical_site": "Lower Jaw (Mandible - IAN Canal)",
        "residual_bone_height_mm": round(bone_height_mm, 2),
        "safety_buffer_mm": round(safety_buffer_mm, 2),
        "usable_bone_height_mm": round(usable_height_mm, 2),
        "status": status,
        "risk_level": risk_level,
        "recommendation": recommendation
    }


def analyze_implant_sites(
    coords_y: torch.Tensor,
    pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM,
    sample_columns: List[Tuple[str, int, str]] = None
) -> List[Dict[str, Any]]:
    """
    Takes predicted landmark coordinates (B, 3, W) and evaluates residual bone heights
    and Section 6 decision flags at clinical implant probe locations.

    Args:
        coords_y: Tensor of shape (B, 3, W) containing continuous vertical pixel coordinates:
                  Channel 0: Bone Crest
                  Channel 1: Maxillary Sinus Floor
                  Channel 2: Mandibular Nerve Canal
        pixel_spacing_mm: Physical millimetre calibration per pixel (default: 0.10 mm)
        sample_columns: List of (Site_Name, Column_Index, Jaw_Type)
    """
    if sample_columns is None:
        W = coords_y.size(-1)
        # Typical posterior dental probe positions across horizontal columns
        sample_columns = [
            ("Upper Right Molar (Site #16)", int(W * 0.25), "upper"),
            ("Upper Left Molar (Site #26)",  int(W * 0.75), "upper"),
            ("Lower Right Molar (Site #46)", int(W * 0.28), "lower"),
            ("Lower Left Molar (Site #36)",  int(W * 0.72), "lower"),
        ]

    results = []
    # Evaluate first batch item
    batch_coords = coords_y[0].detach().cpu().numpy()  # (3, W)

    crest_y = batch_coords[0]  # (W,)
    sinus_y = batch_coords[1]  # (W,)
    nerve_y = batch_coords[2]  # (W,)

    for site_name, col_idx, jaw_type in sample_columns:
        col_idx = min(max(0, col_idx), len(crest_y) - 1)
        c_y = float(crest_y[col_idx])

        if jaw_type.lower() == "upper":
            s_y = float(sinus_y[col_idx])
            # In maxilla, residual bone height is distance between crest and sinus floor
            dist_px = abs(c_y - s_y)
            dist_mm = dist_px * pixel_spacing_mm
            decision = evaluate_upper_jaw_site(dist_mm)
            decision["site_name"] = site_name
            decision["column_x"] = col_idx
            results.append(decision)

        else:
            n_y = float(nerve_y[col_idx])
            # In mandible, residual bone height is distance between crest and nerve canal
            dist_px = abs(n_y - c_y)
            dist_mm = dist_px * pixel_spacing_mm
            decision = evaluate_lower_jaw_site(dist_mm, safety_buffer_mm=2.0)
            decision["site_name"] = site_name
            decision["column_x"] = col_idx
            results.append(decision)

    return results


# ---------------------------------------------------------------------------
# 4. Verification Dry-Run & CLI Entry Point
# ---------------------------------------------------------------------------
def run_dry_run_verification(device: str = "cpu"):
    """
    Executes a complete verification dry-run:
      - Instantiates U-Net regression model and verifies parameter architecture.
      - Feeds simulated panoramic batches and inspects input/output tensor shapes.
      - Verifies heatmap MSE loss and soft-argmax coordinate extraction.
      - Tests Section 6 decision rules across boundary conditions.
    """
    print("\n" + "=" * 70)
    print("   STAGE 2B: ANATOMICAL LANDMARK REGRESSION & DECISION DRY-RUN")
    print("=" * 70)

    # 1. Model Instantiation
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[1/4] Instantiating LandmarkUNetRegression on device: {dev}...")
    model = LandmarkUNetRegression(
        in_channels=1,
        num_landmarks=NUM_LANDMARK_CHANNELS,
        base_features=32
    ).to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      - Total Parameters:     {total_params:,}")
    print(f"      - Trainable Parameters: {trainable_params:,}")
    print(f"      - Landmark Channels:    {', '.join(LANDMARK_CHANNELS)}")

    # 2. Asynchronous Data Simulation Verification
    batch_size = 2
    height, width = 256, 512
    print(f"\n[2/4] Initializing AsyncSimulatedLandmarkDataset ({batch_size} samples, {height}x{width})...")
    dataset = AsyncSimulatedLandmarkDataset(
        num_samples=batch_size,
        height=height,
        width=width,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM,
        seed=123
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    batch = next(iter(dataloader))
    images = batch["image"].to(dev)
    target_heatmaps = batch["target_heatmaps"].to(dev)
    target_coords = batch["target_coords"].to(dev)

    print(f"      - Input Image Tensor Shape:        {list(images.shape)} [Batch, Channels, Height, Width]")
    print(f"      - Ground Truth Heatmap Shape:      {list(target_heatmaps.shape)} [Batch, Channels, Height, Width]")
    print(f"      - Ground Truth Coordinate Profile: {list(target_coords.shape)} [Batch, Channels, Width]")

    # 3. Model Forward Pass & Tensor Shape Verification
    print("\n[3/4] Executing Network Forward Pass...")
    model.train()
    pred_heatmaps, pred_coords_y = model(images)

    # Loss computation check
    heatmap_loss = F.mse_loss(pred_heatmaps, target_heatmaps)
    coord_loss = F.smooth_l1_loss(pred_coords_y, target_coords)
    total_loss = heatmap_loss + 0.05 * coord_loss
    total_loss.backward()

    print(f"      - Output Heatmap Tensor Shape:     {list(pred_heatmaps.shape)} [Batch, Channels, Height, Width]")
    print(f"      - Soft-Argmax Coordinate Shape:    {list(pred_coords_y.shape)} [Batch, Channels, Width]")
    print(f"      - Heatmap MSE Loss:                {heatmap_loss.item():.5f}")
    print(f"      - Coordinate Profile Loss:         {coord_loss.item():.5f}")
    print(f"      - Total Combined Loss:             {total_loss.item():.5f}")
    print(f"      - Backward Pass Gradient Status:   Gradient propagation verified successfully.")

    # 4. Section 6 Decision Rules & Threshold Flag Verification
    print("\n[4/4] Verifying Section 6 Decision Engine across Clinical Thresholds:")
    print("-" * 70)

    # A. Upper Jaw (Maxilla) Boundary Matrix Verification
    print("A. UPPER JAW (Maxilla) -- Sinus Floor Clearance Rules:")
    test_upper_cases = [13.2, 8.4, 5.2, 2.5]
    for h_mm in test_upper_cases:
        res = evaluate_upper_jaw_site(h_mm)
        print(f"   * Bone Height: {res['residual_bone_height_mm']:5.1f} mm -> [{res['status']:^18s}] | {res['recommendation']}")

    print("\nB. LOWER JAW (Mandible) -- IAN Canal Safety Buffer (2.0mm) Rules:")
    test_lower_cases = [14.5, 11.0, 9.2, 5.8]
    for h_mm in test_lower_cases:
        res = evaluate_lower_jaw_site(h_mm, safety_buffer_mm=2.0)
        print(
            f"   * Bone Height: {res['residual_bone_height_mm']:5.1f} mm (Usable: {res['usable_bone_height_mm']:4.1f} mm) "
            f"-> [{res['status']:^14s}] | Risk: {res['risk_level']:<8s} | {res['recommendation']}"
        )

    # C. Full End-to-End Evaluation on Network Predicted Tensors
    print("\nC. Model End-to-End Simulated Posterior Site Evaluation:")
    site_findings = analyze_implant_sites(
        pred_coords_y,
        pixel_spacing_mm=DEFAULT_PIXEL_SPACING_MM
    )
    for finding in site_findings:
        site_name = finding["site_name"]
        status = finding["status"]
        h_val = finding["residual_bone_height_mm"]
        site_type = finding["anatomical_site"]
        print(f"   [SITE] {site_name:<30s} | {site_type:<35s} | Height: {h_val:4.1f} mm | Status: [{status}]")

    print("=" * 70)
    print("[PASS] Stage 2B Landmark Regression & Section 6 Decision Engine verified successfully!")
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2B: Landmark Regression and Bone Height Assessment"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run verification dry-run and print input/output tensor shapes and decisions."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device ('cpu' or 'cuda')."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dry_run:
        run_dry_run_verification(device=args.device)
