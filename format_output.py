"""
format_output.py - Section 7.2 Plain-Language Report Formatter
--------------------------------------------------------------
Post-processing module tailored for dental intake coordinators, call center agents,
and preliminary patient consultations:
  1. Strips all raw decimal probability confidences and continuous model internals.
  2. Standardizes anatomical and surgical flags to clean, plain-language text:
     - CLEAR
     - CAUTION
     - SINUS LIFT
     - NERVE RISK
  3. Appends the mandatory legal disclaimer text string to all report assets:
     "This is a pre-consultation screening report only. All findings require clinical confirmation by a licensed dentist."
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows shells
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# Section 7.2 Mandatory Legal Disclaimer (Exact String)
# ---------------------------------------------------------------------------
MANDATORY_LEGAL_DISCLAIMER = (
    "This is a pre-consultation screening report only. "
    "All findings require clinical confirmation by a licensed dentist."
)


# ---------------------------------------------------------------------------
# Plain-Language Flag Normalization
# ---------------------------------------------------------------------------
def normalize_implant_flag(raw_status: str) -> str:
    """
    Maps detailed model decisions into the 4 standard Section 7.2 plain-language flags:
      - CLEAR
      - CAUTION
      - SINUS LIFT
      - NERVE RISK
    """
    status_upper = raw_status.upper()
    if "NERVE" in status_upper:
        return "NERVE RISK"
    elif "SINUS" in status_upper:
        return "SINUS LIFT"
    elif "CAUTION" in status_upper:
        return "CAUTION"
    elif "CLEAR" in status_upper:
        return "CLEAR"
    return "CAUTION"


def format_tooth_findings_clean(tooth_findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strips raw probability decimals and converts tooth-level detections into clean,
    plain-language status records for intake staff.
    """
    clean_teeth = []
    for t in tooth_findings:
        fdi = t.get("fdi_number")
        desc = t.get("description", f"Tooth #{fdi}")
        conds = t.get("conditions_detected", [])

        has_caries = "Caries" in conds
        has_infection = "Periapical_Lesion" in conds
        has_crown = "Existing_Crown" in conds
        has_rct = "RCT" in conds

        # Synthesize plain-language summary tags
        flags = []
        if has_infection:
            flags.append("Apical Infection Suspected")
        if has_caries:
            flags.append("Caries Cavity Suspected")
        if has_crown:
            flags.append("Existing Crown Present")
        if has_rct:
            flags.append("Root Canal Treated")

        if not flags:
            overall_status = "HEALTHY / CLEAR"
            call_center_note = "No visible pathology detected."
        elif has_infection:
            overall_status = "URGENT ATTENTION NEEDED"
            call_center_note = "Potential infection detected at root apex. Recommend urgent evaluation."
        elif has_caries:
            overall_status = "TREATMENT RECOMMENDED"
            call_center_note = "Cavity/decay suspected. Restorative consultation needed."
        else:
            overall_status = "RESTORED / MONITOR"
            call_center_note = "Existing dental restorations present; routine follow-up."

        clean_teeth.append({
            "tooth_number": fdi,
            "tooth_name": desc,
            "overall_status": overall_status,
            "caries_decay": "SUSPECTED" if has_caries else "CLEAR",
            "root_infection": "SUSPECTED" if has_infection else "CLEAR",
            "prosthetic_crown": "PRESENT" if has_crown else "NONE",
            "root_canal_treatment": "PRESENT" if has_rct else "NONE",
            "clinical_flags": flags if flags else ["None (Healthy)"],
            "agent_talking_point": call_center_note
        })

    return clean_teeth


def format_implant_sites_clean(implant_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strips raw coordinate details and formats implant sites into clean
    call-center-friendly flags (CLEAR / CAUTION / SINUS LIFT / NERVE RISK).
    """
    clean_sites = []
    for site in implant_evaluations:
        raw_status = site.get("status", "CAUTION")
        clean_flag = normalize_implant_flag(raw_status)
        fdi = site.get("fdi_number")
        desc = site.get("description", f"Site #{fdi}")
        site_type = site.get("anatomical_site", "Implant Site")

        if clean_flag == "CLEAR":
            agent_talking_point = "Adequate vertical bone volume for standard implant placement."
        elif clean_flag == "CAUTION":
            agent_talking_point = "Moderate bone height. Doctor will evaluate bone density and short implant options."
        elif clean_flag == "SINUS LIFT":
            agent_talking_point = "Reduced bone height beneath sinus floor. Sinus lift procedure likely discussed during consultation."
        elif clean_flag == "NERVE RISK":
            agent_talking_point = "Proximity to mandibular nerve canal. CBCT scan required to safely plan treatment."
        else:
            agent_talking_point = "Consultation required for surgical assessment."

        clean_sites.append({
            "site_number": fdi,
            "site_name": desc,
            "jaw_region": site_type,
            "status_flag": clean_flag,
            "agent_talking_point": agent_talking_point,
            "clinical_action": site.get("recommendation", "")
        })

    return clean_sites


# ---------------------------------------------------------------------------
# Master Clean Report Transformer
# ---------------------------------------------------------------------------
def format_clean_report(raw_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms raw pipeline output into a clean, human-readable report:
      - Strips all raw probability floats, bounding boxes, and internal coordinates.
      - Standardizes findings into plain-language text flags.
      - Appends the mandatory legal disclaimer.
    """
    meta_raw = raw_report.get("meta", {})
    inv_raw = raw_report.get("dentition_inventory", {})
    patho_raw = raw_report.get("pathology_summary", {})
    teeth_raw = raw_report.get("tooth_level_findings", [])
    implant_raw = raw_report.get("implant_site_evaluations", [])
    plan_raw = raw_report.get("preliminary_treatment_plan", [])

    clean_teeth = format_tooth_findings_clean(teeth_raw)
    clean_implants = format_implant_sites_clean(implant_raw)

    # Simplified treatment summary for call center agents
    action_items = []
    for item in plan_raw:
        action_items.append({
            "urgency": item.get("priority", "ROUTINE"),
            "target": item.get("target", ""),
            "issue": item.get("clinical_finding", ""),
            "proposed_step": item.get("action_item", "")
        })

    clean_report = {
        "report_type": "Dental AI Screening Preliminary Assessment (Section 7.2 Call Center Edition)",
        "scan_file": meta_raw.get("scan_file", "OPG_Radiograph.png"),
        "assessment_timestamp": meta_raw.get("timestamp", ""),
        "patient_dentition_summary": {
            "total_teeth_present": inv_raw.get("total_present_teeth", 0),
            "total_teeth_missing": inv_raw.get("total_missing_teeth", 0),
            "missing_teeth_list": inv_raw.get("missing_fdi_list", []),
        },
        "pathology_overview": {
            "teeth_with_suspected_cavities": patho_raw.get("Caries", 0),
            "teeth_with_suspected_infections": patho_raw.get("Periapical_Lesion", 0),
            "teeth_with_existing_crowns": patho_raw.get("Existing_Crown", 0),
            "teeth_with_prior_root_canals": patho_raw.get("RCT", 0),
        },
        "implant_site_clearance_flags": clean_implants,
        "tooth_by_tooth_records": clean_teeth,
        "recommended_treatment_steps": action_items,
        "legal_disclaimer": MANDATORY_LEGAL_DISCLAIMER
    }

    return clean_report


# ---------------------------------------------------------------------------
# Call Center Text Briefing Generator
# ---------------------------------------------------------------------------
def generate_call_center_text_briefing(clean_report: Dict[str, Any]) -> str:
    """Generates an intuitive, readable call-center briefing script."""
    dent = clean_report["patient_dentition_summary"]
    patho = clean_report["pathology_overview"]
    implants = clean_report["implant_site_clearance_flags"]
    actions = clean_report["recommended_treatment_steps"]

    lines = []
    lines.append("=" * 76)
    lines.append("      PRE-CONSULTATION DENTAL SCREENING -- CALL CENTER SUMMARY")
    lines.append("=" * 76)
    lines.append(f" Scan Reference:     {clean_report['scan_file']}")
    lines.append(f" Assessment Time:    {clean_report['assessment_timestamp']}")
    lines.append("-" * 76)

    # 1. Dentition Overview
    missing_str = ", ".join(map(str, dent["missing_teeth_list"])) if dent["missing_teeth_list"] else "None (Full Dentition)"
    lines.append(" 1. PATIENT DENTITION INVENTORY:")
    lines.append(f"    * Teeth Present:   {dent['total_teeth_present']} of 32 adult teeth")
    lines.append(f"    * Missing Teeth:   {dent['total_teeth_missing']} ({missing_str})")
    lines.append("-" * 76)

    # 2. Key Clinical Alerts
    lines.append(" 2. INTAKE PATHOLOGY OVERVIEW:")
    lines.append(f"    * Active Cavities / Decay:    {patho['teeth_with_suspected_cavities']} tooth site(s)")
    lines.append(f"    * Potential Root Infections:  {patho['teeth_with_suspected_infections']} tooth site(s) [URGENT ATTENTION]")
    lines.append(f"    * Existing Crowns:            {patho['teeth_with_existing_crowns']} tooth site(s)")
    lines.append(f"    * Prior Root Canals:          {patho['teeth_with_prior_root_canals']} tooth site(s)")
    lines.append("-" * 76)

    # 3. Implant Feasibility Flags
    lines.append(" 3. IMPLANT SITE FEASIBILITY & ANATOMICAL CLEARANCE:")
    lines.append("    [Standard Plain-Language Flags: CLEAR | CAUTION | SINUS LIFT | NERVE RISK]")
    for imp in implants:
        fdi = imp["site_number"]
        name = imp["site_name"]
        flag = imp["status_flag"]
        talking_point = imp["agent_talking_point"]
        lines.append(f"    * Tooth #{fdi:2d} ({name:<28s}) -> [{flag:^12s}]")
        lines.append(f"      Agent Note: {talking_point}")
    lines.append("-" * 76)

    # 4. Action Steps for Patient Scheduling
    lines.append(" 4. SUGGESTED APPOINTMENT ACTION ITEMS:")
    for i, act in enumerate(actions[:5], start=1):
        lines.append(f"    [{i}] Urgency: {act['urgency']}")
        lines.append(f"        Site:    {act['target']}")
        lines.append(f"        Reason:  {act['issue']}")
        lines.append(f"        Step:    {act['proposed_step']}")
    lines.append("-" * 76)

    # 5. Mandatory Legal Disclaimer
    lines.append(" MANDATORY LEGAL NOTICE:")
    lines.append(f" \"{clean_report['legal_disclaimer']}\"")
    lines.append("=" * 76)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone Validation Test
# ---------------------------------------------------------------------------
def run_mock_validation():
    """Generates a realistic mock screening case and prints the clean text and JSON output."""
    mock_raw_report = {
        "meta": {
            "scan_file": "sample_patient_opg_042.png",
            "resolution": "1024x512",
            "timestamp": "2026-10-10 16:30:00",
            "total_inference_time_sec": 2.15,
            "timing_breakdown": {
                "stage1_quadrants_sec": 0.21,
                "stage2a_pathology_sec": 1.62,
                "stage2b_landmarks_sec": 0.32
            }
        },
        "dentition_inventory": {
            "total_present_teeth": 29,
            "total_missing_teeth": 3,
            "present_fdi_list": [
                11, 12, 13, 14, 15, 17, 18,
                21, 22, 23, 24, 25, 27, 28,
                31, 32, 33, 34, 35, 37, 38,
                41, 42, 43, 44, 45, 46, 47, 48
            ],
            "missing_fdi_list": [16, 26, 36]
        },
        "pathology_summary": {
            "Caries": 4,
            "Periapical_Lesion": 1,
            "Existing_Crown": 3,
            "RCT": 2
        },
        "tooth_level_findings": [
            {
                "fdi_number": 14,
                "description": "Upper Right First Premolar",
                "quadrant_id": 0,
                "box": [310, 140, 360, 230],
                "conditions_detected": ["Caries"],
                "detailed_probabilities": {
                    "Caries": {"detected": True, "confidence": 0.9412},
                    "Periapical_Lesion": {"detected": False, "confidence": 0.0812},
                    "Existing_Crown": {"detected": False, "confidence": 0.0211},
                    "RCT": {"detected": False, "confidence": 0.0430}
                }
            },
            {
                "fdi_number": 46,
                "description": "Lower Right First Molar",
                "quadrant_id": 3,
                "box": [220, 260, 290, 370],
                "conditions_detected": ["Periapical_Lesion", "RCT"],
                "detailed_probabilities": {
                    "Caries": {"detected": False, "confidence": 0.1245},
                    "Periapical_Lesion": {"detected": True, "confidence": 0.8872},
                    "Existing_Crown": {"detected": False, "confidence": 0.1104},
                    "RCT": {"detected": True, "confidence": 0.9231}
                }
            },
            {
                "fdi_number": 24,
                "description": "Upper Left First Premolar",
                "quadrant_id": 1,
                "box": [640, 145, 690, 235],
                "conditions_detected": ["Existing_Crown"],
                "detailed_probabilities": {
                    "Caries": {"detected": False, "confidence": 0.0512},
                    "Periapical_Lesion": {"detected": False, "confidence": 0.0210},
                    "Existing_Crown": {"detected": True, "confidence": 0.8914},
                    "RCT": {"detected": False, "confidence": 0.0632}
                }
            },
            {
                "fdi_number": 11,
                "description": "Upper Right Central Incisor",
                "quadrant_id": 0,
                "box": [480, 130, 520, 240],
                "conditions_detected": [],
                "detailed_probabilities": {
                    "Caries": {"detected": False, "confidence": 0.0311},
                    "Periapical_Lesion": {"detected": False, "confidence": 0.0105},
                    "Existing_Crown": {"detected": False, "confidence": 0.0150},
                    "RCT": {"detected": False, "confidence": 0.0210}
                }
            }
        ],
        "implant_site_evaluations": [
            {
                "site_name": "Upper Right First Molar (Site #16)",
                "fdi_number": 16,
                "description": "Upper Right First Molar",
                "anatomical_site": "Upper Jaw (Maxilla - Sinus Floor)",
                "residual_bone_height_mm": 5.2,
                "status": "SINUS LIFT likely",
                "recommendation": "Deficient bone height (4-6mm). Sinus floor lift likely indicated (transcrestal or lateral window graft)."
            },
            {
                "site_name": "Upper Left First Molar (Site #26)",
                "fdi_number": 26,
                "description": "Upper Left First Molar",
                "anatomical_site": "Upper Jaw (Maxilla - Sinus Floor)",
                "residual_bone_height_mm": 11.4,
                "status": "CLEAR",
                "recommendation": "Adequate vertical bone volume (>= 10mm). Standard implant fixture placement feasible."
            },
            {
                "site_name": "Lower Left First Molar (Site #36)",
                "fdi_number": 36,
                "description": "Lower Left First Molar",
                "anatomical_site": "Lower Jaw (Mandible - IAN Canal)",
                "residual_bone_height_mm": 6.4,
                "safety_buffer_mm": 2.0,
                "usable_bone_height_mm": 4.4,
                "status": "NERVE RISK",
                "recommendation": "Critical vertical deficiency (< 6-8mm). Hazardous proximity to IAN canal! Severe risk of nerve injury."
            }
        ],
        "preliminary_treatment_plan": [
            {
                "priority": "HIGH (Urgent Infection)",
                "target": "Tooth #46 (Lower Right First Molar)",
                "clinical_finding": "Periapical apical radiolucency / endodontic infection",
                "action_item": "Perform CBCT / vitality testing. Schedule endodontic re-treatment or apicoectomy."
            },
            {
                "priority": "MEDIUM (Restorative)",
                "target": "Tooth #14 (Upper Right First Premolar)",
                "clinical_finding": "Coronal radiolucent caries decay",
                "action_item": "Composite resin restoration or inlay/onlay preparation."
            },
            {
                "priority": "PLANNED (Surgical Augmentation)",
                "target": "Implant Site #16 (Upper Right First Molar)",
                "clinical_finding": "Maxillary sinus proximity (Residual Bone: 5.2mm, Status: SINUS LIFT likely)",
                "action_item": "Sinus floor lift indicated with simultaneous implant fixture placement."
            },
            {
                "priority": "HIGH CAUTION (Nerve Proximity)",
                "target": "Implant Site #36 (Lower Left First Molar)",
                "clinical_finding": "Mandibular canal safety buffer alert (Residual Bone: 6.4mm, Usable: 4.4mm)",
                "action_item": "Critical vertical deficiency. Mandatory CBCT nerve mapping."
            }
        ]
    }

    # Transform mock report
    clean_report = format_clean_report(mock_raw_report)

    # 1. Print Call Center Text Briefing
    text_briefing = generate_call_center_text_briefing(clean_report)
    print(text_briefing)

    # 2. Print Clean JSON Layout
    print("\n" + "=" * 76)
    print("      SAMPLE CLEAN JSON LAYOUT (STRIPPED OF RAW PROBABILITY DECIMALS)")
    print("=" * 76)
    print(json.dumps(clean_report, indent=2))
    print("=" * 76 + "\n")

    return clean_report


if __name__ == "__main__":
    run_mock_validation()
