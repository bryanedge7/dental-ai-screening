"""
app.py - Streamlit Web Dashboard for Dental AI Screening Pipeline
-----------------------------------------------------------------
Evaluates and visualizes panoramic radiographs (OPG) across:
  - Stage 1: Quadrant localization & FDI tooth inventory (present vs missing teeth)
  - Stage 2A: Multi-label pathology classification (Caries, Periapical Lesions, Crowns, RCTs)
  - Stage 2B: Anatomical landmark regression & Section 6 implant site clearance rules
  - Section 7.2 Plain-Language Call Center Reporting & Mandatory Legal Disclaimer
"""

import os
import sys
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

# Ensure UTF-8 output encoding across Windows shells
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Import Master Pipeline & Section 7.2 Formatter
from main_pipeline import DentalAIScreeningPipeline
from format_output import (
    format_clean_report,
    generate_call_center_text_briefing,
    MANDATORY_LEGAL_DISCLAIMER
)


# ---------------------------------------------------------------------------
# Streamlit Page Configuration & Custom Clinical CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dental AI Screening | Clinical Radiograph Evaluator",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for clinical callout cards, status badges, and legal disclaimer
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #E2E8F0 !important;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-clear {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-caution {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-sinus {
        background-color: #FFEDD5;
        color: #C2410C;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-nerve {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .legal-disclaimer-box {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 20px;
        margin-top: 35px;
        margin-bottom: 25px;
        text-align: center;
    }
    .legal-disclaimer-text {
        font-size: 0.95rem;
        font-weight: 600;
        color: #92400E;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline Model Loader (Cached across Streamlit browser sessions)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Dental AI models (YOLOv8, EfficientNet-B3, U-Net)...")
def load_screening_pipeline():
    """Initializes and caches the multi-stage inference pipeline in memory."""
    stage1_weights = "runs/segment/runs/stage1_train/quadrant_segmentation/weights/best.pt"
    stage2a_weights = "runs/stage2a_train/weights/best_stage2a.pt"
    
    pipeline = DentalAIScreeningPipeline(
        stage1_weights=stage1_weights,
        stage2a_weights=stage2a_weights,
        pixel_spacing_mm=0.10,
        device="cpu"
    )
    return pipeline


def get_available_sample_scans() -> List[str]:
    """Finds sample panoramic radiographs in data/processed/images/train."""
    sample_dir = os.path.join("data", "processed", "images", "train")
    if os.path.exists(sample_dir):
        files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return sorted(files)
    return []


# ---------------------------------------------------------------------------
# Main Application Layout
# ---------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">🦷 Dental AI Screening Diagnostic Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Automated panoramic X-ray evaluation: <b>Stage 1</b> Quadrant & FDI Inventory | '
        '<b>Stage 2A</b> Pathology & Restoration Classification | '
        '<b>Stage 2B</b> Landmark Regression & Implant Feasibility'
        '</div>',
        unsafe_allow_html=True
    )

    # Sidebar: Image Acquisition & Settings
    st.sidebar.header("📁 Radiograph Acquisition")

    input_mode = st.sidebar.radio(
        "Choose Input Source:",
        ["Preloaded Sample Scan", "Upload New Panoramic X-ray"],
        index=0
    )

    selected_image_path = None
    input_display_name = ""

    sample_scans = get_available_sample_scans()

    if input_mode == "Preloaded Sample Scan":
        if sample_scans:
            selected_scan = st.sidebar.selectbox(
                "Select a Panoramic X-ray:",
                sample_scans,
                index=0
            )
            selected_image_path = os.path.join("data", "processed", "images", "train", selected_scan)
            input_display_name = selected_scan
        else:
            st.sidebar.warning("No sample scans found in data/processed/images/train.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Panoramic X-ray (PNG, JPG, JPEG):",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            # Save uploaded file temporarily for pipeline consumption
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"uploaded_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_image_path = temp_path
            input_display_name = uploaded_file.name

    # Sidebar Parameters
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Clinical Settings")
    pixel_spacing = st.sidebar.number_input(
        "Spatial Calibration (mm/pixel):",
        min_value=0.05,
        max_value=0.25,
        value=0.10,
        step=0.01,
        help="Standard panoramic radiograph calibration ratio."
    )

    st.sidebar.info("Compute Engine: **AMD Ryzen CPU** (Torch intra-op multi-threading)")

    run_screening_btn = st.sidebar.button("🚀 Run AI Diagnostic Screening", type="primary", use_container_width=True)

    # Initialize Session State to hold inference results
    if "latest_report" not in st.session_state:
        st.session_state.latest_report = None
        st.session_state.latest_visual = None
        st.session_state.latest_file = None

    # Load cached AI pipeline
    pipeline = load_screening_pipeline()

    # Trigger inference on button click or if no report is cached
    if run_screening_btn and selected_image_path:
        with st.spinner(f"Executing Multi-Stage Vision Pipeline on '{input_display_name}'..."):
            output_dir = os.path.join("runs", "pipeline_output")
            pipeline.pixel_spacing_mm = pixel_spacing
            
            t0 = time.time()
            clean_report = pipeline.process_panoramic_scan(
                image_path=selected_image_path,
                output_dir=output_dir,
                save_visual=True
            )
            elapsed = time.time() - t0

            st.session_state.latest_report = clean_report
            st.session_state.latest_visual = os.path.join(output_dir, "annotated_screening.jpg")
            st.session_state.latest_file = input_display_name
            st.sidebar.success(f"Screening complete in {elapsed:.2f}s!")

    # Display Dashboard Results if available
    report = st.session_state.latest_report

    if report is not None:
        dent = report.get("patient_dentition_summary", {})
        patho = report.get("pathology_overview", {})
        implants = report.get("implant_site_clearance_flags", [])
        teeth = report.get("tooth_by_tooth_records", [])
        plan = report.get("recommended_treatment_steps", [])

        # -------------------------------------------------------------------
        # 1. Top KPI Summary Metrics Bar
        # -------------------------------------------------------------------
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #0284C7;">{dent.get('total_teeth_present', 0)} / 32</div>
                <div class="metric-lbl">Teeth Present</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #64748B;">{dent.get('total_teeth_missing', 0)}</div>
                <div class="metric-lbl">Missing Teeth (Gaps)</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #EA580C;">{patho.get('teeth_with_suspected_cavities', 0)}</div>
                <div class="metric-lbl">Active Cavities / Caries</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #DC2626;">{patho.get('teeth_with_suspected_infections', 0)}</div>
                <div class="metric-lbl">Apical Infections (Lesions)</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #4F46E5;">{patho.get('teeth_with_existing_crowns', 0) + patho.get('teeth_with_prior_root_canals', 0)}</div>
                <div class="metric-lbl">Restorations (Crown / RCT)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------------
        # 2. Radiograph Visual Overlays: Side-by-Side View
        # -------------------------------------------------------------------
        st.subheader("🖼️ Panoramic Radiograph & AI Screening Overlay")
        col_img_left, col_img_right = st.columns(2)

        with col_img_left:
            st.caption(f"**Original Panoramic X-ray** (`{st.session_state.latest_file}`)")
            if selected_image_path and os.path.exists(selected_image_path):
                st.image(selected_image_path, use_container_width=True)
            else:
                st.info("Original image source not available.")

        with col_img_right:
            st.caption("**Annotated AI Screening Overlay** (Green: Bone Crest | Cyan: Sinus Floor | Yellow: IAN Canal)")
            vis_path = st.session_state.latest_visual
            if vis_path and os.path.exists(vis_path):
                st.image(vis_path, use_container_width=True)
            else:
                st.info("Annotated overlay being processed...")

        st.markdown("---")

        # -------------------------------------------------------------------
        # 3. Interactive Deep Dive Tabs
        # -------------------------------------------------------------------
        tab_implants, tab_teeth, tab_plan, tab_callcenter, tab_json = st.tabs([
            "🦴 Implant Feasibility & Section 6 Clearances",
            "🦷 Tooth-by-Tooth Pathology Inventory",
            "📋 Prioritized Treatment Action Plan",
            "📞 Call Center Staff Briefing",
            "📄 Clean JSON Inspector (Section 7.2)"
        ])

        # TAB 1: Implant Feasibility
        with tab_implants:
            st.markdown("### Section 6 Implant Site Assessment & Anatomical Clearance")
            st.markdown(
                "Evaluates residual vertical bone height from alveolar bone crest to maxillary sinus floor (Upper Jaw) "
                "or mandibular nerve canal (Lower Jaw with mandatory **2.0 mm safety buffer**)."
            )

            if implants:
                for site in implants:
                    flag = site.get("status_flag", "CAUTION")
                    fdi = site.get("site_number")
                    name = site.get("site_name", f"Site #{fdi}")
                    region = site.get("jaw_region", "")
                    talking = site.get("agent_talking_point", "")
                    action = site.get("clinical_action", "")

                    badge_class = {
                        "CLEAR": "badge-clear",
                        "CAUTION": "badge-caution",
                        "SINUS LIFT": "badge-sinus",
                        "NERVE RISK": "badge-nerve"
                    }.get(flag, "badge-caution")

                    with st.container():
                        st.markdown(f"""
                        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:14px; margin-bottom:12px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <span style="font-size:1.1rem; font-weight:700; color:#1E293B;">Tooth #{fdi} — {name}</span>
                                <span class="{badge_class}">{flag}</span>
                            </div>
                            <div style="font-size:0.85rem; color:#64748B; margin-bottom:6px;"><b>Anatomical Region:</b> {region}</div>
                            <div style="font-size:0.92rem; color:#334155; margin-bottom:4px;"><b>Intake Note:</b> {talking}</div>
                            <div style="font-size:0.88rem; color:#0369A1;"><b>Clinical Surgical Guidance:</b> {action}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No edentulous implant candidate sites detected. Full dentition present.")

        # TAB 2: Tooth-by-Tooth Pathology Inventory
        with tab_teeth:
            st.markdown("### Section 7.2 Plain-Language Tooth Inventory")
            st.markdown("All raw probability decimals stripped. Only clear diagnostic text status flags exposed.")

            # Filter controls
            filter_choice = st.selectbox(
                "Filter by Tooth Status:",
                ["All Teeth", "Urgent Attention Needed", "Treatment Recommended", "Restored / Monitor", "Healthy / Clear"]
            )

            filtered_teeth = teeth
            if filter_choice == "Urgent Attention Needed":
                filtered_teeth = [t for t in teeth if t.get("overall_status") == "URGENT ATTENTION NEEDED"]
            elif filter_choice == "Treatment Recommended":
                filtered_teeth = [t for t in teeth if t.get("overall_status") == "TREATMENT RECOMMENDED"]
            elif filter_choice == "Restored / Monitor":
                filtered_teeth = [t for t in teeth if t.get("overall_status") == "RESTORED / MONITOR"]
            elif filter_choice == "Healthy / Clear":
                filtered_teeth = [t for t in teeth if t.get("overall_status") == "HEALTHY / CLEAR"]

            # Display table
            table_data = []
            for t in filtered_teeth:
                table_data.append({
                    "FDI #": t.get("tooth_number"),
                    "Tooth Description": t.get("tooth_name"),
                    "Overall Status": t.get("overall_status"),
                    "Cavities (Caries)": t.get("caries_decay"),
                    "Apical Infection": t.get("root_infection"),
                    "Crown": t.get("prosthetic_crown"),
                    "Root Canal": t.get("root_canal_treatment"),
                    "Clinical Flags": ", ".join(t.get("clinical_flags", []))
                })

            if table_data:
                st.dataframe(table_data, use_container_width=True, hide_index=True)
            else:
                st.info(f"No teeth matched the filter criteria '{filter_choice}'.")

        # TAB 3: Prioritized Treatment Plan
        with tab_plan:
            st.markdown("### Actionable Preliminary Dental Treatment Plan")
            st.markdown("Consolidated multi-stage diagnostic priorities generated for clinician and patient review.")

            if plan:
                for idx, p in enumerate(plan, start=1):
                    urgency = p.get("urgency", "ROUTINE")
                    color = "#EF4444" if "HIGH" in urgency or "Urgent" in urgency else ("#F59E0B" if "MEDIUM" in urgency else "#3B82F6")
                    
                    st.markdown(f"""
                    <div style="border-left: 4px solid {color}; background:#F8FAFC; padding:12px 16px; margin-bottom:10px; border-radius:0 6px 6px 0;">
                        <span style="font-weight:700; color:{color}; font-size:0.85rem;">[{idx}] {urgency}</span>
                        <div style="font-weight:700; font-size:1.02rem; color:#1E293B; margin-top:2px;">{p.get('target')}</div>
                        <div style="font-size:0.9rem; color:#475569; margin-top:3px;"><b>Finding:</b> {p.get('issue')}</div>
                        <div style="font-size:0.9rem; color:#0F172A; margin-top:2px;"><b>Proposed Action:</b> {p.get('proposed_step')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No actionable treatment procedures required.")

        # TAB 4: Call Center Staff Briefing
        with tab_callcenter:
            st.markdown("### 📞 Call Center & Front Desk Patient Intake Briefing")
            st.caption("Standardized plain-language script for scheduling coordinators discussing findings with patients.")
            text_script = generate_call_center_text_briefing(report)
            st.code(text_script, language="markdown")

        # TAB 5: Clean JSON Inspector
        with tab_json:
            st.markdown("### Section 7.2 Structured JSON Report")
            st.caption("Complete clean JSON layout with zero raw probability floats, ready for EHR/PMS API export.")
            st.json(report)
            
            # Download JSON button
            json_str = json.dumps(report, indent=2)
            st.download_button(
                label="📥 Download Clean report.json",
                data=json_str,
                file_name=f"report_{st.session_state.latest_file}.json",
                mime="application/json"
            )

    else:
        st.info("👋 Welcome! Please select a sample panoramic radiograph or upload an X-ray scan on the sidebar, then click **Run AI Diagnostic Screening**.")

    # -------------------------------------------------------------------
    # 4. Mandatory Section 7.2 Legal Screening Disclaimer
    # -------------------------------------------------------------------
    st.markdown(f"""
    <div class="legal-disclaimer-box">
        <p class="legal-disclaimer-text">
            ⚠️ MANDATORY LEGAL NOTICE: {MANDATORY_LEGAL_DISCLAIMER}
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
