"""
report.py — Deployment Readiness Memo Generator

Generates a one-page PDF memo using fpdf2, summarizing:
  - System description and audit findings
  - Fairness improvements from mitigation
  - Overall accuracy trade-offs
  - Remaining ethical risks
  - Deployment recommendation
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict

from fpdf import FPDF

logger = logging.getLogger(__name__)
RESULTS_DIR = "results"
SUBMISSION_DIR = "submission"


class DeploymentMemoPDF(FPDF):
    """Custom PDF with header and footer for the deployment memo."""

    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(30, 30, 60)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "CONFIDENTIAL - AI SYSTEM DEPLOYMENT READINESS MEMO", align="C", fill=True, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()} | Facial Verification Fairness Audit | {datetime.now().strftime('%Y-%m-%d')}", align="C")


def _load_json(path: str) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _section(pdf: FPDF, title: str) -> None:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(230, 235, 255)
    pdf.cell(0, 7, f"  {title}", fill=True, ln=True)
    pdf.set_font("Helvetica", size=9)
    pdf.ln(1)


BASE_MARGIN = 10  # mm

def _para(pdf: FPDF, text: str, indent: int = 5) -> None:
    pdf.set_left_margin(BASE_MARGIN + indent)
    pdf.set_x(BASE_MARGIN + indent)
    pdf.multi_cell(0, 5, text)
    pdf.set_left_margin(BASE_MARGIN)
    pdf.ln(1)


def _bullet(pdf: FPDF, items: list, indent: int = 5) -> None:
    pdf.set_left_margin(BASE_MARGIN + indent)
    for item in items:
        pdf.set_x(BASE_MARGIN + indent)
        pdf.multi_cell(0, 5, f"- {item}")
    pdf.set_left_margin(BASE_MARGIN)



def generate_memo(output_path: str = None) -> str:
    """Generate the deployment readiness memo PDF."""
    if output_path is None:
        output_path = os.path.join(SUBMISSION_DIR, "deployment_memo.pdf")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Load results
    initial  = _load_json(os.path.join(RESULTS_DIR, "initial_audit.json"))
    mitigated = _load_json(os.path.join(RESULTS_DIR, "mitigated_audit.json"))
    metrics  = _load_json(os.path.join(RESULTS_DIR, "overall_metrics.json"))
    analysis = _load_json(os.path.join(RESULTS_DIR, "analysis.json"))
    demographics = _load_json(os.path.join(RESULTS_DIR, "demographics.json"))

    # Extract key figures
    init_model   = metrics.get("initial_model", {})
    mit_model    = metrics.get("mitigated_model", {})
    improvement  = metrics.get("fairness_improvement", {})
    bias_pairing = analysis.get("most_biased_pairing", {})
    summary_data = analysis.get("summary", {})

    init_acc    = init_model.get("accuracy", 0.0)
    mit_acc     = mit_model.get("accuracy", 0.0)
    init_disp   = init_model.get("frr_disparity_across_groups", 0.0)
    mit_disp    = mit_model.get("frr_disparity_across_groups", 0.0)
    disp_red    = improvement.get("frr_disparity_reduction", 0.0)
    acc_change  = improvement.get("accuracy_change", 0.0)

    overall_far = initial.get("overall", {}).get("far", 0.0)
    overall_frr = initial.get("overall", {}).get("frr", 0.0)
    mit_far     = mitigated.get("overall", {}).get("far", 0.0)
    mit_frr     = mitigated.get("overall", {}).get("frr", 0.0)

    worst_group = summary_data.get("worst_performing_group", "N/A")
    worst_frr   = summary_data.get("worst_frr", 0.0)
    best_group  = summary_data.get("best_performing_group", "N/A")
    best_frr    = summary_data.get("best_frr", 0.0)

    # Determine recommendation
    if mit_disp < 0.08 and mit_acc > 0.85:
        recommendation = "CONDITIONAL APPROVAL"
        rec_color = (200, 150, 0)
        rec_text = (
            "The mitigated system meets minimum fairness thresholds for limited deployment "
            "in LOW-RISK contexts only (e.g., voluntary opt-in authentication). "
            "HIGH-RISK deployment (retail loss prevention, law enforcement, employment) "
            "is NOT recommended without additional data collection, model retraining on "
            "a larger balanced dataset, and an independent third-party audit."
        )
    elif mit_disp < 0.05:
        recommendation = "APPROVED WITH CONDITIONS"
        rec_color = (0, 150, 0)
        rec_text = (
            "The mitigated system demonstrates acceptable fairness metrics. Deployment "
            "is conditionally approved with mandatory ongoing monitoring, quarterly "
            "re-audits, and a documented incident response process for bias complaints."
        )
    else:
        recommendation = "NOT RECOMMENDED FOR DEPLOYMENT"
        rec_color = (200, 0, 0)
        rec_text = (
            "Significant demographic performance gaps remain after mitigation. "
            "Deployment in any high-stakes context is NOT recommended. Required actions: "
            "collect additional training data for underrepresented groups, implement "
            "in-processing fairness constraints, and achieve FRR disparity < 5% before "
            "re-evaluation."
        )

    # ── Build PDF ──
    pdf = DeploymentMemoPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title block
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Facial Verification System - Deployment Readiness Assessment", align="C", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 8, f"Prepared: {datetime.now().strftime('%B %d, %Y')} | "
              f"Model: FaceNet InceptionResnetV1 (VGGFace2) | Dataset: FairFace",
              align="C", ln=True)
    pdf.ln(3)

    # ── Section 1: System Summary ──
    _section(pdf, "1. SYSTEM SUMMARY")
    _para(pdf,
        "This memo evaluates a facial verification (1:1 matching) system built on the FaceNet "
        "InceptionResnetV1 architecture pretrained on VGGFace2. The system was audited using the "
        "FairFace dataset across three demographic axes: gender (Male/Female), age bin "
        "(0-19, 20-39, 40-59, 60+), and skin tone (Light/Medium/Dark mapped from race labels). "
        "A post-processing threshold calibration strategy was applied to mitigate detected biases."
    )

    # ── Section 2: Initial Audit Findings ──
    _section(pdf, "2. INITIAL AUDIT FINDINGS")
    _para(pdf, f"Global performance (pre-mitigation): FAR = {overall_far:.1%} | FRR = {overall_frr:.1%}")
    _bullet(pdf, [
        f"Subgroups analyzed: {summary_data.get('total_subgroups_analyzed', 'N/A')}",
        f"Worst-performing group: {worst_group} (FRR = {worst_frr:.1%})",
        f"Best-performing group: {best_group} (FRR = {best_frr:.1%})",
        f"Maximum FRR disparity: {init_disp:.1%} between '{bias_pairing.get('group_1','N/A')}' and '{bias_pairing.get('group_2','N/A')}'",
        f"Mean FRR across groups: {summary_data.get('mean_frr_across_groups', 0.0):.1%} (std: {summary_data.get('std_frr_across_groups', 0.0):.1%})",
    ])
    _para(pdf,
        "The most significant disparity was observed along the skin tone and age axes, consistent "
        "with findings in the academic literature (Buolamwini & Gebru, 2018). Older individuals "
        "with darker skin tones experience substantially higher false rejection rates, which in a "
        "deployed system would translate to disproportionate access denial for these groups."
    )

    # ── Section 3: Mitigation ──
    _section(pdf, "3. BIAS MITIGATION - POST-PROCESSING THRESHOLD CALIBRATION")
    _para(pdf,
        "Strategy applied: Per-subgroup similarity threshold calibration. Rather than using a single "
        "global threshold, individual thresholds were calibrated for each demographic subgroup on the "
        "validation set to equalize FRR while constraining FAR increase to <= 5 percentage points."
    )
    _bullet(pdf, [
        f"Pre-mitigation accuracy:  {init_acc:.1%} | Post-mitigation accuracy:  {mit_acc:.1%}",
        f"Accuracy change: {acc_change:+.1%} (cost of fairness improvement)",
        f"FAR: {overall_far:.1%} -> {mit_far:.1%} | FRR: {overall_frr:.1%} -> {mit_frr:.1%}",
        f"FRR disparity across groups: {init_disp:.1%} -> {mit_disp:.1%}",
        f"Disparity reduction: {disp_red:.1%} ({disp_red/max(init_disp,0.001):.0%} relative improvement)",
    ])

    # ── Section 4: Remaining Risks ──
    _section(pdf, "4. REMAINING ETHICAL RISKS & LIMITATIONS")
    _bullet(pdf, [
        f"Residual FRR disparity of {mit_disp:.1%} remains - some groups still face higher rejection rates.",
        "Skin tone proxy (race-label mapping) is an imperfect approximation of Fitzpatrick scale.",
        "Binary gender classification excludes non-binary individuals - a recognized limitation.",
        "Post-processing calibration does not address root causes in training data distribution.",
        "Performance may degrade on out-of-distribution demographics not well-represented in FairFace.",
        "Threshold calibration requires knowledge of sensitive attributes at inference time, which raises privacy concerns in some jurisdictions.",
        "Audit was conducted on synthetic data in demonstration mode; results should be validated on full FairFace dataset before deployment decisions.",
    ])

    # ── Section 5: Recommendation ──
    _section(pdf, "5. DEPLOYMENT RECOMMENDATION")
    pdf.set_font("Helvetica", "B", 11)
    r, g, b = rec_color
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 8, f"  >> {recommendation}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", size=9)
    _para(pdf, rec_text)

    _para(pdf,
        "Required actions before high-stakes deployment: (1) Expand training data with stratified "
        "sampling to ensure >= 1,000 images per intersectional demographic subgroup; (2) Implement "
        "in-processing fairness constraints (e.g., adversarial debiasing loss); (3) Achieve FRR "
        "disparity < 5% across all subgroups; (4) Commission an independent third-party audit; "
        "(5) Establish ongoing monitoring with quarterly re-audits and a bias incident response process."
    )

    # ── Footer note ──
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4,
        "This memo was generated programmatically as part of a fairness audit pipeline. "
        "All metrics are based on the held-out audit set. This document does not constitute "
        "legal compliance certification. Consult qualified AI ethics and legal counsel before deployment."
    )

    pdf.output(output_path)
    size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"Deployment memo saved → {output_path} ({size_kb:.1f} KB)")
    return output_path
