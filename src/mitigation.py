"""
mitigation.py — Post-Processing Threshold Calibration

Bias mitigation strategy: per-demographic-group similarity threshold calibration.

Rationale:
  The pretrained FaceNet model has different score distributions for different
  demographic groups. A single global threshold creates disparate FRR across groups.
  By finding the per-group threshold that minimizes each group's FRR while keeping
  FAR below a target level, we reduce inter-group disparity without retraining.

Approach:
  For each demographic subgroup in the validation set:
    1. Collect positive and negative pair similarity scores
    2. Find the threshold minimizing max(FAR, FRR) for that group
       subject to FAR <= global_FAR + tolerance
    3. Apply these per-group thresholds during the re-audit

Trade-offs (reported in overall_metrics.json):
  - Reduced FRR disparity across groups
  - Slight increase in per-group FAR for underperforming groups
  - Overall accuracy may decrease slightly due to threshold fragmentation
"""

import os
import json
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.model import FaceVerificationModel
from src.audit import generate_pairs, compute_embeddings, compute_far_frr, _cosine_sim, run_audit

logger = logging.getLogger(__name__)
RESULTS_DIR = "results"


def calibrate_group_thresholds(
    model: FaceVerificationModel,
    val_df: pd.DataFrame,
    global_threshold: float,
    demographics: Dict,
    n_pairs: int = 100,
    far_tolerance: float = 0.05,
) -> Dict[str, float]:
    """
    Compute per-subgroup thresholds on the validation set.

    For each subgroup, finds the threshold that:
      - Minimizes FRR (reducing false rejections for that group)
      - Keeps FAR <= global_FAR_for_group + far_tolerance

    Args:
        model: Trained face verification model
        val_df: Validation DataFrame with demographic labels
        global_threshold: The global EER threshold from initial audit
        demographics: Demographics definition dict
        n_pairs: Pairs per group for calibration
        far_tolerance: Maximum allowed FAR increase per group

    Returns:
        Dictionary mapping subgroup key → calibrated threshold
    """
    logger.info("Calibrating per-group thresholds on validation set...")

    embeddings = compute_embeddings(model, val_df)
    pairs, labels = generate_pairs(val_df, n_pos=n_pairs, n_neg=n_pairs)
    scores = np.array([_cosine_sim(embeddings[a], embeddings[b]) for a, b, *_ in pairs])
    labels_arr = np.array(labels)

    # Group pairs by subgroup
    sg_data: Dict[str, Tuple[list, list]] = {}
    for i, (ia, ib, is_pos, sg_a, _) in enumerate(pairs):
        if sg_a not in sg_data:
            sg_data[sg_a] = ([], [])
        sg_data[sg_a][0].append(scores[i])
        sg_data[sg_a][1].append(labels_arr[i])

    group_thresholds = {}

    for sg, (sg_scores_list, sg_labels_list) in sg_data.items():
        sg_scores = np.array(sg_scores_list)
        sg_labels = np.array(sg_labels_list)

        if len(np.unique(sg_labels)) < 2 or len(sg_scores) < 10:
            logger.debug(f"  {sg}: insufficient data, using global threshold")
            group_thresholds[sg] = global_threshold
            continue

        # Compute global FAR for this group at global threshold
        global_far, global_frr = compute_far_frr(sg_scores, sg_labels, global_threshold)
        max_allowed_far = min(0.95, global_far + far_tolerance)

        # Search for optimal per-group threshold
        candidate_thresholds = np.linspace(
            max(sg_scores.min(), global_threshold - 0.3),
            min(sg_scores.max(), global_threshold + 0.1),
            num=100,
        )

        best_thr = global_threshold
        best_frr = global_frr

        for thr in candidate_thresholds:
            far, frr = compute_far_frr(sg_scores, sg_labels, thr)
            # Accept if FAR constraint is satisfied and FRR improves
            if far <= max_allowed_far and frr < best_frr:
                best_frr = frr
                best_thr = thr

        group_thresholds[sg] = round(float(best_thr), 4)
        new_far, new_frr = compute_far_frr(sg_scores, sg_labels, best_thr)
        logger.info(
            f"  {sg}: global_thr={global_threshold:.3f}→{best_thr:.3f} | "
            f"FRR: {global_frr:.3f}→{new_frr:.3f} | FAR: {global_far:.3f}→{new_far:.3f}"
        )

    # Fill missing subgroups
    genders = demographics.get("gender", ["Male", "Female"])
    age_bins = list(demographics.get("age_bins", {}).keys())
    skin_tones = list(demographics.get("skin_tone_scale", {}).keys())
    for g in genders:
        for ab in age_bins:
            for st in skin_tones:
                key = f"{g}_{ab}_{st}"
                if key not in group_thresholds:
                    group_thresholds[key] = global_threshold

    logger.info(f"Calibrated thresholds for {len(group_thresholds)} subgroups")
    return group_thresholds


def run_mitigated_audit(
    model: FaceVerificationModel,
    audit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    global_threshold: float,
    demographics: Dict,
    n_pairs: int = 200,
) -> Tuple[Dict, Dict, Dict]:
    """
    Full mitigation pipeline:
      1. Calibrate per-group thresholds on validation set
      2. Re-run audit with calibrated thresholds
      3. Compute overall metrics before and after mitigation

    Returns:
        (mitigated_results, group_thresholds, overall_metrics)
    """
    # Step 1: Calibrate
    group_thresholds = calibrate_group_thresholds(
        model, val_df, global_threshold, demographics
    )

    # Step 2: Re-run audit with per-group thresholds
    mitigated_path = os.path.join(RESULTS_DIR, "mitigated_audit.json")
    mitigated_results = run_audit(
        model=model,
        audit_df=audit_df,
        threshold=global_threshold,
        demographics=demographics,
        n_pairs=n_pairs,
        output_path=mitigated_path,
        group_thresholds=group_thresholds,
    )

    # Step 3: Load initial audit for comparison
    initial_path = os.path.join(RESULTS_DIR, "initial_audit.json")
    try:
        with open(initial_path) as f:
            initial_results = json.load(f)
    except FileNotFoundError:
        initial_results = {"overall": {"far": 0.0, "frr": 0.0}}

    # Compute overall accuracy metrics
    def _group_accuracy(audit_dict: Dict) -> float:
        """Balanced accuracy = 1 - average(FAR + FRR) / 2"""
        vals = [v for k, v in audit_dict.items() if k != "overall"]
        if not vals:
            return 0.0
        avg_far = np.mean([v["far"] for v in vals])
        avg_frr = np.mean([v["frr"] for v in vals])
        return round(1.0 - (avg_far + avg_frr) / 2, 4)

    def _frr_disparity(audit_dict: Dict) -> float:
        """Max FRR - Min FRR across subgroups."""
        frrs = [v["frr"] for k, v in audit_dict.items() if k != "overall"]
        return round(max(frrs) - min(frrs), 4) if frrs else 0.0

    initial_acc = _group_accuracy(initial_results)
    mitigated_acc = _group_accuracy(mitigated_results)
    initial_disp = _frr_disparity(initial_results)
    mitigated_disp = _frr_disparity(mitigated_results)

    # Compute effective threshold for mitigated model (mean of group thresholds)
    mean_mitigated_thr = round(float(np.mean(list(group_thresholds.values()))), 4)

    overall_metrics = {
        "initial_model": {
            "accuracy": initial_acc,
            "threshold": round(global_threshold, 4),
            "overall_far": initial_results.get("overall", {}).get("far", 0.0),
            "overall_frr": initial_results.get("overall", {}).get("frr", 0.0),
            "frr_disparity_across_groups": initial_disp,
        },
        "mitigated_model": {
            "accuracy": mitigated_acc,
            "threshold": mean_mitigated_thr,
            "overall_far": mitigated_results.get("overall", {}).get("far", 0.0),
            "overall_frr": mitigated_results.get("overall", {}).get("frr", 0.0),
            "frr_disparity_across_groups": mitigated_disp,
            "mitigation_strategy": "post_processing_threshold_calibration",
            "group_thresholds": group_thresholds,
        },
        "fairness_improvement": {
            "frr_disparity_reduction": round(initial_disp - mitigated_disp, 4),
            "accuracy_change": round(mitigated_acc - initial_acc, 4),
        },
    }

    metrics_path = os.path.join(RESULTS_DIR, "overall_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(overall_metrics, f, indent=2)
    logger.info(f"Overall metrics saved → {metrics_path}")
    logger.info(
        f"Accuracy: {initial_acc:.4f} → {mitigated_acc:.4f} | "
        f"FRR Disparity: {initial_disp:.4f} → {mitigated_disp:.4f}"
    )

    return mitigated_results, group_thresholds, overall_metrics
