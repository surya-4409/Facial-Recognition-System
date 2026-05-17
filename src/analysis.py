"""
analysis.py — Bias Analysis

Reads initial_audit.json, identifies the most biased group pairing
(largest FRR disparity), and writes structured analysis to analysis.json.
"""

import os
import json
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)
RESULTS_DIR = "results"


def find_most_biased_pairing(audit_results: Dict) -> Tuple[str, str, str, float]:
    """
    Find the pair of subgroups with the largest FRR disparity.

    Returns:
        (group_1, group_2, metric_name, disparity_value)
    """
    subgroups = {k: v for k, v in audit_results.items() if k != "overall"}
    if len(subgroups) < 2:
        return "unknown", "unknown", "frr_disparity", 0.0

    keys = list(subgroups.keys())
    max_disparity = 0.0
    worst_g1, worst_g2 = keys[0], keys[1]

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            frr_diff = abs(subgroups[keys[i]]["frr"] - subgroups[keys[j]]["frr"])
            if frr_diff > max_disparity:
                max_disparity = frr_diff
                # g1 = higher FRR (disadvantaged group)
                if subgroups[keys[i]]["frr"] > subgroups[keys[j]]["frr"]:
                    worst_g1, worst_g2 = keys[i], keys[j]
                else:
                    worst_g1, worst_g2 = keys[j], keys[i]

    return worst_g1, worst_g2, "frr_disparity", round(max_disparity, 4)


def find_highest_frr_group(audit_results: Dict) -> Tuple[str, float]:
    """Find the subgroup with the highest FRR."""
    subgroups = {k: v for k, v in audit_results.items() if k != "overall"}
    if not subgroups:
        return "unknown", 0.0
    worst = max(subgroups.items(), key=lambda x: x[1]["frr"])
    return worst[0], round(worst[1]["frr"], 4)


def generate_hypotheses(group_1: str, overall_frr: float) -> Dict[str, str]:
    """
    Generate data-level and model-level hypotheses based on the biased group.
    """
    parts = group_1.split("_") if "_" in group_1 else [group_1]
    gender = parts[0] if len(parts) > 0 else "unknown"
    age_bin = parts[1] if len(parts) > 1 else "unknown"
    skin_tone = parts[2] if len(parts) > 2 else "unknown"

    data_hyp = (
        f"The training data likely contains fewer images of {gender} individuals "
        f"in the {age_bin} age range with {skin_tone} skin tone. "
        f"FairFace, while balanced at the race level, may still exhibit within-group "
        f"imbalances (e.g., fewer elderly or very young subjects). Underrepresentation "
        f"leads to poorer feature generalization, causing higher FRR for this cohort. "
        f"Image quality factors (lighting, pose variation) may also disproportionately "
        f"affect {skin_tone}-skinned individuals due to camera sensor biases."
    )

    model_hyp = (
        f"The InceptionResnetV1 backbone was pretrained on VGGFace2, which, despite "
        f"its scale (3.3M images), inherits demographic imbalances from internet-sourced "
        f"data (skewed toward younger, lighter-skinned individuals). The learned feature "
        f"space may encode texture and lighting patterns that generalize poorly to "
        f"{skin_tone} skin tones or {age_bin}-aged faces. The model may implicitly learn "
        f"age-correlated features (e.g., wrinkle patterns) that do not provide stable "
        f"identity signals across pose and lighting variation."
    )

    return {"data_level": data_hyp, "model_level": model_hyp}


def run_analysis(
    initial_audit_path: str = None,
    output_path: str = None,
) -> Dict:
    """
    Load initial audit results and produce structured analysis.json.
    """
    if initial_audit_path is None:
        initial_audit_path = os.path.join(RESULTS_DIR, "initial_audit.json")
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "analysis.json")

    with open(initial_audit_path) as f:
        audit_results = json.load(f)

    overall_frr = audit_results.get("overall", {}).get("frr", 0.0)
    overall_far = audit_results.get("overall", {}).get("far", 0.0)

    group_1, group_2, metric, value = find_most_biased_pairing(audit_results)
    worst_group, worst_frr = find_highest_frr_group(audit_results)
    hypotheses = generate_hypotheses(group_1, overall_frr)

    # Compute additional stats
    subgroups = {k: v for k, v in audit_results.items() if k != "overall"}
    frrs = [v["frr"] for v in subgroups.values()]
    fars = [v["far"] for v in subgroups.values()]

    analysis = {
        "summary": {
            "total_subgroups_analyzed": len(subgroups),
            "overall_far": overall_far,
            "overall_frr": overall_frr,
            "worst_performing_group": worst_group,
            "worst_frr": worst_frr,
            "best_performing_group": min(subgroups.items(), key=lambda x: x[1]["frr"])[0],
            "best_frr": round(min(frrs), 4),
            "mean_frr_across_groups": round(float(sum(frrs) / len(frrs)), 4) if frrs else 0.0,
            "std_frr_across_groups": round(float(
                (sum((x - sum(frrs)/len(frrs))**2 for x in frrs) / len(frrs))**0.5
            ), 4) if frrs else 0.0,
        },
        "most_biased_pairing": {
            "group_1": group_1,
            "group_2": group_2,
            "metric": metric,
            "value": value,
            "interpretation": (
                f"Group '{group_1}' has a {value:.1%} higher FRR than '{group_2}'. "
                f"This means the system is {value:.1%} more likely to incorrectly "
                f"reject a legitimate match for individuals in group '{group_1}', "
                f"potentially causing disproportionate access denial."
            ),
        },
        "hypothesized_causes": hypotheses,
        "regulatory_implications": (
            "Under EU AI Act provisions for high-risk biometric systems, "
            "a FRR disparity exceeding 5% between demographic groups would require "
            "documented mitigation measures before deployment. NYC Local Law 144 "
            "would mandate bias audits for employment-related applications. "
            "The identified disparity necessitates mitigation before deployment "
            "in any high-stakes context."
        ),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Analysis saved → {output_path}")
    logger.info(f"Most biased pairing: {group_1} vs {group_2} | FRR disparity: {value:.4f}")
    return analysis
