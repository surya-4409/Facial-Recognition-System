"""
main.py — Facial Verification Fairness Audit Pipeline

Orchestrates the full end-to-end pipeline:
  Phase 0: Environment setup and data loading
  Phase 1: Model preparation (train or quick-mode)
  Phase 2: Initial fairness audit
  Phase 3: Bias analysis
  Phase 4: Mitigation and re-audit
  Phase 5: PDF report generation

Usage:
  python main.py                          # Quick mode (default)
  QUICK_MODE=false python main.py         # Full training mode
  python main.py --quick                  # Force quick mode
  python main.py --no-quick               # Force training mode
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Facial Verification Fairness Audit Pipeline")
    parser.add_argument("--quick", action="store_true", default=None,
                        help="Use pretrained model without fine-tuning (fast)")
    parser.add_argument("--no-quick", dest="quick", action="store_false",
                        help="Fine-tune model with Triplet Loss")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to load (0=all, default from env)")
    parser.add_argument("--audit-pairs", type=int, default=None,
                        help="Pairs per group for audit")
    parser.add_argument("--data-dir", type=str, default="data")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
def get_config(args) -> dict:
    # Environment variables take priority, then CLI args, then defaults
    quick_env = os.environ.get("QUICK_MODE", "true").lower() != "false"
    quick = args.quick if args.quick is not None else quick_env

    return {
        "quick_mode": quick,
        "epochs":     args.epochs     or int(os.environ.get("EPOCHS", "3")),
        "batch_size": args.batch_size or int(os.environ.get("BATCH_SIZE", "32")),
        "max_images": args.max_images if args.max_images is not None
                      else int(os.environ.get("MAX_IMAGES", "5000")),
        "audit_pairs":args.audit_pairs or int(os.environ.get("AUDIT_PAIRS_PER_GROUP", "150")),
        "data_dir":   args.data_dir,
        "model_path": "artifacts/model.pt",
        "device":     torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }


# ─────────────────────────────────────────────────────────────
# Directory Setup
# ─────────────────────────────────────────────────────────────
def setup_directories():
    for d in ["data", "artifacts", "results", "submission", "src"]:
        Path(d).mkdir(exist_ok=True)

    # Ensure src is a package
    src_init = Path("src/__init__.py")
    if not src_init.exists():
        src_init.write_text('"""Facial Verification Fairness Audit — Source Package"""\n')


# ─────────────────────────────────────────────────────────────
# Demographics Definition
# ─────────────────────────────────────────────────────────────
DEMOGRAPHICS = {
    "gender": ["Male", "Female"],
    "age_bins": {
        "0-19":  [0, 19],
        "20-39": [20, 39],
        "40-59": [40, 59],
        "60+":   [60, 150],
    },
    "skin_tone_scale": {
        "Light":  [1, 2],
        "Medium": [3, 4],
        "Dark":   [5, 6],
    },
}


def save_demographics():
    os.makedirs("results", exist_ok=True)
    with open("results/demographics.json", "w") as f:
        json.dump(DEMOGRAPHICS, f, indent=2)
    logger.info("Saved results/demographics.json")


# ─────────────────────────────────────────────────────────────
# Pipeline Stages
# ─────────────────────────────────────────────────────────────
def stage_banner(stage: str, description: str):
    width = 70
    logger.info("=" * width)
    logger.info(f"  {stage}")
    logger.info(f"  {description}")
    logger.info("=" * width)


def main():
    # Force UTF-8 output on Windows to prevent UnicodeEncodeError with special chars
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    start_total = time.time()
    args = parse_args()
    cfg = get_config(args)

    logger.info("=" * 70)
    logger.info("   FACIAL VERIFICATION FAIRNESS AUDIT PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Config: quick_mode={cfg['quick_mode']}, device={cfg['device']}, "
                f"max_images={cfg['max_images']}, audit_pairs={cfg['audit_pairs']}")

    # ── Phase 0: Setup ──────────────────────────────────────────────────────
    stage_banner("PHASE 0: ENVIRONMENT SETUP", "Creating directories and saving demographics")
    setup_directories()
    save_demographics()

    # ── Phase 0: Data Loading ───────────────────────────────────────────────
    stage_banner("PHASE 0: DATA LOADING", "Downloading/loading FairFace dataset")
    from src.data_loader import (
        download_fairface_labels,
        load_and_process_labels,
        split_dataset,
    )

    download_fairface_labels(data_dir=cfg["data_dir"])
    df = load_and_process_labels(data_dir=cfg["data_dir"], max_images=cfg["max_images"])

    if len(df) == 0:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    train_df, val_df, audit_df = split_dataset(df)
    logger.info(f"Data ready — train: {len(train_df)}, val: {len(val_df)}, audit: {len(audit_df)}")

    # ── Phase 1: Model ─────────────────────────────────────────────────────
    stage_banner("PHASE 1: MODEL PREPARATION", "Loading or training the face verification model")
    from src.train import prepare_model_quick_mode, train_model

    if cfg["quick_mode"]:
        logger.info("Quick mode: loading pretrained VGGFace2 model (no fine-tuning)")
        model = prepare_model_quick_mode(save_path=cfg["model_path"])
    else:
        logger.info("Training mode: fine-tuning with Triplet Loss")
        model = train_model(
            train_df=train_df,
            val_df=val_df,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            save_path=cfg["model_path"],
        )

    # ── Phase 2: Find Threshold ────────────────────────────────────────────
    stage_banner("PHASE 2: THRESHOLD SELECTION", "Finding optimal EER threshold on validation set")
    from src.audit import find_optimal_threshold

    threshold = find_optimal_threshold(model, val_df)
    logger.info(f"Global threshold: {threshold:.4f}")

    # ── Phase 2: Initial Audit ─────────────────────────────────────────────
    stage_banner("PHASE 2: INITIAL FAIRNESS AUDIT", "Disaggregated FAR/FRR across all subgroups")
    from src.audit import run_audit

    initial_results = run_audit(
        model=model,
        audit_df=audit_df,
        threshold=threshold,
        demographics=DEMOGRAPHICS,
        n_pairs=cfg["audit_pairs"],
        output_path="results/initial_audit.json",
    )
    logger.info(f"Initial audit complete: {len(initial_results)} subgroup entries")

    # ── Phase 3: Bias Analysis ─────────────────────────────────────────────
    stage_banner("PHASE 3: BIAS ANALYSIS", "Identifying disparities and hypothesizing causes")
    from src.analysis import run_analysis

    analysis = run_analysis(
        initial_audit_path="results/initial_audit.json",
        output_path="results/analysis.json",
    )
    bias_pair = analysis.get("most_biased_pairing", {})
    logger.info(
        f"Most biased pairing: {bias_pair.get('group_1')} vs {bias_pair.get('group_2')} "
        f"(FRR disparity: {bias_pair.get('value', 0):.4f})"
    )

    # ── Phase 4: Mitigation & Re-Audit ────────────────────────────────────
    stage_banner("PHASE 4: BIAS MITIGATION", "Post-processing threshold calibration per subgroup")
    from src.mitigation import run_mitigated_audit

    mitigated_results, group_thresholds, overall_metrics = run_mitigated_audit(
        model=model,
        audit_df=audit_df,
        val_df=val_df,
        global_threshold=threshold,
        demographics=DEMOGRAPHICS,
        n_pairs=cfg["audit_pairs"],
    )

    fi = overall_metrics.get("fairness_improvement", {})
    logger.info(
        f"Mitigation complete — "
        f"FRR disparity reduction: {fi.get('frr_disparity_reduction', 0):.4f} | "
        f"Accuracy change: {fi.get('accuracy_change', 0):+.4f}"
    )

    # ── Phase 5: Report Generation ─────────────────────────────────────────
    stage_banner("PHASE 5: REPORT GENERATION", "Generating deployment readiness memo PDF")
    from src.report import generate_memo

    memo_path = generate_memo(output_path="submission/deployment_memo.pdf")
    logger.info(f"Memo generated: {memo_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_total
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("  Output files:")
    for path in [
        "results/demographics.json",
        "results/initial_audit.json",
        "results/analysis.json",
        "results/mitigated_audit.json",
        "results/overall_metrics.json",
        "artifacts/model.pt",
        "submission/deployment_memo.pdf",
    ]:
        exists = "[PASS]" if os.path.exists(path) else "[MISS]"
        size = f"({os.path.getsize(path)/1024:.1f} KB)" if os.path.exists(path) else ""
        logger.info(f"    {exists}  {path} {size}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
