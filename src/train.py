"""
train.py — Model Training with Triplet Loss

Fine-tunes the FaceNet backbone on the FairFace training split using
online triplet loss with semi-hard negative mining.

Pipeline:
  1. Load train/val DataFrames
  2. Create TripletDataset for training, FairFaceDataset for validation
  3. Fine-tune with Adam optimizer and cosine LR schedule
  4. Validate each epoch: compute EER threshold and accuracy
  5. Save best model to artifacts/model.pt

In QUICK_MODE, this module is skipped and the pretrained VGGFace2
model is used directly (zero-shot evaluation).
"""

import os
import logging
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.model import FaceVerificationModel, TripletLoss
from src.data_loader import FairFaceDataset, TripletDataset, get_transform

logger = logging.getLogger(__name__)


def compute_val_accuracy(
    model: FaceVerificationModel,
    val_df: pd.DataFrame,
    threshold: float = 0.6,
    n_pairs: int = 1000,
    device: torch.device = None,
) -> Tuple[float, float]:
    """
    Estimate verification accuracy on the validation set.

    Generates random positive and negative pairs from val_df,
    computes cosine similarity, and measures accuracy at the given threshold.

    Returns:
        (accuracy, eer_threshold)
    """
    device = device or model.device
    model.eval()

    dataset = FairFaceDataset(val_df, transform=get_transform(train=False))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Collect all embeddings
    all_embs = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for imgs, labels, ids, _ in tqdm(loader, desc="  Val embeddings", leave=False):
            embs = model(imgs.to(device))
            all_embs.append(embs.cpu())
            all_labels.extend(labels.tolist())
            all_ids.extend(ids)

    all_embs = torch.cat(all_embs, dim=0)
    all_labels = np.array(all_labels)

    # Sample pairs
    rng = np.random.default_rng(42)
    n_half = n_pairs // 2
    scores, gt_labels = [], []

    unique_labels = np.unique(all_labels)
    # Positive pairs
    pos_count = 0
    for lbl in rng.permutation(unique_labels):
        idxs = np.where(all_labels == lbl)[0]
        if len(idxs) >= 2:
            i, j = rng.choice(idxs, size=2, replace=False)
            sim = torch.nn.functional.cosine_similarity(
                all_embs[i].unsqueeze(0), all_embs[j].unsqueeze(0)
            ).item()
            scores.append(sim)
            gt_labels.append(1)
            pos_count += 1
            if pos_count >= n_half:
                break

    # Negative pairs
    neg_count = 0
    all_idxs = np.arange(len(all_labels))
    for _ in range(n_half * 3):  # sample more to get n_half valid negatives
        i, j = rng.choice(all_idxs, size=2, replace=False)
        if all_labels[i] != all_labels[j]:
            sim = torch.nn.functional.cosine_similarity(
                all_embs[i].unsqueeze(0), all_embs[j].unsqueeze(0)
            ).item()
            scores.append(sim)
            gt_labels.append(0)
            neg_count += 1
            if neg_count >= n_half:
                break

    scores = np.array(scores)
    gt_labels = np.array(gt_labels)

    # Find EER threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(gt_labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer_threshold = thresholds[eer_idx]

    # Compute accuracy at given threshold
    predictions = (scores >= threshold).astype(int)
    accuracy = (predictions == gt_labels).mean()

    logger.info(f"  Val accuracy @ {threshold:.3f}: {accuracy:.4f} | EER threshold: {eer_threshold:.4f}")
    return float(accuracy), float(eer_threshold)


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-4,
    margin: float = 0.2,
    device: torch.device = None,
    save_path: str = "artifacts/model.pt",
) -> FaceVerificationModel:
    """
    Fine-tune FaceNet on FairFace training split using Triplet Loss.

    Args:
        train_df: Training DataFrame from data_loader
        val_df: Validation DataFrame
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate for Adam
        margin: Triplet loss margin
        device: torch.device
        save_path: Where to save the final model

    Returns:
        Trained FaceVerificationModel
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Training on device: {device}")
    logger.info(f"Train set: {len(train_df)} images, Val set: {len(val_df)} images")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}, Margin: {margin}")

    # Initialize model
    model = FaceVerificationModel(
        pretrained="vggface2",
        use_projection=False,
        device=device,
    )

    # Freeze backbone initially, only train last block + projection
    model.freeze_backbone()
    model.unfreeze_backbone(layers_from_end=3)

    # Data
    train_dataset = TripletDataset(train_df, transform=get_transform(train=True))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Optimization
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = TripletLoss(margin=margin)

    best_accuracy = 0.0
    best_threshold = 0.6

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()

        for batch_idx, (anchors, positives, negatives, _) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        ):
            anchors  = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            emb_a = model(anchors)
            emb_p = model(positives)
            emb_n = model(negatives)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - start_time

        logger.info(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Validate
        accuracy, eer_threshold = compute_val_accuracy(
            model, val_df, threshold=best_threshold, device=device
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = eer_threshold
            model.save(save_path)
            logger.info(f"  ✓ New best model saved (accuracy={best_accuracy:.4f})")

    logger.info(f"Training complete. Best accuracy: {best_accuracy:.4f}, threshold: {best_threshold:.4f}")
    return model


def prepare_model_quick_mode(save_path: str = "artifacts/model.pt") -> FaceVerificationModel:
    """
    In QUICK_MODE: Load the pretrained FaceNet model without fine-tuning,
    save it as the artifact, and return it.
    """
    logger.info("QUICK_MODE: Using pretrained VGGFace2 model without fine-tuning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FaceVerificationModel(
        pretrained="vggface2",
        use_projection=False,
        device=device,
    )
    os.makedirs("artifacts", exist_ok=True)
    model.save(save_path)
    return model
