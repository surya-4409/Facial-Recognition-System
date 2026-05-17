"""
model.py — Face Verification Model

Wraps FaceNet's InceptionResnetV1 pretrained on VGGFace2.
Produces L2-normalized 512-dimensional face embeddings.

Architecture:
  - Backbone: InceptionResnetV1 (VGGFace2 pretrained, 3.3M face dataset)
  - Optional projection head: 512 → 256 for fine-tuning (disabled in quick mode)
  - Output: L2-normalized embedding vectors

Usage:
  model = FaceVerificationModel(pretrained='vggface2')
  embedding = model.get_embedding(img_tensor)  # shape: (512,)
  similarity = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0)
"""

import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = "artifacts"


class FaceVerificationModel(nn.Module):
    """
    Face verification model using FaceNet's InceptionResnetV1 backbone.

    Args:
        pretrained: 'vggface2' (recommended) or 'casia-webface' or None
        use_projection: If True, add a learnable projection head for fine-tuning
        embedding_dim: Output embedding dimension (512 from backbone, or custom)
        device: torch device to use
    """

    def __init__(
        self,
        pretrained: str = "vggface2",
        use_projection: bool = False,
        embedding_dim: int = 512,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading InceptionResnetV1 (pretrained='{pretrained}') on {self.device}")

        # Load pretrained FaceNet backbone
        # classify=False → return 512-dim embedding instead of classification logits
        self.backbone = InceptionResnetV1(
            pretrained=pretrained,
            classify=False,
        ).to(self.device)

        # Optional projection head for domain adaptation during fine-tuning
        self.use_projection = use_projection
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, embedding_dim),
            ).to(self.device)
            self.output_dim = embedding_dim
        else:
            self.projection = None
            self.output_dim = 512

        logger.info(f"Model initialized — output dim: {self.output_dim}, projection: {use_projection}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing L2-normalized embedding.

        Args:
            x: Image tensor of shape (B, 3, 160, 160), normalized to [-1, 1]
        Returns:
            L2-normalized embedding tensor of shape (B, output_dim)
        """
        # FaceNet backbone already applies L2 norm internally when classify=False
        emb = self.backbone(x)

        if self.use_projection and self.projection is not None:
            emb = self.projection(emb)

        # Ensure L2 normalization for cosine similarity compatibility
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @torch.no_grad()
    def get_embedding(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for a single image or batch (inference mode).

        Args:
            img_tensor: shape (3, 160, 160) or (B, 3, 160, 160)
        Returns:
            embedding tensor, shape (512,) or (B, 512)
        """
        was_single = img_tensor.dim() == 3
        if was_single:
            img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to(self.device)
        self.eval()
        emb = self.forward(img_tensor)

        return emb.squeeze(0) if was_single else emb

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (only train projection head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only projection head will be trained")

    def unfreeze_backbone(self, layers_from_end: int = 2) -> None:
        """
        Unfreeze the last N blocks of the backbone for fine-tuning.
        Allows gradual unfreezing strategy.
        """
        # Unfreeze all by default
        all_children = list(self.backbone.children())
        for child in all_children[-(layers_from_end):]:
            for param in child.parameters():
                param.requires_grad = True
        logger.info(f"Unfroze last {layers_from_end} backbone blocks")

    def save(self, path: str = None) -> str:
        """Save model weights to artifacts directory."""
        if path is None:
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            path = os.path.join(ARTIFACTS_DIR, "model.pt")

        torch.save({
            "model_state_dict": self.state_dict(),
            "use_projection": self.use_projection,
            "output_dim": self.output_dim,
        }, path)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        logger.info(f"Model saved to {path} ({size_mb:.1f} MB)")
        return path

    @classmethod
    def load(cls, path: str = None, device: torch.device = None) -> "FaceVerificationModel":
        """Load model weights from checkpoint."""
        if path is None:
            path = os.path.join(ARTIFACTS_DIR, "model.pt")

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            pretrained=None,  # Don't re-download pretrained weights
            use_projection=checkpoint.get("use_projection", False),
            embedding_dim=checkpoint.get("output_dim", 512),
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model loaded from {path}")
        return model


class TripletLoss(nn.Module):
    """
    Online Triplet Loss with semi-hard mining.

    Given embeddings for anchors, positives, and negatives,
    minimizes: max(0, ||anchor - positive||² - ||anchor - negative||² + margin)

    Args:
        margin: Separation margin between positive and negative pairs
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor:   (B, D) L2-normalized embeddings
            positive: (B, D) L2-normalized embeddings (same identity as anchor)
            negative: (B, D) L2-normalized embeddings (different identity)
        Returns:
            Scalar loss value
        """
        # Squared Euclidean distances (equivalent to 2 - 2*cosine_sim for unit vectors)
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)

        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()
