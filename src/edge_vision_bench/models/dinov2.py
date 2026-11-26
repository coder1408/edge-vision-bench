"""DINOv2 Vision Transformer embedding wrapper with ONNX-safe attention export."""

from __future__ import annotations

from typing import Any

import timm
import timm.layers
import torch
import torch.nn.functional as F
from torch import nn

DINOV2_VARIANTS: dict[str, str] = {
    "dinov2-vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2-vitb14": "vit_base_patch14_dinov2.lvd142m",
}


def _normalize_dinov2_variant(variant: str) -> str:
    normalized = variant.strip().lower().replace("_", "-")
    if normalized not in DINOV2_VARIANTS:
        supported = ", ".join(sorted(DINOV2_VARIANTS))
        raise ValueError(f"Unknown DINOv2 variant {variant!r}. Supported variants: {supported}")
    return normalized


class DINOv2Model(nn.Module):
    """DINOv2 Vision Transformer returning L2-normalized float32 embeddings."""

    def __init__(
        self,
        variant: str = "dinov2-vits14",
        pretrained: bool = True,
        input_size: tuple[int, int] = (518, 518),
    ) -> None:
        super().__init__()
        canonical = _normalize_dinov2_variant(variant)
        self.variant = canonical
        self.timm_name = DINOV2_VARIANTS[canonical]
        self.input_size = input_size

        # Disable fused SDPA attention globally and per-block for ONNX opset 17 compatibility.
        if hasattr(timm.layers, "set_fused_attn"):
            timm.layers.set_fused_attn(False)

        self.encoder: nn.Module = timm.create_model(
            self.timm_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=input_size,
        )
        self._disable_fused_attention()
        self.embedding_dim: int = int(getattr(self.encoder, "num_features", 384))

    def _disable_fused_attention(self) -> None:
        module: Any
        for module in self.encoder.modules():
            if hasattr(module, "fused_attn"):
                module.fused_attn = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: torch.Tensor = self.encoder(x)
        normalized = F.normalize(feats, p=2, dim=-1, eps=1e-6)
        return normalized.float() if normalized.dtype != torch.float32 else normalized


def load_dinov2(
    variant: str = "dinov2-vits14",
    pretrained: bool = True,
    input_size: tuple[int, int] = (518, 518),
) -> DINOv2Model:
    """Instantiate a DINOv2 embedding model in evaluation mode.

    Args:
        variant: One of ``"dinov2-vits14"`` or ``"dinov2-vitb14"``.
        pretrained: If True, loads pretrained weights via ``timm``.
        input_size: Spatial input dimensions ``(H, W)``, divisible by patch size 14.

    Returns:
        ``DINOv2Model`` configured for ONNX-safe inference in evaluation mode.
    """
    model = DINOv2Model(variant=variant, pretrained=pretrained, input_size=input_size)
    model.eval()
    return model
