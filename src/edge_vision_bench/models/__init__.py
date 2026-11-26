"""Model definitions and checkpoint loaders for detection and embedding tracks."""

from edge_vision_bench.models.dinov2 import DINOV2_VARIANTS, DINOv2Model, load_dinov2
from edge_vision_bench.models.yolox import (
    YOLOX_VARIANTS,
    YOLOX_WEIGHT_URLS,
    YOLOXModel,
    load_yolox,
)

__all__ = [
    "DINOV2_VARIANTS",
    "DINOv2Model",
    "YOLOXModel",
    "YOLOX_VARIANTS",
    "YOLOX_WEIGHT_URLS",
    "load_dinov2",
    "load_yolox",
]
