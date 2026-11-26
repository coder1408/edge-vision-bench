"""Preprocessing transforms for detection (YOLOX) and embedding (DINOv2) tracks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class LetterboxResult:
    """Geometry and tensor bundle produced by letterbox preprocessing."""

    tensor_nchw: np.ndarray
    scale: float
    pad_x: int
    pad_y: int
    orig_h: int
    orig_w: int


def _check_bgr_image(image_bgr: np.ndarray) -> tuple[int, int]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H, W, 3), got {image_bgr.shape!r}")
    h, w = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Image height and width must be positive, got ({h}, {w})")
    return h, w


def _check_target_size(input_size: tuple[int, int]) -> tuple[int, int]:
    if len(input_size) != 2:
        raise ValueError(f"Expected (H, W) input_size tuple, got {input_size!r}")
    th, tw = int(input_size[0]), int(input_size[1])
    if th <= 0 or tw <= 0:
        raise ValueError(f"Target input_size must be positive, got ({th}, {tw})")
    return th, tw


def letterbox_preprocess(
    image_bgr: np.ndarray,
    input_size: tuple[int, int] = (640, 640),
    pad_value: int = 114,
    normalize_imagenet: bool = False,
) -> LetterboxResult:
    """Aspect-preserving resize and centered padding for YOLOX models.

    When ``normalize_imagenet`` is False (default), retains BGR channel order in
    ``[0, 255]`` float32 format matching official Megvii YOLOX checkpoints.
    """
    orig_h, orig_w = _check_bgr_image(image_bgr)
    target_h, target_w = _check_target_size(input_size)

    scale = min(target_h / orig_h, target_w / orig_w)
    resized_w = min(target_w, max(1, int(round(orig_w * scale))))
    resized_h = min(target_h, max(1, int(round(orig_h * scale))))

    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2

    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pixels: np.ndarray
    if image_bgr.dtype == np.uint8:
        canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
        pixels = canvas.astype(np.float32)
    else:
        pixels = np.full((target_h, target_w, 3), float(pad_value), dtype=np.float32)
        pixels[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized.astype(np.float32)

    if normalize_imagenet:
        pixels = pixels[:, :, ::-1] / 255.0
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD

    chw = np.transpose(pixels, (2, 0, 1))
    tensor = np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)

    return LetterboxResult(
        tensor_nchw=tensor,
        scale=float(scale),
        pad_x=int(pad_x),
        pad_y=int(pad_y),
        orig_h=orig_h,
        orig_w=orig_w,
    )


def dinov2_preprocess(
    image_bgr: np.ndarray,
    input_size: tuple[int, int] = (518, 518),
) -> np.ndarray:
    """Resize and normalize a BGR image for DINOv2 ViT inference."""
    _check_bgr_image(image_bgr)
    target_h, target_w = _check_target_size(input_size)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    scaled = resized.astype(np.float32) / 255.0
    normalized = (scaled - IMAGENET_MEAN) / IMAGENET_STD

    chw = np.transpose(normalized, (2, 0, 1))
    return np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)


def preprocess_batch(
    images_bgr: list[np.ndarray],
    track: str,
    input_size: tuple[int, int],
    pad_value: int = 114,
    normalize_imagenet: bool = False,
) -> tuple[np.ndarray, list[LetterboxResult]]:
    """Preprocess a batch of BGR images for either detection or embedding tracks."""
    if not images_bgr:
        raise ValueError("images_bgr must not be empty")

    if track == "detection":
        metas = [
            letterbox_preprocess(
                img,
                input_size=input_size,
                pad_value=pad_value,
                normalize_imagenet=normalize_imagenet,
            )
            for img in images_bgr
        ]
        batch = np.ascontiguousarray(
            np.concatenate([m.tensor_nchw for m in metas], axis=0),
            dtype=np.float32,
        )
        return batch, metas

    if track == "embedding":
        tensors = [dinov2_preprocess(img, input_size=input_size) for img in images_bgr]
        batch = np.ascontiguousarray(np.concatenate(tensors, axis=0), dtype=np.float32)
        return batch, []

    raise ValueError(f"Unknown track {track!r}; expected 'detection' or 'embedding'")
