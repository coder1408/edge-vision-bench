"""YOLOX output decoding, pure-NumPy class-aware NMS, and coordinate inversion."""

from __future__ import annotations

import numpy as np

from edge_vision_bench.pipeline.preproc import LetterboxResult
from edge_vision_bench.schemas import DetectionBox

_COCO_CLASS_NAMES_CSV = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,"
    "fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,"
    "elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,"
    "skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
    "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,"
    "apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
    "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
    "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,"
    "scissors,teddy bear,hair drier,toothbrush"
)
COCO_CLASSES: tuple[str, ...] = tuple(_COCO_CLASS_NAMES_CSV.split(","))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _build_yolox_grids(
    input_size: tuple[int, int],
    strides: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    img_h, img_w = int(input_size[0]), int(input_size[1])
    grid_parts: list[np.ndarray] = []
    stride_parts: list[np.ndarray] = []

    for stride in strides:
        if stride <= 0:
            raise ValueError(f"Strides must be positive integers, got {stride}")
        gh = img_h // stride
        gw = img_w // stride
        gy, gx = np.meshgrid(
            np.arange(gh, dtype=np.float32),
            np.arange(gw, dtype=np.float32),
            indexing="ij",
        )
        grid = np.stack((gx, gy), axis=-1).reshape(1, gh * gw, 2)
        stride_vec = np.full((1, gh * gw, 1), float(stride), dtype=np.float32)
        grid_parts.append(grid)
        stride_parts.append(stride_vec)

    grids = np.concatenate(grid_parts, axis=1)
    expanded_strides = np.concatenate(stride_parts, axis=1)
    return grids, expanded_strides


def decode_yolox_outputs(
    raw_output: np.ndarray,
    input_size: tuple[int, int] = (640, 640),
    strides: tuple[int, ...] = (8, 16, 32),
    already_decoded: bool = True,
) -> np.ndarray:
    """Normalize YOLOX head predictions into ``(B, N, 5 + C)`` pixel-space tensors.

    Output layout per anchor: ``[cx, cy, w, h, obj_conf, cls_prob_0, ...]`` where
    box geometry is in ``input_size`` pixel coordinates and confidences lie in ``[0, 1]``.
    """
    if raw_output.ndim == 2:
        decoded = np.array(raw_output[np.newaxis, ...], dtype=np.float32, copy=True)
    elif raw_output.ndim == 3:
        decoded = np.array(raw_output, dtype=np.float32, copy=True)
    else:
        raise ValueError(
            f"raw_output must have shape (B, N, 5+C) or (N, 5+C), got {raw_output.shape!r}"
        )

    if decoded.shape[-1] < 6:
        raise ValueError(
            f"Expected last dimension >= 6 (4 box + 1 obj + >=1 cls), got {decoded.shape[-1]}"
        )

    if already_decoded:
        return decoded

    grids, expanded_strides = _build_yolox_grids(input_size=input_size, strides=strides)
    if decoded.shape[1] != grids.shape[1]:
        raise ValueError(
            f"Anchor count mismatch for input_size={input_size} and strides={strides}: "
            f"got {decoded.shape[1]}, expected {grids.shape[1]}"
        )

    decoded[..., 0:2] = (decoded[..., 0:2] + grids) * expanded_strides
    decoded[..., 2:4] = np.exp(np.clip(decoded[..., 2:4], -20.0, 20.0)) * expanded_strides

    scores_slice = decoded[..., 4:]
    if scores_slice.size > 0 and (
        float(scores_slice.min()) < 0.0 or float(scores_slice.max()) > 1.0
    ):
        decoded[..., 4:] = _sigmoid(scores_slice)

    return decoded


def _nms_single_class_shifted(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    nms_threshold: float,
) -> np.ndarray:
    """Class-aware greedy NMS via per-class coordinate offsetting in pure NumPy."""
    if boxes_xyxy.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    coord_span = float(np.max(boxes_xyxy) - np.min(boxes_xyxy)) + 16.0
    class_stride = max(4096.0, coord_span)
    offsets = class_ids.astype(np.float64) * class_stride

    x1 = boxes_xyxy[:, 0].astype(np.float64) + offsets
    y1 = boxes_xyxy[:, 1].astype(np.float64) + offsets
    x2 = boxes_xyxy[:, 2].astype(np.float64) + offsets
    y2 = boxes_xyxy[:, 3].astype(np.float64) + offsets

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores, kind="mergesort")[::-1]

    kept: list[int] = []
    while order.size > 0:
        top = int(order[0])
        kept.append(top)
        if order.size == 1:
            break

        rest = order[1:]
        inter_x1 = np.maximum(x1[top], x1[rest])
        inter_y1 = np.maximum(y1[top], y1[rest])
        inter_x2 = np.minimum(x2[top], x2[rest])
        inter_y2 = np.minimum(y2[top], y2[rest])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = areas[top] + areas[rest] - inter_area
        iou = np.where(union_area > 0.0, inter_area / union_area, 0.0)
        order = rest[iou <= nms_threshold]

    return np.asarray(kept, dtype=np.int64)


def batched_nms(
    decoded: np.ndarray,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
) -> list[np.ndarray]:
    """Filter decoded YOLOX predictions by confidence and apply class-aware NMS.

    Returns:
        List of ``(M, 6)`` float32 arrays ``[x1, y1, x2, y2, score, class_id]``,
        one entry per batch item.
    """
    if decoded.ndim == 2:
        batch_preds = decoded[np.newaxis, ...]
    elif decoded.ndim == 3:
        batch_preds = decoded
    else:
        raise ValueError(f"decoded must have shape (B, N, 5+C) or (N, 5+C), got {decoded.shape!r}")

    results: list[np.ndarray] = []
    for pred in batch_preds:
        if pred.shape[0] == 0:
            results.append(np.empty((0, 6), dtype=np.float32))
            continue

        obj_conf = pred[:, 4]
        cls_probs = pred[:, 5:]
        class_ids = np.argmax(cls_probs, axis=1)
        max_cls_prob = cls_probs[np.arange(pred.shape[0]), class_ids]
        scores = obj_conf * max_cls_prob

        keep_mask = scores >= conf_threshold
        if not np.any(keep_mask):
            results.append(np.empty((0, 6), dtype=np.float32))
            continue

        filtered_xywh = pred[keep_mask, :4]
        filtered_scores = scores[keep_mask].astype(np.float32)
        filtered_classes = class_ids[keep_mask]

        half_w = filtered_xywh[:, 2] * 0.5
        half_h = filtered_xywh[:, 3] * 0.5
        boxes_xyxy = np.stack(
            (
                filtered_xywh[:, 0] - half_w,
                filtered_xywh[:, 1] - half_h,
                filtered_xywh[:, 0] + half_w,
                filtered_xywh[:, 1] + half_h,
            ),
            axis=1,
        ).astype(np.float32)

        kept_indices = _nms_single_class_shifted(
            boxes_xyxy=boxes_xyxy,
            scores=filtered_scores,
            class_ids=filtered_classes,
            nms_threshold=nms_threshold,
        )

        selected = np.concatenate(
            (
                boxes_xyxy[kept_indices],
                filtered_scores[kept_indices, np.newaxis],
                filtered_classes[kept_indices, np.newaxis].astype(np.float32),
            ),
            axis=1,
        ).astype(np.float32)
        results.append(selected)

    return results


def postprocess_detections(
    raw_output: np.ndarray,
    letterbox: LetterboxResult,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    input_size: tuple[int, int] = (640, 640),
    already_decoded: bool = True,
    strides: tuple[int, ...] = (8, 16, 32),
) -> list[DetectionBox]:
    """Decode, NMS-filter, and map YOLOX detections back to original image pixels."""
    decoded = decode_yolox_outputs(
        raw_output=raw_output,
        input_size=input_size,
        strides=strides,
        already_decoded=already_decoded,
    )
    nms_batch = batched_nms(
        decoded=decoded,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )
    if not nms_batch or nms_batch[0].shape[0] == 0:
        return []

    inv_scale = 1.0 / letterbox.scale
    max_w = float(letterbox.orig_w)
    max_h = float(letterbox.orig_h)

    boxes: list[DetectionBox] = []
    for row in nms_batch[0]:
        x1 = float(np.clip((float(row[0]) - letterbox.pad_x) * inv_scale, 0.0, max_w))
        y1 = float(np.clip((float(row[1]) - letterbox.pad_y) * inv_scale, 0.0, max_h))
        x2 = float(np.clip((float(row[2]) - letterbox.pad_x) * inv_scale, 0.0, max_w))
        y2 = float(np.clip((float(row[3]) - letterbox.pad_y) * inv_scale, 0.0, max_h))

        if x2 <= x1 or y2 <= y1:
            continue

        confidence = float(np.clip(float(row[4]), 0.0, 1.0))
        class_id = int(round(float(row[5])))
        class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else None

        boxes.append(
            DetectionBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )
        )

    return boxes
