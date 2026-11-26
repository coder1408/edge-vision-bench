"""Accuracy evaluators (COCO mAP, MOTA/IDF1, embedding cosine) and Pareto Markdown rendering."""

from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from edge_vision_bench.schemas import BenchmarkResult, DetectionBox


def evaluate_coco_map(
    predictions_by_image_id: dict[int, list[DetectionBox]],
    coco_gt_json: str | Path,
    image_ids: list[int] | None = None,
) -> tuple[float, float]:
    """Compute COCO ``(mAP@50:95, mAP@50)`` via ``pycocotools``."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_path = Path(coco_gt_json)
    if not gt_path.is_file():
        raise FileNotFoundError(f"COCO ground-truth JSON not found: {gt_path}")

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))

    sorted_cat_ids = sorted(int(c) for c in coco_gt.getCatIds())
    cat_id_set = set(sorted_cat_ids)
    uses_standard_coco90 = len(sorted_cat_ids) == 80 and sorted_cat_ids[-1] == 90

    coco_results: list[dict[str, Any]] = []
    for img_id, boxes in predictions_by_image_id.items():
        for box in boxes:
            width = max(0.0, float(box.x2 - box.x1))
            height = max(0.0, float(box.y2 - box.y1))
            if width <= 0.0 or height <= 0.0:
                continue

            if uses_standard_coco90 and 0 <= box.class_id < 80:
                category_id = sorted_cat_ids[box.class_id]
            elif box.class_id in cat_id_set:
                category_id = int(box.class_id)
            elif 0 <= box.class_id < len(sorted_cat_ids):
                category_id = sorted_cat_ids[box.class_id]
            else:
                continue

            coco_results.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "bbox": [float(box.x1), float(box.y1), width, height],
                    "score": float(box.confidence),
                }
            )

    if not coco_results:
        return 0.0, 0.0

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        if image_ids is not None:
            coco_eval.params.imgIds = [int(i) for i in image_ids]
        elif predictions_by_image_id:
            coco_eval.params.imgIds = [int(i) for i in predictions_by_image_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    map50_95 = float(coco_eval.stats[0]) if not math.isnan(float(coco_eval.stats[0])) else 0.0
    map50 = float(coco_eval.stats[1]) if not math.isnan(float(coco_eval.stats[1])) else 0.0
    return max(0.0, map50_95), max(0.0, map50)


def _compute_iou_distance_matrix(
    gt_boxes_tlwh: np.ndarray,
    pred_boxes_tlwh: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Compute ``1 - IoU`` distance matrix (NaN where IoU < iou_threshold) for motmetrics."""
    n_gt = gt_boxes_tlwh.shape[0]
    n_pred = pred_boxes_tlwh.shape[0]
    if n_gt == 0 or n_pred == 0:
        return np.empty((n_gt, n_pred), dtype=np.float64)

    gt_x1 = gt_boxes_tlwh[:, 0:1]
    gt_y1 = gt_boxes_tlwh[:, 1:2]
    gt_x2 = gt_x1 + np.maximum(0.0, gt_boxes_tlwh[:, 2:3])
    gt_y2 = gt_y1 + np.maximum(0.0, gt_boxes_tlwh[:, 3:4])

    pr_x1 = pred_boxes_tlwh[:, 0:1].T
    pr_y1 = pred_boxes_tlwh[:, 1:2].T
    pr_x2 = pr_x1 + np.maximum(0.0, pred_boxes_tlwh[:, 2:3].T)
    pr_y2 = pr_y1 + np.maximum(0.0, pred_boxes_tlwh[:, 3:4].T)

    inter_w = np.maximum(0.0, np.minimum(gt_x2, pr_x2) - np.maximum(gt_x1, pr_x1))
    inter_h = np.maximum(0.0, np.minimum(gt_y2, pr_y2) - np.maximum(gt_y1, pr_y1))
    inter_area = inter_w * inter_h

    gt_area = np.maximum(0.0, gt_x2 - gt_x1) * np.maximum(0.0, gt_y2 - gt_y1)
    pr_area = np.maximum(0.0, pr_x2 - pr_x1) * np.maximum(0.0, pr_y2 - pr_y1)
    union_area = gt_area + pr_area - inter_area

    iou = np.where(union_area > 0.0, inter_area / np.maximum(union_area, 1e-12), 0.0)
    distances = 1.0 - iou
    distances = np.where(iou >= iou_threshold, distances, np.nan)
    return distances.astype(np.float64, copy=False)


def evaluate_mot_metrics(
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]],
    pred_by_frame: dict[int, list[DetectionBox]],
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    """Compute Multi-Object Tracking ``(MOTA, IDF1)`` via ``motmetrics``."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    if not all_frames:
        return 0.0, 0.0

    for frame_idx in all_frames:
        gt_entries = gt_by_frame.get(frame_idx, [])
        pred_entries = pred_by_frame.get(frame_idx, [])

        gt_ids = [int(item[0]) for item in gt_entries]
        gt_tlwh = (
            np.asarray(
                [[g[1], g[2], max(0.0, g[3] - g[1]), max(0.0, g[4] - g[2])] for g in gt_entries],
                dtype=np.float64,
            )
            if gt_entries
            else np.empty((0, 4), dtype=np.float64)
        )

        pred_ids = [
            int(box.track_id) if box.track_id is not None else -(idx + 1)
            for idx, box in enumerate(pred_entries)
        ]
        pred_tlwh = (
            np.asarray(
                [
                    [box.x1, box.y1, max(0.0, box.x2 - box.x1), max(0.0, box.y2 - box.y1)]
                    for box in pred_entries
                ],
                dtype=np.float64,
            )
            if pred_entries
            else np.empty((0, 4), dtype=np.float64)
        )

        # Pure NumPy IoU distance matrix avoids NumPy 2.x np.asfarray issues in motmetrics 1.4.0.
        dists = _compute_iou_distance_matrix(gt_tlwh, pred_tlwh, iou_threshold=iou_threshold)
        acc.update(gt_ids, pred_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "idf1"], name="sequence")
    mota_raw = float(summary.loc["sequence", "mota"])
    idf1_raw = float(summary.loc["sequence", "idf1"])
    mota = 0.0 if math.isnan(mota_raw) else mota_raw
    idf1 = 0.0 if math.isnan(idf1_raw) else idf1_raw
    return mota, idf1


def evaluate_embedding_parity(
    ref_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
) -> float:
    """Compute mean row-wise cosine similarity between reference and target embeddings."""
    ref = np.asarray(ref_embeddings, dtype=np.float64)
    tgt = np.asarray(target_embeddings, dtype=np.float64)

    if ref.ndim == 1:
        ref = ref[None, :]
    if tgt.ndim == 1:
        tgt = tgt[None, :]

    if ref.shape != tgt.shape:
        raise ValueError(f"Embedding shape mismatch: reference {ref.shape} vs target {tgt.shape}.")
    if ref.shape[0] == 0:
        return 0.0

    ref_norms = np.linalg.norm(ref, axis=-1)
    tgt_norms = np.linalg.norm(tgt, axis=-1)
    denom = np.maximum(ref_norms * tgt_norms, 1e-12)
    cos_sim = np.sum(ref * tgt, axis=-1) / denom
    return float(np.mean(cos_sim))


def save_benchmark_results(
    results: list[BenchmarkResult],
    json_path: str | Path,
    markdown_path: str | Path | None = None,
) -> None:
    """Persist benchmark results to JSON and optionally render a Markdown Pareto summary."""
    json_file = Path(json_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    payload = [res.model_dump(mode="json") for res in results]
    json_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if markdown_path is not None:
        md_file = Path(markdown_path)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.write_text(render_pareto_markdown(results), encoding="utf-8")


def _fmt_opt(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _fmt_ood_delta(value: float | None, baseline: float | None, decimals: int = 3) -> str:
    if value is None:
        return "—"
    if baseline is None:
        return f"{value:.{decimals}f}"
    delta = value - baseline
    return f"{value:.{decimals}f} ({delta:+.{decimals}f})"


def render_pareto_markdown(results: list[BenchmarkResult]) -> str:
    """Render Track A (Detection/Tracking) and Track B (DINOv2 Embedding) Pareto tables."""
    lines: list[str] = ["# Edge Vision Bench — Pareto Benchmark Summary", ""]

    if not results:
        lines.append("_No benchmark results available._")
        lines.append("")
        return "\n".join(lines)

    hardware_targets = sorted({r.hardware_target for r in results})
    lines.append(f"**Hardware Targets:** {', '.join(hardware_targets)}  ")
    lines.append(f"**Total Benchmark Runs:** {len(results)}")
    lines.append("")

    det_results = sorted(
        [r for r in results if r.track == "detection"],
        key=lambda r: (r.model_name, r.hardware_target, r.p95_compute_ms),
    )
    emb_results = sorted(
        [r for r in results if r.track == "embedding"],
        key=lambda r: (r.model_name, r.hardware_target, r.p95_compute_ms),
    )

    if det_results:
        lines.extend(
            [
                "## Track A — Detection & Multi-Object Tracking",
                "",
                "| Model | Runtime | Prec | Hardware | B | p50 / p95 / p99 (ms) "
                "| H2D / Compute / D2H (ms) | FPS | VRAM (MB) | RAM (MB) | Size (MB) "
                "| Clean mAP50:95 | OOD Blur / Occ / LowLight (Δ) | Clean MOTA / IDF1 |",
                "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: "
                "| :---: | :---: | :---: | :---: |",
            ]
        )
        for r in det_results:
            lat_str = f"{r.p50_compute_ms:.2f} / {r.p95_compute_ms:.2f} / {r.p99_compute_ms:.2f}"
            stage_str = f"{r.p50_h2d_ms:.2f} / {r.p50_compute_ms:.2f} / {r.p50_d2h_ms:.2f}"
            ood_str = (
                f"{_fmt_ood_delta(r.ood_blur_map50_95, r.clean_map50_95)} / "
                f"{_fmt_ood_delta(r.ood_occlusion_map50_95, r.clean_map50_95)} / "
                f"{_fmt_ood_delta(r.ood_lowlight_map50_95, r.clean_map50_95)}"
            )
            mot_str = f"{_fmt_opt(r.clean_mota)} / {_fmt_opt(r.clean_idf1)}"
            lines.append(
                f"| `{r.model_name}` | `{r.runtime}` | `{r.precision}` | {r.hardware_target} "
                f"| {r.batch_size} | {lat_str} | {stage_str} | {r.end_to_end_fps:.1f} "
                f"| {r.peak_vram_mb:.1f} | {r.peak_ram_mb:.1f} | {r.artifact_size_mb:.2f} "
                f"| {_fmt_opt(r.clean_map50_95)} | {ood_str} | {mot_str} |"
            )
        lines.append("")

    if emb_results:
        lines.extend(
            [
                "## Track B — Vision Transformer Embeddings (DINOv2)",
                "",
                "| Model | Runtime | Prec | Hardware | B | p50 / p95 / p99 (ms) "
                "| H2D / Compute / D2H (ms) | FPS | VRAM (MB) | RAM (MB) | Size (MB) "
                "| Clean CosSim | OOD Blur / Occ / LowLight CosSim (Δ) |",
                "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: "
                "| :---: | :---: | :---: |",
            ]
        )
        for r in emb_results:
            lat_str = f"{r.p50_compute_ms:.2f} / {r.p95_compute_ms:.2f} / {r.p99_compute_ms:.2f}"
            stage_str = f"{r.p50_h2d_ms:.2f} / {r.p50_compute_ms:.2f} / {r.p50_d2h_ms:.2f}"
            ood_str = (
                f"{_fmt_ood_delta(r.ood_blur_cosine_sim, r.embedding_cosine_sim, 4)} / "
                f"{_fmt_ood_delta(r.ood_occlusion_cosine_sim, r.embedding_cosine_sim, 4)} / "
                f"{_fmt_ood_delta(r.ood_lowlight_cosine_sim, r.embedding_cosine_sim, 4)}"
            )
            lines.append(
                f"| `{r.model_name}` | `{r.runtime}` | `{r.precision}` | {r.hardware_target} "
                f"| {r.batch_size} | {lat_str} | {stage_str} | {r.end_to_end_fps:.1f} "
                f"| {r.peak_vram_mb:.1f} | {r.peak_ram_mb:.1f} | {r.artifact_size_mb:.2f} "
                f"| {_fmt_opt(r.embedding_cosine_sim, 4)} | {ood_str} |"
            )
        lines.append("")

    return "\n".join(lines)
