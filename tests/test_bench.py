"""Tests for runtime engines, factory, datasets, profiler, evaluator, and Typer CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from typer.testing import CliRunner

from edge_vision_bench.bench.datasets import (
    COCO_TO_VISDRONE_MAP,
    COCOSubsetDataset,
    MOTSequenceDataset,
    VisDroneMOTDataset,
    generate_synthetic_mot_sequence,
)
from edge_vision_bench.bench.evaluator import (
    evaluate_coco_map,
    evaluate_embedding_parity,
    evaluate_mot_metrics,
    render_pareto_markdown,
    save_benchmark_results,
)
from edge_vision_bench.bench.perturbations import get_default_ood_suite
from edge_vision_bench.bench.profiler import (
    NVMLMemoryMonitor,
    compute_artifact_size_mb,
    profile_engine,
)
from edge_vision_bench.cli import _default_artifact_path, app
from edge_vision_bench.export.onnx_exporter import export_to_onnx
from edge_vision_bench.export.openvino_exporter import export_to_openvino
from edge_vision_bench.runtimes.base import EngineTiming
from edge_vision_bench.runtimes.factory import create_engine, list_available_runtimes
from edge_vision_bench.runtimes.onnx_engine import ONNXRuntimeEngine
from edge_vision_bench.runtimes.openvino_engine import OpenVINORuntimeEngine
from edge_vision_bench.runtimes.pytorch_engine import PyTorchRuntimeEngine
from edge_vision_bench.runtimes.trt_engine import TensorRTRuntimeEngine
from edge_vision_bench.schemas import BenchmarkResult, DetectionBox, PipelineConfig


class TinyDetectorNet(nn.Module):
    """Lightweight synthetic detector emitting decoded YOLOX-format (B, N, 5 + C) tensors."""

    def __init__(self, num_boxes: int = 8, num_classes: int = 80) -> None:
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, num_boxes * (5 + num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        raw = self.fc(pooled).view(x.shape[0], self.num_boxes, 5 + self.num_classes)
        cxcy = torch.sigmoid(raw[..., 0:2]) * 128.0
        wh = torch.sigmoid(raw[..., 2:4]) * 64.0 + 8.0
        probs = torch.sigmoid(raw[..., 4:])
        return torch.cat([cxcy, wh, probs], dim=-1)


class TinyEmbeddingNet(nn.Module):
    """Lightweight synthetic embedding network emitting L2-normalized (B, D) vectors."""

    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.fc(self.pool(x).flatten(1))
        return nn.functional.normalize(feats, p=2, dim=-1, eps=1e-6)


def test_list_available_runtimes_and_factory_errors() -> None:
    runtimes = list_available_runtimes()
    assert "pytorch" in runtimes
    assert "pytorch_compile" in runtimes
    assert "onnxruntime" in runtimes
    assert "openvino" in runtimes

    with pytest.raises(ValueError, match="Unsupported runtime"):
        create_engine("unknown_rt", artifact_path="")  # type: ignore[arg-type]


def test_cpu_engines_parity_timing_and_lifecycle(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    torch.manual_seed(42)
    tiny_model = TinyDetectorNet(num_boxes=6, num_classes=80).eval()
    input_shape = (1, 3, 64, 64)

    onnx_path = export_to_onnx(
        model=tiny_model,
        output_path=tmp_path / "tiny.onnx",
        input_shape=input_shape,
        opset_version=17,
        simplify=False,
    )
    ov_xml_path = export_to_openvino(
        onnx_path=onnx_path,
        output_dir=tmp_path / "ov_fp32",
        precision="fp32",
    )
    assert ov_xml_path.is_file()
    assert ov_xml_path.with_suffix(".bin").is_file()
    # OpenVINO artifact size must sum .xml + .bin.
    xml_only_mb = ov_xml_path.stat().st_size / (1024.0 * 1024.0)
    assert compute_artifact_size_mb(ov_xml_path) > xml_only_mb

    rng = np.random.default_rng(7)
    sample = rng.standard_normal(input_shape).astype(np.float32)

    pt_engine = create_engine(
        "pytorch", artifact_path="", precision="fp32", device="cpu", model=tiny_model
    )
    ort_engine = create_engine(
        "onnxruntime", artifact_path=onnx_path, precision="fp32", device="cpu"
    )
    ov_engine = create_engine("openvino", artifact_path=ov_xml_path, precision="fp32", device="cpu")

    assert isinstance(pt_engine, PyTorchRuntimeEngine)
    assert isinstance(ort_engine, ONNXRuntimeEngine)
    assert isinstance(ov_engine, OpenVINORuntimeEngine)

    outputs_by_rt: dict[str, np.ndarray] = {}
    for name, eng in (("pytorch", pt_engine), ("onnxruntime", ort_engine), ("openvino", ov_engine)):
        eng.warmup(input_shape, runs=2)
        outs, timing = eng.infer(sample)
        assert isinstance(timing, EngineTiming)
        assert timing.h2d_ms == 0.0
        assert timing.d2h_ms == 0.0
        assert timing.compute_ms >= 0.0
        assert timing.total_device_ms == pytest.approx(timing.compute_ms)
        assert eng.get_vram_usage_mb() == 0.0
        assert eng.get_ram_usage_mb() > 0.0
        assert len(outs) == 1
        assert outs[0].shape == (1, 6, 85)
        outputs_by_rt[name] = outs[0]
        eng.close()

    np.testing.assert_allclose(
        outputs_by_rt["pytorch"], outputs_by_rt["onnxruntime"], rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        outputs_by_rt["pytorch"], outputs_by_rt["openvino"], rtol=1e-3, atol=1e-4
    )

    # FP16 on CPU should warn and stay in FP32 without crashing.
    with caplog.at_level(logging.WARNING):
        pt_fp16_cpu = PyTorchRuntimeEngine(
            artifact_path="", precision="fp16", device="cpu", model=tiny_model
        )
        outs_fp16, _ = pt_fp16_cpu.infer(sample)
        assert outs_fp16[0].dtype == np.float32
        pt_fp16_cpu.close()
    assert any("FP16 requested on CPU" in rec.message for rec in caplog.records)


def test_tensorrt_engine_import_safety_and_cpu_error(tmp_path: Path) -> None:
    # Importing TensorRTRuntimeEngine already succeeded at module import time on CPU.
    if "tensorrt" in list_available_runtimes():
        pytest.skip("TensorRT GPU environment is available on this host.")

    dummy_plan = tmp_path / "dummy.engine"
    dummy_plan.write_bytes(b"fake-trt-plan")

    with pytest.raises(RuntimeError, match="TensorRT"):
        TensorRTRuntimeEngine(artifact_path=dummy_plan, precision="fp16", device="cuda")


def test_synthetic_mot_and_evaluate_mot_metrics() -> None:
    frames_a, gt_a = generate_synthetic_mot_sequence(
        num_frames=12, width=320, height=240, num_objects=3, seed=42
    )
    frames_b, gt_b = generate_synthetic_mot_sequence(
        num_frames=12, width=320, height=240, num_objects=3, seed=42
    )
    assert len(frames_a) == 12
    assert frames_a[0].shape == (240, 320, 3)
    assert frames_a[0].dtype == np.uint8
    np.testing.assert_array_equal(frames_a[0], frames_b[0])
    assert gt_a == gt_b

    # Perfect predictions matching ground truth -> MOTA == 1.0, IDF1 == 1.0.
    perfect_preds: dict[int, list[DetectionBox]] = {}
    for frame_idx, boxes in gt_a.items():
        perfect_preds[frame_idx] = [
            DetectionBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=0.95,
                class_id=0,
                track_id=tid,
            )
            for tid, x1, y1, x2, y2 in boxes
        ]

    mota_perfect, idf1_perfect = evaluate_mot_metrics(gt_a, perfect_preds)
    assert mota_perfect == pytest.approx(1.0, abs=1e-6)
    assert idf1_perfect == pytest.approx(1.0, abs=1e-6)

    # Dropping predictions on half the frames must strictly lower MOTA.
    degraded_preds: dict[int, list[DetectionBox]] = {
        idx: (boxes if idx % 2 == 0 else []) for idx, boxes in perfect_preds.items()
    }
    mota_degraded, idf1_degraded = evaluate_mot_metrics(gt_a, degraded_preds)
    assert mota_degraded < mota_perfect
    assert idf1_degraded < idf1_perfect


def test_coco_disjoint_splits_and_map_evaluation(tmp_path: Path) -> None:
    images_dir = tmp_path / "val2017"
    images_dir.mkdir(parents=True)

    images_meta = []
    annotations_meta = []
    ann_id = 1
    for img_id in range(1, 11):
        fname = f"{img_id:06d}.jpg"
        img = np.zeros((100, 120, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / fname), img)
        images_meta.append({"id": img_id, "file_name": fname, "width": 120, "height": 100})
        annotations_meta.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [10.0, 15.0, 40.0, 30.0],
                "area": 1200.0,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    ann_json = tmp_path / "instances_val2017.json"
    ann_json.write_text(
        json.dumps(
            {
                "images": images_meta,
                "annotations": annotations_meta,
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )

    calib_ds = COCOSubsetDataset(
        images_dir=images_dir,
        annotations_json=ann_json,
        split="calibration",
        calibration_size=4,
        eval_size=4,
        seed=42,
    )
    eval_ds = COCOSubsetDataset(
        images_dir=images_dir,
        annotations_json=ann_json,
        split="eval",
        calibration_size=4,
        eval_size=4,
        seed=42,
    )

    assert len(calib_ds) == 4
    assert len(eval_ds) == 4
    assert set(calib_ds.image_ids).isdisjoint(set(eval_ds.image_ids))

    # Perfect predictions on eval_ds should yield mAP@50:95 == 1.0 and mAP@50 == 1.0.
    perfect_preds = {
        img_id: [DetectionBox(x1=10.0, y1=15.0, x2=50.0, y2=45.0, confidence=0.99, class_id=1)]
        for img_id in eval_ds.image_ids
    }
    map50_95, map50 = evaluate_coco_map(perfect_preds, ann_json, image_ids=eval_ds.image_ids)
    assert map50_95 == pytest.approx(1.0, abs=1e-3)
    assert map50 == pytest.approx(1.0, abs=1e-3)

    # Empty predictions return (0.0, 0.0) cleanly.
    empty_map, empty_map50 = evaluate_coco_map(
        {img_id: [] for img_id in eval_ds.image_ids}, ann_json, image_ids=eval_ds.image_ids
    )
    assert empty_map == 0.0
    assert empty_map50 == 0.0


def test_motchallenge_and_visdrone_dataset_loaders(tmp_path: Path) -> None:
    # 1. MOTChallenge layout
    mot_root = tmp_path / "motchallenge" / "SEQ01"
    (mot_root / "img1").mkdir(parents=True)
    (mot_root / "gt").mkdir(parents=True)
    (mot_root / "seqinfo.ini").write_text(
        "[Sequence]\nname=SEQ01\nimDir=img1\nframeRate=30\n", encoding="utf-8"
    )
    for i in range(1, 4):
        cv2.imwrite(str(mot_root / "img1" / f"{i:06d}.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
    (mot_root / "gt" / "gt.txt").write_text(
        "1,1,10,12,20,24,1,1,1.0\n2,1,12,14,20,24,1,1,1.0\n", encoding="utf-8"
    )

    mot_ds = MOTSequenceDataset(tmp_path / "motchallenge")
    assert len(mot_ds) == 1
    seq = mot_ds.get_sequence("SEQ01")
    assert len(seq.frame_paths) == 3
    assert seq.fps == 30.0
    assert 0 in seq.gt_by_frame and seq.gt_by_frame[0][0] == (1, 10.0, 12.0, 30.0, 36.0)

    # 2. VisDrone2019-MOT layout + COCO_TO_VISDRONE_MAP remapping
    vd_root = tmp_path / "visdrone_mot"
    (vd_root / "sequences" / "uav001").mkdir(parents=True)
    (vd_root / "annotations").mkdir(parents=True)
    for i in range(1, 3):
        cv2.imwrite(
            str(vd_root / "sequences" / "uav001" / f"{i:07d}.jpg"),
            np.zeros((64, 64, 3), dtype=np.uint8),
        )
    # Row 1: car (cat 4, valid), Row 2: ignored (cat 0), Row 3: van (cat 5, filtered under COCO)
    (vd_root / "annotations" / "uav001.txt").write_text(
        "1,7,5,6,20,15,1,4,0,0\n1,8,0,0,10,10,0,0,0,0\n1,9,10,10,15,15,1,5,0,0\n",
        encoding="utf-8",
    )

    vd_ds = VisDroneMOTDataset(vd_root, coco_compatible_only=True)
    assert len(vd_ds) == 1
    vd_seq = vd_ds[0]
    assert vd_seq.is_visdrone is True
    assert len(vd_seq.gt_by_frame[0]) == 1
    assert vd_seq.gt_by_frame[0][0][0] == 7

    # 3. Nested VisDrone2019-MOT archive layout (`<root>/VisDrone2019-MOT-val/sequences/...`)
    nested_root = tmp_path / "visdrone_nested"
    nested_split = nested_root / "VisDrone2019-MOT-val"
    (nested_split / "sequences" / "uav002").mkdir(parents=True)
    (nested_split / "annotations").mkdir(parents=True)
    cv2.imwrite(
        str(nested_split / "sequences" / "uav002" / "0000001.jpg"),
        np.zeros((64, 64, 3), dtype=np.uint8),
    )
    (nested_split / "annotations" / "uav002.txt").write_text(
        "1,3,10,10,20,20,1,4,0,0\n", encoding="utf-8"
    )
    nested_ds = MOTSequenceDataset(nested_root)
    assert len(nested_ds) == 1
    assert nested_ds[0].name == "uav002"
    assert nested_ds[0].is_visdrone is True
    assert nested_ds[0].gt_by_frame[0][0][0] == 3

    coco_boxes = [
        DetectionBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9, class_id=0),  # person -> 1
        DetectionBox(x1=0, y1=0, x2=10, y2=10, confidence=0.8, class_id=2),  # car -> 4
        DetectionBox(x1=0, y1=0, x2=10, y2=10, confidence=0.7, class_id=15),  # cat -> dropped
    ]
    remapped = VisDroneMOTDataset.remap_coco_detections(coco_boxes)
    assert [b.class_id for b in remapped] == [COCO_TO_VISDRONE_MAP[0], COCO_TO_VISDRONE_MAP[2]]


def test_profile_engine_save_and_render_pareto_markdown(tmp_path: Path) -> None:
    frames, _ = generate_synthetic_mot_sequence(
        num_frames=6, width=160, height=120, num_objects=2, seed=11
    )

    # Track A: Detection profiling
    det_cfg = PipelineConfig(
        model_name="yolox-tiny-test",
        track="detection",
        input_size=(64, 64),
        conf_threshold=0.01,
        warmup_runs=1,
    )
    det_engine = PyTorchRuntimeEngine(
        artifact_path="", precision="fp32", device="cpu", model=TinyDetectorNet().eval()
    )
    det_result, det_preds = profile_engine(
        engine=det_engine,
        frames_bgr=frames,
        config=det_cfg,
        runtime_name="pytorch",
        precision="fp32",
        batch_size=2,
        warmup_runs=1,
        enable_tracking=True,
    )
    det_engine.close()

    assert isinstance(det_result, BenchmarkResult)
    assert det_result.track == "detection"
    assert det_result.end_to_end_fps > 0.0
    assert det_result.peak_ram_mb > 0.0
    assert det_result.p50_compute_ms <= det_result.p95_compute_ms <= det_result.p99_compute_ms
    assert len(det_preds) == len(frames)

    det_result = det_result.model_copy(
        update={
            "clean_map50_95": 0.412,
            "ood_blur_map50_95": 0.310,
            "clean_mota": 0.72,
            "clean_idf1": 0.78,
        }
    )

    # Track B: Embedding profiling + OOD perturbation + cosine parity
    emb_cfg = PipelineConfig(
        model_name="dinov2-tiny-test",
        track="embedding",
        input_size=(56, 56),
        warmup_runs=1,
    )
    emb_engine = PyTorchRuntimeEngine(
        artifact_path="",
        precision="fp32",
        device="cpu",
        model=TinyEmbeddingNet(embed_dim=32).eval(),
    )
    emb_result, clean_embs = profile_engine(
        engine=emb_engine,
        frames_bgr=frames,
        config=emb_cfg,
        runtime_name="pytorch",
        precision="fp32",
        batch_size=1,
        warmup_runs=1,
    )
    ood_blur = get_default_ood_suite()["blur"]
    _, blur_embs = profile_engine(
        engine=emb_engine,
        frames_bgr=frames,
        config=emb_cfg,
        runtime_name="pytorch",
        precision="fp32",
        batch_size=1,
        warmup_runs=0,
        perturbation=ood_blur,
    )
    emb_engine.close()

    clean_arr = np.stack([np.asarray(e) for e in clean_embs], axis=0)
    blur_arr = np.stack([np.asarray(e) for e in blur_embs], axis=0)
    assert evaluate_embedding_parity(clean_arr, clean_arr) == pytest.approx(1.0, abs=1e-6)
    blur_cos = evaluate_embedding_parity(clean_arr, blur_arr)
    assert -1.0 <= blur_cos <= 1.0

    emb_result = emb_result.model_copy(
        update={"embedding_cosine_sim": 1.0, "ood_blur_cosine_sim": blur_cos}
    )

    json_out = tmp_path / "results" / "bench.json"
    md_out = tmp_path / "results" / "pareto.md"
    save_benchmark_results([det_result, emb_result], json_path=json_out, markdown_path=md_out)

    assert json_out.is_file()
    assert md_out.is_file()
    loaded = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(loaded) == 2

    markdown = render_pareto_markdown([det_result, emb_result])
    assert "Track A — Detection & Multi-Object Tracking" in markdown
    assert "Track B — Vision Transformer Embeddings (DINOv2)" in markdown
    assert "yolox-tiny-test" in markdown
    assert "dinov2-tiny-test" in markdown
    assert "(-0.102)" in markdown

    # Also verify NVMLMemoryMonitor context manager works cleanly on CPU host.
    with NVMLMemoryMonitor(sample_interval_s=0.01) as mon:
        mon.sample()
    assert mon.peak_ram_mb > 0.0


def test_cli_help_and_report_subcommand(tmp_path: Path) -> None:
    runner = CliRunner()

    for args in (
        ["--help"],
        ["export", "--help"],
        ["verify-parity", "--help"],
        ["bench", "--help"],
        ["serve", "--help"],
        ["report", "--help"],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, f"CLI {args} failed: {res.output}"

    # Verify `evb report` merges multiple JSON files cleanly.
    results_dir = tmp_path / "results"
    sample_res = BenchmarkResult(
        model_name="yolox-s",
        track="detection",
        runtime="onnxruntime",
        precision="fp32",
        hardware_target="Test CPU",
        batch_size=1,
        p50_compute_ms=12.5,
        p95_compute_ms=14.1,
        p99_compute_ms=15.0,
        end_to_end_fps=72.0,
    )
    save_benchmark_results([sample_res], results_dir / "run1.json")
    save_benchmark_results(
        [sample_res.model_copy(update={"runtime": "openvino", "p95_compute_ms": 9.8})],
        results_dir / "run2.json",
    )

    out_md = tmp_path / "pareto_summary.md"
    report_res = runner.invoke(
        app, ["report", "--results-dir", str(results_dir), "--output", str(out_md)]
    )
    assert report_res.exit_code == 0, report_res.output
    assert out_md.is_file()
    content = out_md.read_text(encoding="utf-8")
    assert "onnxruntime" in content
    assert "openvino" in content


def test_cli_bench_combined_coco_and_mot_with_ood(tmp_path: Path) -> None:
    # Build tiny ONNX detector artifact
    torch.manual_seed(0)
    tiny_model = TinyDetectorNet(num_boxes=4, num_classes=80).eval()
    onnx_path = export_to_onnx(
        model=tiny_model,
        output_path=tmp_path / "tiny_det.onnx",
        input_shape=(1, 3, 64, 64),
        opset_version=17,
        simplify=False,
    )

    cfg_path = tmp_path / "tiny_det.yaml"
    cfg_path.write_text(
        "model_name: yolox-tiny\n"
        "track: detection\n"
        "input_size: [64, 64]\n"
        "num_classes: 80\n"
        "conf_threshold: 0.05\n"
        "nms_threshold: 0.45\n"
        "warmup_runs: 1\n",
        encoding="utf-8",
    )

    # Mini COCO directory
    coco_root = tmp_path / "coco"
    (coco_root / "val2017").mkdir(parents=True)
    (coco_root / "annotations").mkdir(parents=True)
    images_meta = []
    anns_meta = []
    for img_id in (1, 2):
        fname = f"{img_id:06d}.jpg"
        cv2.imwrite(str(coco_root / "val2017" / fname), np.full((64, 64, 3), 80, dtype=np.uint8))
        images_meta.append({"id": img_id, "file_name": fname, "width": 64, "height": 64})
        anns_meta.append(
            {
                "id": img_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [5.0, 5.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            }
        )
    (coco_root / "annotations" / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": images_meta,
                "annotations": anns_meta,
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )

    # Nested VisDrone-MOT directory
    mot_root = tmp_path / "visdrone_mot"
    split_dir = mot_root / "VisDrone2019-MOT-val"
    (split_dir / "sequences" / "uav_test").mkdir(parents=True)
    (split_dir / "annotations").mkdir(parents=True)
    for frame_num in (1, 2):
        cv2.imwrite(
            str(split_dir / "sequences" / "uav_test" / f"{frame_num:07d}.jpg"),
            np.full((64, 64, 3), 100, dtype=np.uint8),
        )
    (split_dir / "annotations" / "uav_test.txt").write_text(
        "1,1,10,10,20,20,1,4,0,0\n2,1,12,12,20,20,1,4,0,0\n",
        encoding="utf-8",
    )

    out_json = tmp_path / "combined.json"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "bench",
            "--config",
            str(cfg_path),
            "--runtime",
            "onnxruntime",
            "--precision",
            "fp32",
            "--device",
            "cpu",
            "--artifact",
            str(onnx_path),
            "--num-frames",
            "2",
            "--warmup",
            "1",
            "--ood",
            "--coco-dir",
            str(coco_root),
            "--mot-dir",
            str(mot_root),
            "--output-json",
            str(out_json),
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_json.is_file()

    rows = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = BenchmarkResult.model_validate(rows[0])
    assert row.clean_map50_95 is not None
    assert row.clean_map50 is not None
    assert row.ood_blur_map50_95 is not None
    assert row.ood_occlusion_map50_95 is not None
    assert row.ood_lowlight_map50_95 is not None
    assert row.clean_mota is not None
    assert row.clean_idf1 is not None
    assert row.ood_blur_mota is not None


def test_artifact_size_empty_path_and_onnx_fp16_cli_export(tmp_path: Path) -> None:
    # Guard against Path("") == Path(".") summing CWD.
    assert compute_artifact_size_mb(None) == 0.0
    assert compute_artifact_size_mb("") == 0.0
    assert compute_artifact_size_mb(".") == 0.0
    assert compute_artifact_size_mb(Path("")) == 0.0

    artifact_dir = tmp_path / "artifacts"
    cfg = PipelineConfig(
        model_name="yolox-tiny",
        track="detection",
        input_size=(64, 64),
        artifact_dir=str(artifact_dir),
    )
    assert _default_artifact_path(cfg, "pytorch", "fp32") is None
    assert _default_artifact_path(cfg, "pytorch_compile", "fp16") is None
    assert _default_artifact_path(cfg, "onnxruntime", "fp32") == artifact_dir / "yolox-tiny.onnx"
    assert (
        _default_artifact_path(cfg, "onnxruntime", "fp16") == artifact_dir / "yolox-tiny_fp16.onnx"
    )

    # Export tiny FP32 ONNX, then convert via `evb export --runtime onnx --precision fp16`.
    torch.manual_seed(0)
    fp32_onnx = export_to_onnx(
        model=TinyDetectorNet(num_boxes=4, num_classes=80).eval(),
        output_path=artifact_dir / "yolox-tiny.onnx",
        input_shape=(1, 3, 64, 64),
        opset_version=17,
        simplify=False,
    )
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(
        f"model_name: yolox-tiny\n"
        f"track: detection\n"
        f"input_size: [64, 64]\n"
        f"artifact_dir: {artifact_dir}\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "export",
            "--config",
            str(cfg_path),
            "--runtime",
            "onnx",
            "--precision",
            "fp16",
            "--no-pretrained",
            "--onnx-path",
            str(fp32_onnx),
        ],
    )
    assert res.exit_code == 0, res.output
    fp16_onnx = artifact_dir / "yolox-tiny_fp16.onnx"
    assert fp16_onnx.is_file()
    assert fp16_onnx.stat().st_size < fp32_onnx.stat().st_size

    # Verify ONNXRuntimeEngine executes the FP16 ONNX cleanly with FP32 I/O.
    eng = ONNXRuntimeEngine(artifact_path=fp16_onnx, precision="fp16", device="cpu")
    try:
        outs, timing = eng.infer(np.zeros((1, 3, 64, 64), dtype=np.float32))
        assert outs[0].shape == (1, 4, 85)
        assert timing.compute_ms >= 0.0
    finally:
        eng.close()
