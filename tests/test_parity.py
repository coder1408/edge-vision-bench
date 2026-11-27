"""Offline numerical parity and export tests for Track A (YOLOX) and Track B (DINOv2)."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import openvino as ov
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from edge_vision_bench.export.onnx_exporter import (
    convert_onnx_to_fp16,
    export_to_onnx,
    verify_onnx_parity,
)
from edge_vision_bench.export.openvino_exporter import export_to_openvino
from edge_vision_bench.export.trt_builder import (
    _qdq_has_int32_dequant,
    build_trt_engine,
    quantize_onnx_qdq_int8,
)
from edge_vision_bench.export.trt_calibrator import EntropyCalibrator
from edge_vision_bench.models.dinov2 import load_dinov2
from edge_vision_bench.models.yolox import YOLOXModel, load_yolox
from edge_vision_bench.schemas import PrecisionName


class _TinyEmbeddingNet(nn.Module):
    """Lightweight deterministic L2-normalized embedding network for fast CI parity checks."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.stem(x).flatten(1)
        return F.normalize(self.proj(pooled), p=2, dim=-1, eps=1e-6)


@pytest.fixture(scope="module")
def yolox_tiny_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[YOLOXModel, Path]:
    """Build an unpretrained YOLOX-s model and export it to ONNX at 160x160."""
    torch.manual_seed(0)
    model = YOLOXModel(depth=0.33, width=0.50, num_classes=80).eval()
    work_dir = tmp_path_factory.mktemp("yolox_parity")
    onnx_path = export_to_onnx(
        model=model,
        output_path=work_dir / "yolox_s_160.onnx",
        input_shape=(1, 3, 160, 160),
        opset_version=17,
        simplify=True,
    )
    return model, onnx_path


def test_yolox_onnx_parity_and_dynamic_batch(
    yolox_tiny_artifacts: tuple[YOLOXModel, Path],
) -> None:
    """Verify YOLOX-s ONNX parity at 160x160 and dynamic batch inference for B=1 and B=2."""
    model, onnx_path = yolox_tiny_artifacts
    assert onnx_path.exists()

    report = verify_onnx_parity(
        pytorch_model=model,
        onnx_path=onnx_path,
        input_shape=(1, 3, 160, 160),
        track="detection",
        model_name="yolox-s-unpretrained",
        rtol=1e-3,
        atol=1e-4,
        num_samples=4,
        seed=42,
        raise_on_fail=True,
    )
    assert report.passed
    assert report.reference_runtime == "pytorch"
    assert report.target_runtime == "onnxruntime"
    assert report.num_samples == 4

    # Verify dynamic batch axis supports both B=1 and B=2 cleanly.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    expected_anchors = (160 // 8) ** 2 + (160 // 16) ** 2 + (160 // 32) ** 2

    rng = np.random.default_rng(123)
    for batch_size in (1, 2):
        sample = rng.standard_normal((batch_size, 3, 160, 160)).astype(np.float32)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(sample)).numpy()
        ort_out = np.asarray(session.run(None, {input_name: sample})[0], dtype=np.float32)
        assert ort_out.shape == (batch_size, expected_anchors, 85)
        assert np.allclose(pt_out, ort_out, rtol=1e-3, atol=1e-4)


def test_convert_onnx_to_fp16_yolox(
    yolox_tiny_artifacts: tuple[YOLOXModel, Path],
    tmp_path: Path,
) -> None:
    """Convert YOLOX FP32 ONNX to FP16 and verify size reduction, FLOAT16 weights, and parity."""
    model, fp32_onnx_path = yolox_tiny_artifacts
    fp16_onnx_path = convert_onnx_to_fp16(
        onnx_path=fp32_onnx_path,
        output_path=tmp_path / "yolox_s_160_fp16.onnx",
    )
    assert fp16_onnx_path.exists()
    assert fp16_onnx_path.stat().st_size < fp32_onnx_path.stat().st_size * 0.65

    loaded = onnx.load(str(fp16_onnx_path))
    initializer_dtypes = {init.data_type for init in loaded.graph.initializer}
    assert onnx.TensorProto.FLOAT16 in initializer_dtypes

    session = ort.InferenceSession(str(fp16_onnx_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].type == "tensor(float)"

    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(321)
    for batch_size in (1, 2):
        sample = rng.standard_normal((batch_size, 3, 160, 160)).astype(np.float32)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(sample)).numpy()
        ort_fp16_out = np.asarray(session.run(None, {input_name: sample})[0], dtype=np.float32)
        assert ort_fp16_out.shape == pt_out.shape
        assert np.allclose(pt_out, ort_fp16_out, rtol=2e-2, atol=2e-2)


def test_yolox_openvino_fp32_fp16_int8_parity(
    yolox_tiny_artifacts: tuple[YOLOXModel, Path],
    tmp_path: Path,
) -> None:
    """Export YOLOX ONNX to OpenVINO FP32, FP16, and NNCF INT8 and verify CPU outputs."""
    model, onnx_path = yolox_tiny_artifacts
    rng = np.random.default_rng(7)
    probe = rng.standard_normal((1, 3, 160, 160)).astype(np.float32)
    with torch.no_grad():
        pt_out = model(torch.from_numpy(probe)).numpy()

    core = ov.Core()

    # FP32 & FP16 OpenVINO IR export and numerical parity verification.
    ov_cases: tuple[tuple[PrecisionName, float, float], ...] = (
        ("fp32", 1e-3, 1e-4),
        ("fp16", 2e-2, 2e-2),
    )
    for precision, rtol, atol in ov_cases:
        xml_path = export_to_openvino(
            onnx_path=onnx_path,
            output_dir=tmp_path / f"ov_{precision}",
            precision=precision,
        )
        assert xml_path.exists()
        assert xml_path.with_suffix(".bin").exists()

        compiled = core.compile_model(str(xml_path), "CPU")
        ov_out = np.asarray(compiled([probe])[compiled.output(0)], dtype=np.float32)
        assert ov_out.shape == pt_out.shape
        assert np.allclose(pt_out, ov_out, rtol=rtol, atol=atol)

    # INT8 NNCF PTQ with 4 synthetic calibration tensors.
    calib_tensors = [rng.standard_normal((1, 3, 160, 160)).astype(np.float32) for _ in range(4)]
    int8_xml = export_to_openvino(
        onnx_path=onnx_path,
        output_dir=tmp_path / "ov_int8",
        precision="int8",
        calibration_tensors=calib_tensors,
    )
    assert int8_xml.exists()
    assert int8_xml.with_suffix(".bin").exists()

    compiled_int8 = core.compile_model(str(int8_xml), "CPU")
    ov_int8_out = np.asarray(compiled_int8([probe])[compiled_int8.output(0)], dtype=np.float32)
    assert ov_int8_out.shape == pt_out.shape
    assert np.isfinite(ov_int8_out).all()

    # Check INT8 maintains strong correlation with FP32 reference outputs.
    pt_flat = pt_out.reshape(-1).astype(np.float64)
    int8_flat = ov_int8_out.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(pt_flat) * np.linalg.norm(int8_flat))
    cosine = float(np.dot(pt_flat, int8_flat) / denom)
    assert cosine >= 0.90


def test_lightweight_embedding_onnx_parity(tmp_path: Path) -> None:
    """Verify embedding track ONNX export and cosine parity on a lightweight module."""
    torch.manual_seed(1)
    model = _TinyEmbeddingNet(embed_dim=64).eval()
    onnx_path = export_to_onnx(
        model=model,
        output_path=tmp_path / "tiny_embed.onnx",
        input_shape=(1, 3, 64, 64),
        opset_version=17,
        simplify=True,
    )
    report = verify_onnx_parity(
        pytorch_model=model,
        onnx_path=onnx_path,
        input_shape=(2, 3, 64, 64),
        track="embedding",
        model_name="tiny-embed",
        min_cosine=0.995,
        num_samples=4,
        seed=99,
        raise_on_fail=True,
    )
    assert report.passed
    assert report.cosine_similarity is not None
    assert report.cosine_similarity >= 0.995


def test_dinov2_vits14_unpretrained_onnx_parity(tmp_path: Path) -> None:
    """Verify unpretrained DINOv2 ViT-S/14 exports cleanly to ONNX opset 17 with >=0.995 cosine."""
    torch.manual_seed(2)
    model = load_dinov2("dinov2-vits14", pretrained=False, input_size=(224, 224))
    onnx_path = export_to_onnx(
        model=model,
        output_path=tmp_path / "dinov2_vits14_224.onnx",
        input_shape=(1, 3, 224, 224),
        opset_version=17,
        simplify=True,
    )
    assert onnx_path.exists()

    report = verify_onnx_parity(
        pytorch_model=model,
        onnx_path=onnx_path,
        input_shape=(1, 3, 224, 224),
        track="embedding",
        model_name="dinov2-vits14-unpretrained",
        min_cosine=0.995,
        num_samples=3,
        seed=42,
        raise_on_fail=True,
    )
    assert report.passed
    assert report.cosine_similarity is not None
    assert report.cosine_similarity >= 0.995

    # Verify unit-norm embedding output with dynamic batch size B=2.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    batch = np.random.default_rng(11).standard_normal((2, 3, 224, 224)).astype(np.float32)
    embeddings = np.asarray(session.run(None, {input_name: batch})[0], dtype=np.float32)
    assert embeddings.shape == (2, model.embedding_dim)
    norms = np.linalg.norm(embeddings, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_trt_cpu_guard_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Confirm TensorRT builder and calibrator raise clear RuntimeError when TRT is unavailable."""
    import edge_vision_bench.export.trt_builder as trt_builder_mod
    import edge_vision_bench.export.trt_calibrator as trt_calib_mod

    monkeypatch.setattr(trt_builder_mod, "trt", None)
    monkeypatch.setattr(trt_calib_mod, "trt", None)

    dummy_onnx = tmp_path / "dummy.onnx"
    dummy_onnx.write_bytes(b"placeholder")

    with pytest.raises(RuntimeError, match="TensorRT is only available on CUDA GPU hosts"):
        build_trt_engine(
            onnx_path=dummy_onnx,
            engine_path=tmp_path / "dummy.engine",
            precision="fp16",
        )

    with pytest.raises(RuntimeError, match="CUDA GPU hosts"):
        EntropyCalibrator(
            calibration_frames=[],
            preprocess_fn=lambda p: np.zeros((3, 64, 64), dtype=np.float32),
            batch_size=1,
            input_shape=(3, 64, 64),
            cache_file=tmp_path / "calib.cache",
        )


def test_trt_int8_dedicated_calibration_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify INT8 TRT build sets a dedicated static calibration profile matching calib_batch."""
    import edge_vision_bench.export.trt_builder as trt_builder_mod

    created_profiles: list[Any] = []
    config_state: dict[str, Any] = {}
    calibrator_state: dict[str, Any] = {"closed": False}

    class _FakeProfile:
        def __init__(self) -> None:
            self.shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}

        def set_shape(
            self,
            name: str,
            min: tuple[int, ...],
            opt: tuple[int, ...],
            max: tuple[int, ...],
        ) -> None:
            self.shapes[name] = (min, opt, max)

    class _FakeConfig:
        def __init__(self) -> None:
            self.flags: set[str] = set()
            self.profiles: list[_FakeProfile] = []
            self.calibration_profile: _FakeProfile | None = None
            self.int8_calibrator: Any = None
            self.workspace_limit: int = 0

        def set_memory_pool_limit(self, _pool_type: str, size: int) -> None:
            self.workspace_limit = size

        def add_optimization_profile(self, profile: _FakeProfile) -> None:
            self.profiles.append(profile)

        def set_calibration_profile(self, profile: _FakeProfile) -> None:
            self.calibration_profile = profile

        def set_flag(self, flag: str) -> None:
            self.flags.add(flag)

    class _FakeInput:
        name = "images"

    class _FakeNetwork:
        def get_input(self, _idx: int) -> _FakeInput:
            return _FakeInput()

    class _FakeParser:
        num_errors = 0

        def __init__(self, _network: _FakeNetwork, _logger: Any) -> None:
            pass

        def parse_from_file(self, _path: str) -> bool:
            return True

    class _FakeBuilder:
        def __init__(self, _logger: Any) -> None:
            pass

        def create_network(self, _flags: int) -> _FakeNetwork:
            return _FakeNetwork()

        def create_builder_config(self) -> _FakeConfig:
            cfg = _FakeConfig()
            config_state["config"] = cfg
            return cfg

        def create_optimization_profile(self) -> _FakeProfile:
            prof = _FakeProfile()
            created_profiles.append(prof)
            return prof

        def build_serialized_network(self, _network: _FakeNetwork, _config: _FakeConfig) -> bytes:
            return b"fake-trt-engine"

    class _FakeLogger:
        WARNING = 1

        def __init__(self, _level: int) -> None:
            pass

    class _FakeTRT:
        Logger = _FakeLogger
        Builder = _FakeBuilder
        OnnxParser = _FakeParser
        IInt8EntropyCalibrator2 = object

        class NetworkDefinitionCreationFlag:
            EXPLICIT_BATCH = 0

        class MemoryPoolType:
            WORKSPACE = "workspace"

        class BuilderFlag:
            FP16 = "fp16"
            INT8 = "int8"

    class _FakeCalibrator:
        def __init__(
            self,
            calibration_frames: list[Path],
            preprocess_fn: Any,
            batch_size: int,
            input_shape: tuple[int, int, int],
            cache_file: Path,
        ) -> None:
            del calibration_frames, preprocess_fn, cache_file
            self.batch_size = batch_size
            self.input_shape = input_shape
            calibrator_state["batch_size"] = batch_size

        def close(self) -> None:
            calibrator_state["closed"] = True

    monkeypatch.setattr(trt_builder_mod, "trt", _FakeTRT)
    monkeypatch.setattr(trt_builder_mod, "EntropyCalibrator", _FakeCalibrator)

    dummy_onnx = tmp_path / "model.onnx"
    dummy_onnx.write_bytes(b"onnx-placeholder")
    calib_frames = [tmp_path / "f1.jpg", tmp_path / "f2.jpg"]
    engine_path = build_trt_engine(
        onnx_path=dummy_onnx,
        engine_path=tmp_path / "model_int8.engine",
        precision="int8",
        input_shape=(3, 160, 160),
        min_batch=1,
        opt_batch=8,
        max_batch=16,
        calibration_frames=calib_frames,
        preprocess_fn=lambda p: np.zeros((3, 160, 160), dtype=np.float32),
    )

    assert engine_path.read_bytes() == b"fake-trt-engine"
    assert calibrator_state["batch_size"] == 2
    assert calibrator_state["closed"] is True

    cfg = config_state["config"]
    assert len(created_profiles) == 2
    main_profile, calib_profile = created_profiles
    assert cfg.profiles == [main_profile]
    assert cfg.calibration_profile is calib_profile
    assert cfg.calibration_profile is not main_profile
    assert main_profile.shapes["images"] == (
        (1, 3, 160, 160),
        (8, 3, 160, 160),
        (16, 3, 160, 160),
    )
    assert calib_profile.shapes["images"] == (
        (2, 3, 160, 160),
        (2, 3, 160, 160),
        (2, 3, 160, 160),
    )
    assert {"int8", "fp16"} <= cfg.flags


def test_quantize_onnx_qdq_int8_yolox(
    yolox_tiny_artifacts: tuple[YOLOXModel, Path],
    tmp_path: Path,
) -> None:
    """Quantize tiny YOLOX ONNX to Q/DQ INT8 across 3 strided chunks and verify ORT execution."""
    _, fp32_onnx_path = yolox_tiny_artifacts
    rng = np.random.default_rng(77)

    # 9 frames -> calib_stride=3, exercising 3 distinct set_range + collect_data chunks.
    frames: list[Path] = []
    tensors: dict[Path, np.ndarray] = {}
    for idx in range(9):
        frame_path = tmp_path / f"calib_{idx}.jpg"
        frame_path.write_bytes(f"frame-{idx}".encode())
        frames.append(frame_path)
        shape = (3, 160, 160) if idx % 2 == 0 else (1, 3, 160, 160)
        tensors[frame_path] = rng.standard_normal(shape).astype(np.float32)

    visited_frames: list[Path] = []

    def _preprocess(path: Path) -> np.ndarray:
        visited_frames.append(path)
        return tensors[path]

    qdq_path = quantize_onnx_qdq_int8(
        onnx_path=fp32_onnx_path,
        output_path=tmp_path / "yolox_s_160_int8_qdq.onnx",
        calibration_frames=frames,
        preprocess_fn=_preprocess,
        input_shape=(3, 160, 160),
    )
    assert visited_frames == frames
    assert qdq_path.exists()
    onnx.checker.check_model(str(qdq_path))
    assert not _qdq_has_int32_dequant(qdq_path)

    qdq_model = onnx.load(str(qdq_path))
    op_types = {node.op_type for node in qdq_model.graph.node}
    assert "QuantizeLinear" in op_types
    assert "DequantizeLinear" in op_types

    # Verify _qdq_has_int32_dequant catches stale ORT graphs with INT32 bias DequantizeLinear.
    stale_qdq = tmp_path / "stale_int32_qdq.onnx"
    stale_graph = onnx.helper.make_graph(
        nodes=[
            onnx.helper.make_node(
                "DequantizeLinear",
                inputs=["bias_q", "bias_scale", "bias_zp"],
                outputs=["bias_dq"],
            )
        ],
        name="stale_int32_qdq",
        inputs=[],
        outputs=[onnx.helper.make_tensor_value_info("bias_dq", onnx.TensorProto.FLOAT, [1])],
        initializer=[
            onnx.helper.make_tensor("bias_q", onnx.TensorProto.INT32, [1], [1]),
            onnx.helper.make_tensor("bias_scale", onnx.TensorProto.FLOAT, [1], [0.1]),
            onnx.helper.make_tensor("bias_zp", onnx.TensorProto.INT32, [1], [0]),
        ],
    )
    onnx.save(onnx.helper.make_model(stale_graph), str(stale_qdq))
    assert _qdq_has_int32_dequant(stale_qdq)

    session = ort.InferenceSession(str(qdq_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    expected_anchors = (160 // 8) ** 2 + (160 // 16) ** 2 + (160 // 32) ** 2
    for batch_size in (1, 2):
        sample = rng.standard_normal((batch_size, 3, 160, 160)).astype(np.float32)
        out = np.asarray(session.run(None, {input_name: sample})[0], dtype=np.float32)
        assert out.shape == (batch_size, expected_anchors, 85)
        assert np.isfinite(out).all()


def test_trt11_calibrator_import_and_builder_strongly_typed(
    yolox_tiny_artifacts: tuple[YOLOXModel, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Confirm TensorRT 11+ (without IInt8EntropyCalibrator2/FP16/INT8 flags) imports and builds."""
    import edge_vision_bench.export.trt_builder as trt_builder_mod
    import edge_vision_bench.export.trt_calibrator as trt_calib_mod

    # Verify importing trt_calibrator when tensorrt lacks IInt8EntropyCalibrator2 never crashes.
    fake_trt11_module = types.ModuleType("tensorrt")
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt11_module)
    reloaded_calib = importlib.reload(trt_calib_mod)
    try:
        assert reloaded_calib._CalibratorBase is object
        monkeypatch.setattr(reloaded_calib, "cudart", object())
        with pytest.raises(RuntimeError, match="TensorRT 11\\+"):
            reloaded_calib.EntropyCalibrator(
                calibration_frames=[],
                preprocess_fn=lambda p: np.zeros((3, 160, 160), dtype=np.float32),
                batch_size=1,
                input_shape=(3, 160, 160),
                cache_file=tmp_path / "calib.cache",
            )
    finally:
        monkeypatch.delitem(sys.modules, "tensorrt", raising=False)
        importlib.reload(trt_calib_mod)

    # Verify TRT 11.x strongly-typed FP16 and explicit Q/DQ INT8 engine build paths.
    parsed_paths: list[str] = []
    network_flags_seen: list[int] = []
    flags_set: list[str] = []

    class _FakeProfile11:
        def set_shape(
            self,
            _name: str,
            min: tuple[int, ...],
            opt: tuple[int, ...],
            max: tuple[int, ...],
        ) -> None:
            del min, opt, max

    class _FakeConfig11:
        def set_memory_pool_limit(self, _pool_type: str, _size: int) -> None:
            pass

        def add_optimization_profile(self, _profile: _FakeProfile11) -> None:
            pass

        def set_flag(self, flag: str) -> None:
            flags_set.append(flag)

    class _FakeInput11:
        name = "images"

    class _FakeNetwork11:
        def get_input(self, _idx: int) -> _FakeInput11:
            return _FakeInput11()

    class _FakeParser11:
        num_errors = 0

        def __init__(self, _network: _FakeNetwork11, _logger: Any) -> None:
            pass

        def parse_from_file(self, path: str) -> bool:
            parsed_paths.append(path)
            return True

    class _FakeBuilder11:
        def __init__(self, _logger: Any) -> None:
            pass

        def create_network(self, flags: int) -> _FakeNetwork11:
            network_flags_seen.append(flags)
            return _FakeNetwork11()

        def create_builder_config(self) -> _FakeConfig11:
            return _FakeConfig11()

        def create_optimization_profile(self) -> _FakeProfile11:
            return _FakeProfile11()

        def build_serialized_network(
            self,
            _network: _FakeNetwork11,
            _config: _FakeConfig11,
        ) -> bytes:
            return b"fake-trt11-engine"

    class _FakeLogger11:
        WARNING = 1

        def __init__(self, _level: int) -> None:
            pass

    class _FakeTRT11:
        Logger = _FakeLogger11
        Builder = _FakeBuilder11
        OnnxParser = _FakeParser11

        class NetworkDefinitionCreationFlag:
            pass

        class MemoryPoolType:
            WORKSPACE = "workspace"

        class BuilderFlag:
            pass

    monkeypatch.setattr(trt_builder_mod, "trt", _FakeTRT11)

    _, fp32_onnx_path = yolox_tiny_artifacts
    local_onnx = tmp_path / "yolox_s_160.onnx"
    local_onnx.write_bytes(fp32_onnx_path.read_bytes())

    fp16_engine = build_trt_engine(
        onnx_path=local_onnx,
        engine_path=tmp_path / "yolox_fp16.engine",
        precision="fp16",
        input_shape=(3, 160, 160),
    )
    assert fp16_engine.read_bytes() == b"fake-trt11-engine"
    assert Path(parsed_paths[-1]) == tmp_path / "yolox_s_160_fp16.onnx"
    assert Path(parsed_paths[-1]).exists()

    calib_frame = tmp_path / "calib_trt11.jpg"
    calib_frame.write_bytes(b"img")
    rng = np.random.default_rng(5)
    int8_engine = build_trt_engine(
        onnx_path=local_onnx,
        engine_path=tmp_path / "yolox_int8.engine",
        precision="int8",
        input_shape=(3, 160, 160),
        calibration_frames=[calib_frame],
        preprocess_fn=lambda _p: rng.standard_normal((3, 160, 160)).astype(np.float32),
    )
    assert int8_engine.read_bytes() == b"fake-trt11-engine"
    assert Path(parsed_paths[-1]) == tmp_path / "yolox_s_160_int8_qdq.onnx"
    assert Path(parsed_paths[-1]).exists()
    assert network_flags_seen == [0, 0]
    assert flags_set == []


@pytest.mark.slow
def test_yolox_s_pretrained_strict_load_and_parity(tmp_path: Path) -> None:
    """Download official Megvii yolox_s.pth, verify strict=True loading and ONNX parity."""
    model = load_yolox("yolox-s", pretrained=True, weights_dir="models/weights")
    onnx_path = export_to_onnx(
        model=model,
        output_path=tmp_path / "yolox_s_640.onnx",
        input_shape=(1, 3, 640, 640),
        opset_version=17,
        simplify=True,
    )
    report = verify_onnx_parity(
        pytorch_model=model,
        onnx_path=onnx_path,
        input_shape=(1, 3, 640, 640),
        track="detection",
        model_name="yolox-s",
        rtol=1e-3,
        atol=1e-4,
        num_samples=2,
        seed=42,
        raise_on_fail=True,
    )
    assert report.passed
