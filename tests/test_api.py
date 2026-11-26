"""Unit and integration tests for the FastAPI inference and Pareto reporting API."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from edge_vision_bench.api.app import create_app
from edge_vision_bench.runtimes.base import BaseRuntimeEngine, EngineTiming
from edge_vision_bench.schemas import BenchmarkResult, PipelineConfig


class MockDetectionEngine(BaseRuntimeEngine):
    """Deterministic mock engine emitting synthetic decoded YOLOX detection tensors."""

    def __init__(
        self,
        precision: str = "fp32",
        device: str = "cpu",
        shift_per_call_px: float = 0.0,
        num_classes: int = 80,
    ) -> None:
        super().__init__(artifact_path="mock.onnx", precision=precision, device=device)
        self.shift_per_call_px = shift_per_call_px
        self.num_classes = num_classes
        self.call_count = 0
        self.warmup_runs_called = 0
        self.closed = False

    def warmup(self, input_shape: tuple[int, int, int, int], runs: int = 50) -> None:
        self.warmup_runs_called = runs
        return None

    def close(self) -> None:
        self.closed = True
        return None

    def infer(self, tensor_nchw: np.ndarray) -> tuple[list[np.ndarray], EngineTiming]:
        batch_size = int(tensor_nchw.shape[0])
        out = np.zeros((batch_size, 4, 5 + self.num_classes), dtype=np.float32)

        dx = self.shift_per_call_px * self.call_count
        self.call_count += 1

        # Detection 0: high-confidence person box (cx=200+dx, cy=300, w=60, h=100)
        out[:, 0, 0] = 200.0 + dx
        out[:, 0, 1] = 300.0
        out[:, 0, 2] = 60.0
        out[:, 0, 3] = 100.0
        out[:, 0, 4] = 0.95
        out[:, 0, 5 + 0] = 0.95

        # Detection 1: high-confidence car box (cx=450+dx, cy=220, w=80, h=60)
        out[:, 1, 0] = 450.0 + dx
        out[:, 1, 1] = 220.0
        out[:, 1, 2] = 80.0
        out[:, 1, 3] = 60.0
        out[:, 1, 4] = 0.90
        out[:, 1, 5 + 2] = 0.92

        timing = EngineTiming(h2d_ms=0.12, compute_ms=1.85, d2h_ms=0.08)
        return [out], timing

    def get_vram_usage_mb(self) -> float:
        return 0.0


class MockEmbeddingEngine(BaseRuntimeEngine):
    """Deterministic mock engine emitting unit-norm DINOv2-style embeddings."""

    def __init__(self, embedding_dim: int = 384) -> None:
        super().__init__(artifact_path="mock_dinov2.onnx", precision="fp32", device="cpu")
        self.embedding_dim = embedding_dim

    def warmup(self, input_shape: tuple[int, int, int, int], runs: int = 50) -> None:
        return None

    def infer(self, tensor_nchw: np.ndarray) -> tuple[list[np.ndarray], EngineTiming]:
        batch_size = int(tensor_nchw.shape[0])
        vec = np.ones((batch_size, self.embedding_dim), dtype=np.float32)
        vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
        return [vec], EngineTiming(h2d_ms=0.05, compute_ms=2.10, d2h_ms=0.04)

    def get_vram_usage_mb(self) -> float:
        return 0.0


def _make_png_bytes(width: int = 640, height: int = 640, color: int = 128) -> bytes:
    image = np.full((height, width, 3), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _make_sequence_zip_bytes(num_frames: int = 4) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        # Write frames in reverse order to verify natural filename sorting
        for idx in reversed(range(num_frames)):
            zf.writestr(f"seq/frame_{idx:04d}.png", _make_png_bytes(color=50 + idx * 20))
        # Include non-image metadata file that should be safely ignored
        zf.writestr("seq/notes.txt", "ignore me")
        zf.writestr("__MACOSX/._frame_0000.png", b"not an image")
    return buf.getvalue()


def _detection_config(with_homography: bool = True) -> PipelineConfig:
    homography = [[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]] if with_homography else None
    return PipelineConfig(
        model_name="yolox-s",
        track="detection",
        input_size=(640, 640),
        homography_matrix=homography,
    )


def test_healthz_loaded_and_unloaded() -> None:
    loaded_app = create_app(
        config=_detection_config(),
        engines={("onnxruntime", "fp32"): MockDetectionEngine()},
        active_runtime="onnxruntime",
        active_precision="fp32",
        device="cpu",
    )
    with TestClient(loaded_app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["engine_loaded"] is True
        assert body["active_runtime"] == "onnxruntime"
        assert body["precision"] == "fp32"
        assert body["device"] == "cpu"
        assert "onnxruntime" in body["available_runtimes"]

    unloaded_app = create_app(
        config=_detection_config(),
        engines={},
        active_runtime="onnxruntime",
        active_precision="fp32",
        device="cpu",
    )
    with TestClient(unloaded_app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["engine_loaded"] is False

        infer_resp = client.post(
            "/v1/infer/image",
            files={"file": ("frame.png", _make_png_bytes(), "image/png")},
        )
        assert infer_resp.status_code == 503


def test_infer_image_valid_homography_and_latency_breakdown() -> None:
    app = create_app(
        config=_detection_config(with_homography=True),
        engines={("onnxruntime", "fp32"): MockDetectionEngine()},
        active_runtime="onnxruntime",
        active_precision="fp32",
        device="cpu",
    )
    png_bytes = _make_png_bytes()

    with TestClient(app) as client:
        # Without homography
        resp_no_h = client.post(
            "/v1/infer/image",
            files={"file": ("frame.png", png_bytes, "image/png")},
        )
        assert resp_no_h.status_code == 200
        payload = resp_no_h.json()
        assert payload["runtime"] == "onnxruntime"
        assert payload["precision"] == "fp32"
        assert len(payload["detections"]) == 2
        assert all(det["world_xy"] is None for det in payload["detections"])

        latency = payload["latency_breakdown_ms"]
        for key in ("preproc", "h2d", "compute", "d2h", "postproc", "total"):
            assert key in latency
            assert isinstance(latency[key], float)
            assert latency[key] >= 0.0

        # With homography enabled via query parameter
        resp_with_h = client.post(
            "/v1/infer/image",
            params={"apply_homography": "true"},
            files={"file": ("frame.png", png_bytes, "image/png")},
        )
        assert resp_with_h.status_code == 200
        dets = resp_with_h.json()["detections"]
        assert len(dets) == 2
        person_det = next(det for det in dets if det["class_id"] == 0)
        assert person_det["class_name"] == "person"
        # Footpoint of (170, 250, 230, 350) is (200, 350) -> H*(200,350,1) = (410, 720)
        assert person_det["world_xy"] == pytest.approx([410.0, 720.0], abs=1e-2)

        # Also verify form-field override for apply_homography
        resp_form_h = client.post(
            "/v1/infer/image",
            files={"file": ("frame.png", png_bytes, "image/png")},
            data={"apply_homography": "true"},
        )
        assert resp_form_h.status_code == 200
        assert all(det["world_xy"] is not None for det in resp_form_h.json()["detections"])


def test_infer_image_invalid_bytes_and_unknown_runtime() -> None:
    app = create_app(
        config=_detection_config(),
        engines={("onnxruntime", "fp32"): MockDetectionEngine()},
        active_runtime="onnxruntime",
        active_precision="fp32",
    )
    with TestClient(app) as client:
        bad_resp = client.post(
            "/v1/infer/image",
            files={"file": ("corrupt.png", b"not-a-valid-image-payload", "image/png")},
        )
        assert bad_resp.status_code == 400

        unknown_rt = client.post(
            "/v1/infer/image",
            params={"runtime": "tensorrt", "precision": "int8"},
            files={"file": ("frame.png", _make_png_bytes(), "image/png")},
        )
        assert unknown_rt.status_code == 404


def test_infer_image_embedding_track() -> None:
    cfg = PipelineConfig(model_name="dinov2-vits14", track="embedding", input_size=(224, 224))
    app = create_app(
        config=cfg,
        engines={("onnxruntime", "fp32"): MockEmbeddingEngine(embedding_dim=384)},
        active_runtime="onnxruntime",
        active_precision="fp32",
    )
    with TestClient(app) as client:
        resp = client.post(
            "/v1/infer/image",
            files={"file": ("frame.png", _make_png_bytes(224, 224), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["detections"] == []
        assert data["embedding_dim"] == 384
        assert data["embedding_norm"] == pytest.approx(1.0, abs=1e-4)


def test_track_sequence_multifile_and_zip() -> None:
    # 1. Multi-file upload
    app_multi = create_app(
        config=_detection_config(with_homography=True),
        engines={("onnxruntime", "fp32"): MockDetectionEngine(shift_per_call_px=3.0)},
        active_runtime="onnxruntime",
        active_precision="fp32",
    )
    png_bytes = _make_png_bytes()
    multi_files = [("files", (f"frame_{idx:03d}.png", png_bytes, "image/png")) for idx in range(4)]

    with TestClient(app_multi) as client:
        resp = client.post(
            "/v1/infer/track-sequence",
            params={"apply_homography": "true"},
            files=multi_files,
        )
        assert resp.status_code == 200
        frames = resp.json()
        assert len(frames) == 4
        assert [frame["frame_index"] for frame in frames] == [0, 1, 2, 3]

        person_ids: list[int] = []
        for frame in frames:
            assert len(frame["detections"]) == 2
            for det in frame["detections"]:
                assert det["track_id"] is not None
                assert det["world_xy"] is not None
            person_det = next(d for d in frame["detections"] if d["class_id"] == 0)
            person_ids.append(int(person_det["track_id"]))

        # Smooth 3px/frame motion keeps the same ByteTrack track_id across all frames
        assert len(set(person_ids)) == 1

    # 2. Zip archive upload (via 'files' field as well as 'file' field)
    app_zip = create_app(
        config=_detection_config(with_homography=False),
        engines={("onnxruntime", "fp32"): MockDetectionEngine(shift_per_call_px=3.0)},
        active_runtime="onnxruntime",
        active_precision="fp32",
    )
    zip_bytes = _make_sequence_zip_bytes(num_frames=4)

    with TestClient(app_zip) as client:
        resp_zip = client.post(
            "/v1/infer/track-sequence",
            files={"files": ("sequence.zip", zip_bytes, "application/zip")},
        )
        assert resp_zip.status_code == 200
        zip_frames = resp_zip.json()
        assert len(zip_frames) == 4
        assert [frame["frame_index"] for frame in zip_frames] == [0, 1, 2, 3]
        first_ids = sorted(int(d["track_id"]) for d in zip_frames[0]["detections"])
        last_ids = sorted(int(d["track_id"]) for d in zip_frames[-1]["detections"])
        assert first_ids == last_ids

        # Also test single 'file' upload field for zip archive
        resp_zip_single = client.post(
            "/v1/infer/track-sequence",
            files={"file": ("sequence.zip", zip_bytes, "application/zip")},
        )
        assert resp_zip_single.status_code == 200
        assert len(resp_zip_single.json()) == 4


def test_benchmarks_pareto_filtering_and_sorting(tmp_path: Path) -> None:
    sample_results = [
        BenchmarkResult(
            model_name="yolox-s",
            track="detection",
            runtime="pytorch",
            precision="fp32",
            hardware_target="A100",
            p50_compute_ms=8.0,
            p95_compute_ms=9.5,
            p99_compute_ms=10.2,
            end_to_end_fps=95.0,
            clean_map50_95=0.405,
        ),
        BenchmarkResult(
            model_name="yolox-s",
            track="detection",
            runtime="tensorrt",
            precision="int8",
            hardware_target="A100",
            p50_compute_ms=1.1,
            p95_compute_ms=1.3,
            p99_compute_ms=1.5,
            end_to_end_fps=420.0,
            clean_map50_95=0.398,
        ),
        BenchmarkResult(
            model_name="yolox-s",
            track="detection",
            runtime="tensorrt",
            precision="fp16",
            hardware_target="A100",
            p50_compute_ms=1.8,
            p95_compute_ms=2.1,
            p99_compute_ms=2.4,
            end_to_end_fps=310.0,
            clean_map50_95=0.404,
        ),
        BenchmarkResult(
            model_name="dinov2-vits14",
            track="embedding",
            runtime="tensorrt",
            precision="fp16",
            hardware_target="A100",
            p50_compute_ms=2.9,
            p95_compute_ms=3.2,
            p99_compute_ms=3.5,
            end_to_end_fps=250.0,
            embedding_cosine_sim=0.9994,
        ),
    ]
    results_file = tmp_path / "bench_results.json"
    results_file.write_text(
        json.dumps([item.model_dump() for item in sample_results], indent=2),
        encoding="utf-8",
    )

    app = create_app(
        config=_detection_config(),
        engines={},
        results_dir=tmp_path,
    )

    with TestClient(app) as client:
        # Ascending latency sort for yolox-s
        resp_lat = client.get(
            "/v1/benchmarks/pareto",
            params={"model": "yolox-s", "metric": "p95_compute_ms"},
        )
        assert resp_lat.status_code == 200
        lat_rows = resp_lat.json()
        assert len(lat_rows) == 3
        assert [row["p95_compute_ms"] for row in lat_rows] == [1.3, 2.1, 9.5]

        # Descending FPS sort across detection track
        resp_fps = client.get(
            "/v1/benchmarks/pareto",
            params={"track": "detection", "metric": "end_to_end_fps"},
        )
        assert resp_fps.status_code == 200
        fps_rows = resp_fps.json()
        assert [row["end_to_end_fps"] for row in fps_rows] == [420.0, 310.0, 95.0]

        # Descending mAP sort across all models (None placed last)
        resp_map = client.get(
            "/v1/benchmarks/pareto",
            params={"metric": "clean_map50_95"},
        )
        assert resp_map.status_code == 200
        map_rows = resp_map.json()
        assert [row["clean_map50_95"] for row in map_rows] == [0.405, 0.404, 0.398, None]

        # Track filter for embedding track
        resp_emb = client.get(
            "/v1/benchmarks/pareto",
            params={"track": "embedding", "metric": "embedding_cosine_sim"},
        )
        assert resp_emb.status_code == 200
        emb_rows = resp_emb.json()
        assert len(emb_rows) == 1
        assert emb_rows[0]["model_name"] == "dinov2-vits14"

        # Invalid metric returns 400
        resp_bad = client.get("/v1/benchmarks/pareto", params={"metric": "not_a_metric"})
        assert resp_bad.status_code == 400


def test_lifespan_env_autoload_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("model_name: yolox-s\ntrack: detection\ninput_size: [640, 640]\n")

    mock_engine = MockDetectionEngine()
    monkeypatch.setattr("edge_vision_bench.api.app.create_engine", lambda **kwargs: mock_engine)
    monkeypatch.setenv("EVB_CONFIG", str(cfg_path))
    monkeypatch.setenv("EVB_RUNTIME", "onnxruntime")
    monkeypatch.setenv("EVB_PRECISION", "fp32")
    monkeypatch.setenv("EVB_DEVICE", "cpu")
    monkeypatch.setenv("EVB_ARTIFACT", "dummy.onnx")
    monkeypatch.setenv("EVB_WARMUP_RUNS", "2")
    monkeypatch.setenv("EVB_RESULTS_DIR", str(tmp_path))

    app = create_app()
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["engine_loaded"] is True
        assert mock_engine.warmup_runs_called == 2

        resp = client.post(
            "/v1/infer/image",
            files={"file": ("frame.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        assert len(resp.json()["detections"]) == 2

    assert mock_engine.closed is True
