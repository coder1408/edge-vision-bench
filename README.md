# Edge Vision Bench (`edge-vision-bench`)

A production-oriented computer vision deployment, cross-runtime parity verification, OOD stress benchmarking, and FastAPI inference serving suite.

`edge-vision-bench` bridges the gap between research checkpoints and latency-constrained edge deployment by enforcing **automated numerical parity gates**, **microsecond-level latency stage decomposition** (`preproc` / `h2d` / `compute` / `d2h` / `postproc` / `track`), **out-of-distribution (OOD) robustness profiling**, and **reproducible Pareto trade-off reporting** across PyTorch, `torch.compile`, ONNX Runtime, OpenVINO, and TensorRT 10.x.

---

## Dual-Track Architecture

| Track | Models | Downstream Pipeline | Primary Metrics |
| :--- | :--- | :--- | :--- |
| **Track A: Detection + MOT** | Self-contained Apache-2.0 PyTorch `YOLOX-s` & `YOLOX-m` (loads official Megvii `.pth` weights with `strict=True`) | Letterbox preproc -> Pure-NumPy class-aware NMS -> Clean-room 8-state Kalman **ByteTrack** -> 3x3 **Planar Homography** footpoint projection | p50/p95/p99 latency, End-to-End FPS, Peak VRAM/RAM, Artifact Size, COCO `mAP@50:95` (clean + OOD), VisDrone2019-MOT `MOTA` / `IDF1` |
| **Track B: ViT Embeddings** | `DINOv2` (`vit_small_patch14_dinov2.lvd142m`) with ONNX-safe self-attention & L2-normalized outputs | Bicubic RGB ImageNet preproc -> Embedding extraction -> Cosine parity & OOD drift analysis | p50/p95/p99 latency, End-to-End FPS, Peak VRAM/RAM, Embedding Cosine Similarity (clean + OOD drift) |

### High-Level Design (HLD)

```text
                       +-------------------------------------------+
                       |       YAML Configs (configs/*.yaml)       |
                       | yolox_s | yolox_m | dinov2_vits14 | trt   |
                       +---------------------+---------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                        1. Offline Export & Parity Gate (`evb`)                          |
|  PyTorch (FP32) ---> ONNX (opset 17, dynamic batch) ---> Parity Gate (rtol/atol/cos)    |
|                            |                                                            |
|                            +---> OpenVINO IR (.xml/.bin): FP32 / FP16 / NNCF INT8       |
|                            +---> TensorRT 10.x (.engine): FP32 / FP16 / Entropy INT8    |
+--------------------------------------------+--------------------------------------------+
                                             |
                     +-----------------------+-----------------------+
                     |                                               |
                     v                                               v
+--------------------------------------------+  +-----------------------------------------+
|     2. Benchmark & OOD Stress Harness      |  |       3. FastAPI Serving Layer          |
|  - Unified `BaseRuntimeEngine` interface   |  |  - GET  /healthz                        |
|  - Stage timing: preproc/h2d/compute/d2h   |  |  - POST /v1/infer/image                 |
|  - NVML VRAM + psutil RSS monitoring       |  |  - POST /v1/infer/track-sequence        |
|  - Disjoint COCO calib (500) / eval (500)  |  |         (multi-file or .zip archive)    |
|  - VisDrone2019-MOT cross-domain eval      |  |  - GET  /v1/benchmarks/pareto           |
|  - Deterministic OOD suite:                |  |  - Docker CPU / GPU + Compose profiles  |
|    MotionBlur | CutoutOcclusion | LowLight |  +--------------------+--------------------+
+--------------------+-----------------------+                       |
                     |                                               |
                     v                                               |
+--------------------------------------------+                       |
|      4. Persisted Pareto Artifacts         |<----------------------+
|  results/*.json  &  results/pareto_summary.md  (git-tracked, zero binary bloat)         |
+--------------------------------------------+
```

---

## Quickstart

### 1. Local CPU Setup (Linux / macOS, Python 3.10–3.12)

```bash
git clone https://github.com/coder1408/edge-vision-bench.git
cd edge-vision-bench

# Install lightweight CPU PyTorch wheels first, then CPU + dev extras
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[cpu,dev]"

# Verify CLI and run fast offline test suite
evb --help
pytest -v -m "not slow and not gpu"
```

### 2. Local CPU End-to-End Smoke Run (Export -> Parity -> Bench -> Report)

```bash
# Export YOLOX-s to ONNX and OpenVINO (FP32 & FP16)
evb export --config configs/yolox_s_sportsmot.yaml --runtime onnx --precision fp32
evb export --config configs/yolox_s_sportsmot.yaml --runtime openvino --precision fp32
evb export --config configs/yolox_s_sportsmot.yaml --runtime openvino --precision fp16

# Run strict numerical parity gate against PyTorch FP32
evb verify-parity \
  --config configs/yolox_s_sportsmot.yaml \
  --artifact models/artifacts/yolox-s.onnx \
  --runtime onnxruntime

# Benchmark ONNX Runtime CPU and OpenVINO CPU
evb bench \
  --config configs/yolox_s_sportsmot.yaml \
  --runtime onnxruntime --precision fp32 --device cpu \
  --artifact models/artifacts/yolox-s.onnx \
  --num-frames 50 --warmup 10 \
  --output-json results/cpu_yolox_s_ort_fp32.json

evb bench \
  --config configs/yolox_s_sportsmot.yaml \
  --runtime openvino --precision fp16 --device cpu \
  --artifact models/artifacts/openvino/yolox-s_fp16.xml \
  --num-frames 50 --warmup 10 \
  --output-json results/cpu_yolox_s_ov_fp16.json

# Generate unified Pareto Markdown report
evb report --results-dir results --output results/pareto_summary.md
```

### 3. Docker & Docker Compose

```bash
# Start CPU inference API (mounts ./models/artifacts and ./results)
EVB_ARTIFACT=/app/models/artifacts/yolox-s.onnx \
  docker compose --profile api-cpu up --build

# Start NVIDIA GPU TensorRT API (requires NVIDIA Container Toolkit)
EVB_ARTIFACT=/app/models/artifacts/yolox-s_fp16.engine \
  docker compose --profile api-gpu up --build

# Run CLI benchmark inside the container
docker compose --profile bench-runner run --rm bench-runner \
  report --results-dir /app/results --output /app/results/pareto_summary.md
```

### 4. Google Colab Pro A100 Full GPU Benchmark Suite

Open [`notebooks/colab_gpu_benchmark.ipynb`](notebooks/colab_gpu_benchmark.ipynb) on a **Colab Pro A100 (40 GB / 80 GB)** runtime with Google Drive mounted. The notebook provides an idempotent end-to-end workflow that:
1. Persists **COCO `val2017`** and **VisDrone2019-DET / VisDrone2019-MOT** onto Google Drive (`/content/drive/MyDrive/edge-vision-bench-data/`) so downloads happen only once.
2. Exports Track A (`yolox-s`, `yolox-m`) and Track B (`dinov2-vits14`) to ONNX, passes the parity gate, and builds TensorRT 10.x `FP32` / `FP16` / 500-frame entropy-calibrated `INT8` `.engine` plans.
3. Sweeps PyTorch (`FP32`/`FP16`), `torch.compile` (`FP16`), ORT-CUDA (`FP32`/`FP16`), and TensorRT (`FP32`/`FP16`/`INT8`) across clean + OOD COCO `mAP@50:95`, VisDrone2019-MOT `MOTA`/`IDF1`, and DINOv2 cosine drift.
4. Renders the unified Pareto Markdown report and optionally commits lightweight `results/*.json` & `results/pareto_summary.md` back to GitHub (zero `.engine` binary bloat).

---

## CLI Reference (`evb`)

| Command | Purpose | Key Flags |
| :--- | :--- | :--- |
| `evb export` | Export PyTorch model to ONNX, OpenVINO IR, or TensorRT `.engine` | `--config`, `--runtime`, `--precision`, `--pretrained/--no-pretrained`, `--calib-dir` |
| `evb verify-parity` | Run deterministic numerical parity check vs PyTorch FP32 | `--config`, `--artifact`, `--runtime`, `--pretrained/--no-pretrained` |
| `evb bench` | Profile latency breakdown, VRAM/RAM, accuracy, MOT, and OOD robustness | `--config`, `--runtime`, `--precision`, `--device`, `--artifact`, `--num-frames`, `--warmup`, `--ood/--no-ood`, `--coco-dir`, `--mot-dir`, `--output-json`, `--output-md` |
| `evb serve` | Launch Uvicorn FastAPI inference server | `--config`, `--runtime`, `--precision`, `--device`, `--artifact`, `--host`, `--port` |
| `evb report` | Aggregate all `results/*.json` files into a unified Pareto Markdown table | `--results-dir`, `--output` |

---

## REST API Contract

Start the server locally via `evb serve` or Uvicorn:

```bash
EVB_CONFIG=configs/yolox_s_sportsmot.yaml \
EVB_RUNTIME=onnxruntime \
EVB_PRECISION=fp32 \
EVB_DEVICE=cpu \
EVB_ARTIFACT=models/artifacts/yolox-s.onnx \
uvicorn edge_vision_bench.api.app:app --host 0.0.0.0 --port 8000
```

| Method & Path | Input | Response Schema | Notes |
| :--- | :--- | :--- | :--- |
| `GET /healthz` | None | `HealthResponse` | Reports `status` (`ok`/`degraded`), `active_runtime`, `precision`, `device`, `engine_loaded`, `available_runtimes`. |
| `POST /v1/infer/image` | `multipart/form-data`: `file` + optional query/form `runtime`, `precision`, `apply_homography` | `InferenceResponse` | Returns detections (or `embedding_dim`/`embedding_norm` for Track B) + `latency_breakdown_ms` (`preproc`, `h2d`, `compute`, `d2h`, `postproc`, `total`). Returns `400` on invalid image, `404` on unregistered runtime/precision, `503` if no engine is loaded. |
| `POST /v1/infer/track-sequence` | `multipart/form-data`: multiple ordered `files` **or** a single `.zip` archive (`files` or `file`) + optional `runtime`, `precision`, `apply_homography` | `list[InferenceResponse]` | Instantiates a fresh `ByteTracker` per request, sorts zip image members naturally in memory, and populates `frame_index`, `track_id`, and optional `world_xy`. |
| `GET /v1/benchmarks/pareto` | Query params: `?model=...&metric=p95_compute_ms&track=...` | `list[BenchmarkResult]` | Loads all persisted JSON benchmark files from `results_dir`, filters by `model`/`track`, sorts ascending for latency/memory/size metrics or descending for FPS/mAP/MOTA/cosine metrics. |

### Example Requests

```bash
# Health check
curl -s http://localhost:8000/healthz | jq .

# Single-image inference with ground-plane homography projection
curl -s -X POST "http://localhost:8000/v1/infer/image?apply_homography=true" \
  -F "file=@sample.png" | jq .

# Multi-frame tracking via zip archive
curl -s -X POST "http://localhost:8000/v1/infer/track-sequence?apply_homography=false" \
  -F "file=@frames.zip" | jq '.[0]'

# Query Pareto frontier sorted by p95 compute latency
curl -s "http://localhost:8000/v1/benchmarks/pareto?model=yolox-s&metric=p95_compute_ms" | jq .
```

---

## Methodology: Parity, Calibration & OOD Robustness

1. **Automated Numerical Parity Gate**:
   - Every exported ONNX / OpenVINO artifact is validated against PyTorch FP32 on deterministic seeded inputs (`seed=42`).
   - **Track A (Detection)**: enforces `np.allclose(pt, target, rtol=1e-3, atol=1e-4)` across full decoded `(B, N, 5 + C)` tensors.
   - **Track B (Embedding)**: enforces mean cosine similarity `>= 0.995` on L2-normalized DINOv2 embeddings.
2. **Leakage-Free INT8 Calibration**:
   - `COCOSubsetDataset` deterministically shuffles COCO `val2017` (`seed=42`) and splits it into strictly **disjoint** `calibration` (500 images for TensorRT `IInt8EntropyCalibrator2` / OpenVINO NNCF) and `eval` (500 images for `pycocotools` `mAP@50:95`) partitions, verified by a runtime disjointness assertion.
3. **Deterministic OOD Stress Suite**:
   - **Motion Blur** (`MotionBlurTransform`, 15 px linear PSF kernel) simulates camera pan/drone vibration.
   - **Cutout Occlusion** (`CutoutOcclusionTransform`, 30% area across 4 seeded rectangular masks) tests partial visibility and ByteTrack low-confidence recovery.
   - **Low-Light Gamma + Read Noise** (`LowLightGammaTransform`, `gamma=2.4`, Gaussian sensor noise) evaluates night/indoor exposure degradation.
4. **Cross-Domain Aerial Evaluation (VisDrone2019-MOT)**:
   - Pretrained COCO YOLOX checkpoints are evaluated on VisDrone2019-MOT through an explicit, transparent `COCO_TO_VISDRONE_MAP` (`person->pedestrian`, `bicycle->bicycle`, `car->car`, `motorcycle->motor`, `bus->bus`, `truck->truck`), filtering GT to COCO-compatible classes so domain-shift degradation is isolated cleanly from label-space mismatch.

---

## Benchmark Results & Pareto Frontier

All benchmark runs persist structured JSON files to `results/*.json` and compile into [`results/pareto_summary.md`](results/pareto_summary.md) via `evb report`.

### Local CPU Smoke Summary (AMD Ryzen / x86_64)

> Run `evb bench` locally and regenerate [`results/pareto_summary.md`](results/pareto_summary.md) to populate host-specific CPU latency, RAM, and artifact size numbers across PyTorch CPU, ONNX Runtime CPU, and OpenVINO CPU (`FP32` / `FP16`).

### Colab Pro A100 GPU Summary (Track A & Track B)

> Execute [`notebooks/colab_gpu_benchmark.ipynb`](notebooks/colab_gpu_benchmark.ipynb) on an NVIDIA A100 runtime to populate full FP32 / FP16 / INT8 TensorRT 10.x, ORT-CUDA, `torch.compile`, and PyTorch Pareto tables covering clean + OOD COCO `mAP@50:95`, VisDrone2019-MOT `MOTA`/`IDF1`, and DINOv2 cosine drift.

---

## Repository Hygiene

- Binary model artifacts (`*.onnx`, `*.engine`, `*.pth`, `*.bin`, `*.xml`, `*.cache`) and local datasets (`data/`, `models/artifacts/`, `models/weights/`) are strictly gitignored.
- Lightweight JSON benchmark telemetry (`results/*.json`) and Markdown Pareto reports (`results/*.md`) remain tracked for reproducible cross-hardware comparison.
- CI (`.github/workflows/ci.yml`) enforces `ruff check`, `ruff format --check`, `mypy src`, fast CPU `pytest`, and CPU Docker image builds on every push and pull request.
