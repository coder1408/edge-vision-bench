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

### 4. Google Colab Pro GPU Benchmark Suite (NVIDIA L4 / A100)

Open [`notebooks/colab_gpu_benchmark.ipynb`](notebooks/colab_gpu_benchmark.ipynb) on a **Colab Pro NVIDIA L4 (24 GB) or A100 (40 GB / 80 GB)** runtime with Google Drive mounted. The notebook provides an idempotent end-to-end workflow that:
1. Persists **COCO `val2017`** and **VisDrone2019-DET / VisDrone2019-MOT** onto Google Drive (`/content/drive/MyDrive/edge-vision-bench-data/`) with partial-download resume checks so downloads happen only once.
2. Exports Track A (`yolox-s`, `yolox-m`) and Track B (`dinov2-vits14`) to FP32/FP16 ONNX, passes the parity gate, and builds TensorRT 10.x/11.x `FP32` / `FP16` / 500-frame entropy-calibrated `INT8` `.engine` plans.
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

All **28 benchmark runs** (24 on **NVIDIA L4 24 GB** with TensorRT 11.5 / CUDA + 4 local CPU runs on **AMD Ryzen 7 5800HS**) are persisted in `results/*.json` and compiled into [`results/pareto_summary.md`](results/pareto_summary.md) via `evb report`.

### Track A — Detection & Multi-Object Tracking (`NVIDIA L4`, `batch=1`)

Evaluated on 500 strictly disjoint COCO `val2017` eval frames (with 500 disjoint calibration frames for INT8 Q/DQ) plus cross-domain zero-shot **VisDrone2019-MOT** aerial tracking (`COCO_TO_VISDRONE_MAP` + clean-room ByteTrack).

| Model | Runtime | Prec | p50 / p95 / p99 (ms) | H2D / Compute / D2H (ms) | FPS | VRAM (MB) | Size (MB) | Clean mAP50:95 | OOD Blur / Occ / LowLight (Δ) | Clean MOTA / IDF1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolox-s`** | **`tensorrt`** | **`fp16`** | **1.50 / 1.55 / 1.66** | 1.08 / 1.50 / 0.66 | **127.8** | 470.0 | 20.49 | **0.422** | 0.190 (-0.233) / 0.162 (-0.261) / 0.371 (-0.051) | **0.245 / 0.411** |
| `yolox-s` | `tensorrt` | `fp32` | 2.44 / 2.47 / 2.49 | 1.08 / 2.44 / 0.69 | 112.0 | 812.0 | 37.87 | 0.423 | 0.190 (-0.232) / 0.162 (-0.261) / 0.371 (-0.052) | 0.243 / 0.405 |
| `yolox-s` | `pytorch_compile` | `fp16` | 2.70 / 3.05 / 3.28 | 0.87 / 2.70 / 0.70 | 111.5 | 362.0 | — | 0.422 | 0.191 (-0.232) / 0.162 (-0.261) / 0.371 (-0.051) | 0.243 / 0.405 |
| `yolox-s` | `onnxruntime` | `fp32` | 3.32 / 3.62 / 4.61 | 1.13 / 3.32 / 0.58 | 102.7 | 344.0 | 34.33 | 0.423 | 0.190 (-0.233) / 0.162 (-0.261) / 0.371 (-0.052) | 0.243 / 0.405 |
| `yolox-s` | `onnxruntime` | `fp16` | 3.43 / 3.68 / 4.77 | 1.13 / 3.43 / 0.59 | 102.6 | 286.0 | 17.21 | 0.422 | 0.190 (-0.232) / 0.162 (-0.261) / 0.371 (-0.051) | 0.243 / 0.405 |
| `yolox-s` | `tensorrt` | `int8` | 3.67 / 3.95 / 4.19 | 1.08 / 3.67 / 0.68 | 99.1 | 766.0 | **11.96** | 0.369 | 0.122 (-0.247) / 0.150 (-0.219) / 0.302 (-0.068) | 0.243 / 0.391 |
| `yolox-s` | `pytorch` | `fp32` | 12.34 / 13.17 / 15.59 | 1.14 / 12.34 / 0.67 | 52.3 | 328.0 | — | 0.423 | 0.190 (-0.233) / 0.162 (-0.261) / 0.371 (-0.052) | 0.243 / 0.407 |
| `yolox-s` | `pytorch` | `fp16` | 14.41 / 15.11 / 16.55 | 0.90 / 14.41 / 0.79 | 47.4 | 288.0 | — | 0.422 | 0.190 (-0.232) / 0.162 (-0.260) / 0.371 (-0.050) | 0.243 / 0.405 |
| **`yolox-m`** | **`tensorrt`** | **`fp16`** | **2.47 / 2.53 / 2.62** | 1.06 / 2.47 / 0.64 | **115.2** | 606.0 | 51.91 | **0.493** | 0.253 (-0.240) / 0.209 (-0.284) / 0.438 (-0.056) | 0.212 / **0.467** |
| `yolox-m` | `tensorrt` | `int8` | 4.71 / 4.93 / 5.74 | 1.06 / 4.71 / 0.70 | 90.7 | 1024.0 | **28.66** | 0.405 | 0.191 (-0.214) / 0.186 (-0.220) / 0.343 (-0.062) | **0.222** / 0.440 |
| `yolox-m` | `tensorrt` | `fp32` | 4.91 / 4.94 / 4.96 | 1.07 / 4.91 / 0.61 | 89.0 | 1146.0 | 99.92 | 0.494 | 0.253 (-0.241) / 0.209 (-0.285) / 0.438 (-0.056) | 0.211 / 0.466 |
| `yolox-m` | `onnxruntime` | `fp16` | 4.78 / 5.01 / 6.13 | 1.18 / 4.78 / 0.68 | 84.7 | 350.0 | 48.42 | 0.493 | 0.253 (-0.240) / 0.209 (-0.284) / 0.438 (-0.055) | 0.212 / 0.467 |
| `yolox-m` | `pytorch_compile` | `fp16` | 4.57 / 5.09 / 5.40 | 0.88 / 4.57 / 0.65 | 92.6 | 460.0 | — | 0.494 | 0.253 (-0.241) / 0.209 (-0.284) / 0.438 (-0.056) | 0.212 / 0.466 |
| `yolox-m` | `onnxruntime` | `fp32` | 6.52 / 6.65 / 6.70 | 1.13 / 6.52 / 0.58 | 77.1 | 472.0 | 96.72 | 0.494 | 0.253 (-0.241) / 0.209 (-0.285) / 0.438 (-0.056) | 0.212 / 0.466 |
| `yolox-m` | `pytorch` | `fp32` | 16.38 / 17.67 / 20.31 | 1.18 / 16.38 / 0.79 | 42.3 | 448.0 | — | 0.494 | 0.253 (-0.241) / 0.209 (-0.285) / 0.437 (-0.057) | 0.212 / 0.466 |
| `yolox-m` | `pytorch` | `fp16` | 18.07 / 18.87 / 19.89 | 0.94 / 18.07 / 0.83 | 40.0 | 352.0 | — | 0.493 | 0.253 (-0.240) / 0.209 (-0.284) / 0.438 (-0.055) | 0.212 / 0.467 |

### Track A — Local CPU Smoke Benchmark (`AMD Ryzen 7 5800HS`, `yolox-s`, `batch=1`)

| Model | Runtime | Prec | p50 / p95 / p99 (ms) | End-to-End FPS | Peak RAM (MB) | Size (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`yolox-s`** | **`openvino`** | **`fp16`** | **58.55 / 61.70 / 61.82** | **15.7** | 699.8 | **17.49** |
| `yolox-s` | `openvino` | `fp32` | 60.09 / 71.83 / 76.56 | 14.9 | 716.1 | 34.51 |
| `yolox-s` | `onnxruntime` | `fp32` | 75.24 / 139.60 / 147.32 | 10.6 | 628.3 | 34.33 |
| `yolox-s` | `pytorch` | `fp32` | 96.67 / 119.07 / 121.97 | 9.6 | 596.2 | — |

### Track B — Vision Transformer Embeddings (`dinov2-vits14`, `518x518`, `NVIDIA L4`, `batch=1`)

| Model | Runtime | Prec | p50 / p95 / p99 (ms) | H2D / Compute / D2H (ms) | FPS | VRAM (MB) | Size (MB) | Clean CosSim | OOD Blur / Occ / LowLight CosSim (Δ) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dinov2-vits14`** | **`tensorrt`** | **`fp16`** | **2.21 / 2.24 / 2.28** | 0.75 / 2.21 / 0.02 | **96.7** | 464.0 | 44.93 | **1.0000** | 0.7210 (-0.2789) / 0.8720 (-0.1280) / 0.9498 (-0.0502) |
| `dinov2-vits14` | `onnxruntime` | `fp16` | 5.56 / 5.67 / 5.72 | 0.83 / 5.56 / 0.07 | 71.8 | 346.0 | 42.18 | 1.0000 | 0.7208 (-0.2792) / 0.8719 (-0.1281) / 0.9498 (-0.0501) |
| `dinov2-vits14` | `pytorch_compile` | `fp16` | 5.70 / 6.02 / 6.12 | 0.66 / 5.70 / 0.07 | 71.5 | 468.0 | — | 1.0000 | 0.7208 (-0.2792) / 0.8718 (-0.1282) / 0.9498 (-0.0502) |
| `dinov2-vits14` | `tensorrt` | `int8` | 6.36 / 6.39 / 6.40 | 0.77 / 6.36 / 0.02 | 66.5 | 858.0 | **25.82** | **0.0335** | 0.0223 (-0.0112) / 0.0410 (+0.0075) / 0.0252 (-0.0083) |
| `dinov2-vits14` | `pytorch` | `fp16` | 9.25 / 10.50 / 11.58 | 0.69 / 9.25 / 0.09 | 55.8 | 394.0 | — | 1.0000 | 0.7208 (-0.2792) / 0.8719 (-0.1281) / 0.9498 (-0.0502) |
| `dinov2-vits14` | `tensorrt` | `fp32` | 12.93 / 13.54 / 13.58 | 0.76 / 12.93 / 0.02 | 46.5 | 1954.0 | 85.58 | 1.0000 | 0.7208 (-0.2792) / 0.8719 (-0.1281) / 0.9499 (-0.0501) |
| `dinov2-vits14` | `onnxruntime` | `fp32` | 13.41 / 13.98 / 14.04 | 0.85 / 13.41 / 0.08 | 45.1 | 468.0 | 84.25 | 1.0000 | 0.7208 (-0.2792) / 0.8719 (-0.1281) / 0.9499 (-0.0501) |
| `dinov2-vits14` | `pytorch` | `fp32` | 18.83 / 19.70 / 19.83 | 0.84 / 18.83 / 0.08 | 36.3 | 446.0 | — | 1.0000 | 0.7209 (-0.2791) / 0.8719 (-0.1281) / 0.9499 (-0.0501) |

### Key Engineering Findings

1. **TensorRT 11 FP16 Dominates the Pareto Frontier Across Both Tracks**:
   - **`yolox-s`**: `1.50 ms` p50 compute (`1.55 ms` p95), **`127.8 FPS`** end-to-end, **`8.2x` compute speedup** over eager PyTorch FP32 (`12.34 ms`), with zero mAP loss (`0.422` vs `0.423`).
   - **`yolox-m`**: `2.47 ms` p50 compute (`2.53 ms` p95), **`115.2 FPS`**, **`6.6x` compute speedup** over PyTorch FP32 (`16.38 ms`), preserving **`0.493` COCO `mAP@50:95`**.
   - **`dinov2-vits14`**: `2.21 ms` p50 compute (`2.24 ms` p95), **`96.7 FPS`**, **`8.5x` compute speedup** over PyTorch FP32 (`18.83 ms`), with **`1.0000` (`0.99998`)** embedding cosine similarity.
   - On CPU (`AMD Ryzen 7 5800HS`), **OpenVINO FP16** achieves `58.55 ms` p50 (`15.7 FPS`) with `1.93x` tighter p95 tail latency than PyTorch CPU (`61.70 ms` vs `119.07 ms`) at half the model footprint (`17.49 MB`).
2. **End-to-End Bottleneck Attribution (Amdahl's Law Beyond GPU Compute)**:
   - Once TensorRT FP16 compresses GPU compute to `1.50 ms` (`yolox-s`) and `2.21 ms` (`dinov2-vits14`), **host-side preprocessing, postprocessing, and PCIe transfers become the primary wall-clock bottleneck**.
   - On `yolox-s` (`tensorrt/fp16`), GPU compute is only **~19.5%** of the frame budget, while CPU letterbox preproc (`2.38 ms`), CPU NMS postproc (`2.07 ms`), H2D (`1.08 ms`), and D2H (`0.66 ms`) consume **`6.19 ms` (~80.5%)**.
   - On `dinov2-vits14` (`tensorrt/fp16`), CPU `518x518` bicubic ImageNet preproc alone consumes **`7.19 ms`** (~70% of end-to-end latency), explaining why an `8.5x` compute speedup yields `2.66x` end-to-end FPS (`36.3 -> 96.7 FPS`) and pointing directly to GPU-accelerated preprocessing (DALI / CUDA kernels) and pinned async double-buffering as the next high-leverage optimization.
3. **CNN vs. Vision Transformer INT8 PTQ Divergence (Why Parity Gating Is Mandatory)**:
   - On CNN detectors (`yolox-s`, `yolox-m`), 500-frame disjoint COCO symmetric Q/DQ entropy calibration shrinks serialized TensorRT engines **~3.2x–3.5x** (`37.87 -> 11.96 MB` and `99.92 -> 28.66 MB`) while retaining usable detection accuracy (`0.369` and `0.405` mAP, though `batch=1` explicit Q/DQ reformat nodes add compute overhead relative to pure FP16).
   - On the Vision Transformer (`dinov2-vits14`), naive symmetric post-training INT8 calibration **catastrophically collapses** embedding fidelity to **`CosSim = 0.0335`** due to extreme activation outliers in self-attention / LayerNorm projections — empirically proving why automated numerical parity gates (`min_cosine >= 0.995`) are critical in production pipelines and why ViTs require FP16 or outlier-aware quantization (SmoothQuant / QAT).
4. **OOD Robustness & Zero-Shot Aerial Domain Shift**:
   - Across both detectors, **30% cutout occlusion** (`-0.261` to `-0.284` mAP) and **15 px motion blur** (`-0.233` to `-0.240` mAP) cause severe recall degradation, whereas **low-light gamma (`gamma=2.4`) + read noise** causes only a modest **`-0.051` to `-0.056` mAP** drop.
   - Interestingly, DINOv2's global self-attention is substantially more resilient to spatial cutout (`-0.128` CosSim, `0.8720`) than to high-frequency motion blur (`-0.279` CosSim, `0.7210`).
   - On zero-shot cross-domain **VisDrone2019-MOT**, COCO-pretrained weights + ByteTrack achieve `0.245 MOTA / 0.411 IDF1` (`yolox-s`) and `0.212 MOTA / 0.467 IDF1` (`yolox-m`), quantifying aerial viewpoint & small-object domain shift cleanly without conflating label-space differences.

Full raw table across all 28 runs: [`results/pareto_summary.md`](results/pareto_summary.md).

---

## Repository Hygiene

- Binary model artifacts (`*.onnx`, `*.engine`, `*.pth`, `*.bin`, `*.xml`, `*.cache`) and local datasets (`data/`, `models/artifacts/`, `models/weights/`) are strictly gitignored.
- Lightweight JSON benchmark telemetry (`results/*.json`) and Markdown Pareto reports (`results/*.md`) remain tracked for reproducible cross-hardware comparison.
- CI (`.github/workflows/ci.yml`) enforces `ruff check`, `ruff format --check`, `mypy src`, fast CPU `pytest`, and CPU Docker image builds on every push and pull request.
