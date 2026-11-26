# Edge Vision Bench — Pareto Benchmark Summary

**Hardware Targets:** AMD Ryzen 7 5800HS with Radeon Graphics  
**Total Benchmark Runs:** 4

## Track A — Detection & Multi-Object Tracking

| Model | Runtime | Prec | Hardware | B | p50 / p95 / p99 (ms) | H2D / Compute / D2H (ms) | FPS | VRAM (MB) | RAM (MB) | Size (MB) | Clean mAP50:95 | OOD Blur / Occ / LowLight (Δ) | Clean MOTA / IDF1 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolox-s` | `openvino` | `fp16` | AMD Ryzen 7 5800HS with Radeon Graphics | 1 | 58.55 / 61.70 / 61.82 | 0.00 / 58.55 / 0.00 | 15.7 | 0.0 | 699.8 | 17.49 | — | — / — / — | — / — |
| `yolox-s` | `openvino` | `fp32` | AMD Ryzen 7 5800HS with Radeon Graphics | 1 | 60.09 / 71.83 / 76.56 | 0.00 / 60.09 / 0.00 | 14.9 | 0.0 | 716.1 | 34.51 | — | — / — / — | — / — |
| `yolox-s` | `pytorch` | `fp32` | AMD Ryzen 7 5800HS with Radeon Graphics | 1 | 96.67 / 119.07 / 121.97 | 0.00 / 96.67 / 0.00 | 9.6 | 0.0 | 596.2 | 0.00 | — | — / — / — | — / — |
| `yolox-s` | `onnxruntime` | `fp32` | AMD Ryzen 7 5800HS with Radeon Graphics | 1 | 75.24 / 139.60 / 147.32 | 0.00 / 75.24 / 0.00 | 10.6 | 0.0 | 628.3 | 34.33 | — | — / — / — | — / — |
