"""Benchmarking datasets, OOD perturbations, profiling, and Pareto evaluation."""

from edge_vision_bench.bench.datasets import (
    COCO_TO_VISDRONE_MAP,
    VISDRONE_CLASSES,
    COCOSubsetDataset,
    MOTSequence,
    MOTSequenceDataset,
    VisDroneDETDataset,
    VisDroneMOTDataset,
    discover_coco_paths,
    generate_synthetic_mot_sequence,
)
from edge_vision_bench.bench.evaluator import (
    evaluate_coco_map,
    evaluate_embedding_parity,
    evaluate_mot_metrics,
    render_pareto_markdown,
    save_benchmark_results,
)
from edge_vision_bench.bench.profiler import (
    LatencyStats,
    NVMLMemoryMonitor,
    compute_artifact_size_mb,
    detect_hardware_target,
    profile_engine,
)

__all__ = [
    "COCO_TO_VISDRONE_MAP",
    "COCOSubsetDataset",
    "LatencyStats",
    "MOTSequence",
    "MOTSequenceDataset",
    "NVMLMemoryMonitor",
    "VISDRONE_CLASSES",
    "VisDroneDETDataset",
    "VisDroneMOTDataset",
    "compute_artifact_size_mb",
    "detect_hardware_target",
    "discover_coco_paths",
    "evaluate_coco_map",
    "evaluate_embedding_parity",
    "evaluate_mot_metrics",
    "generate_synthetic_mot_sequence",
    "profile_engine",
    "render_pareto_markdown",
    "save_benchmark_results",
]
