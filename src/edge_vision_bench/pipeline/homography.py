"""Planar projective homography for image-to-world ground-plane mapping."""

from __future__ import annotations

import cv2
import numpy as np

from edge_vision_bench.schemas import DetectionBox

_SINGULARITY_EPS = 1e-12


class PlanarHomography:
    """Validated 3x3 projective homography projecting image pixels onto a world plane."""

    def __init__(self, matrix: np.ndarray | list[list[float]]) -> None:
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(f"Homography matrix must have shape (3, 3), got {mat.shape!r}")
        if not np.all(np.isfinite(mat)):
            raise ValueError("Homography matrix must contain only finite values")
        det = float(np.linalg.det(mat))
        if abs(det) < _SINGULARITY_EPS:
            raise ValueError(f"Homography matrix is singular (det={det:.3e})")
        self._matrix: np.ndarray = mat.copy()

    @property
    def matrix(self) -> np.ndarray:
        """Return a defensive float64 copy of the 3x3 homography matrix."""
        return self._matrix.copy()

    def project_point(self, x: float, y: float) -> tuple[float, float]:
        """Project an image pixel ``(x, y)`` into world plane coordinates ``(X, Y)``."""
        homog = np.array([float(x), float(y), 1.0], dtype=np.float64)
        projected = self._matrix @ homog
        w = float(projected[2])
        if abs(w) < _SINGULARITY_EPS:
            raise ValueError(f"Homogeneous denominator w={w:.3e} is near zero for point ({x}, {y})")
        return (float(projected[0] / w), float(projected[1] / w))

    def project_detections(self, detections: list[DetectionBox]) -> list[DetectionBox]:
        """Project each detection's bottom-center footpoint into ``world_xy``.

        Returns new ``DetectionBox`` copies without mutating the input boxes.
        """
        projected_boxes: list[DetectionBox] = []
        for det in detections:
            foot_x = (det.x1 + det.x2) * 0.5
            foot_y = det.y2
            world_xy = self.project_point(foot_x, foot_y)
            projected_boxes.append(det.model_copy(update={"world_xy": world_xy}))
        return projected_boxes

    @classmethod
    def from_correspondences(
        cls,
        image_points: np.ndarray,
        world_points: np.ndarray,
    ) -> PlanarHomography:
        """Estimate a planar homography from >= 4 corresponding 2D point pairs."""
        src = np.asarray(image_points, dtype=np.float64)
        dst = np.asarray(world_points, dtype=np.float64)

        if src.ndim != 2 or src.shape[1] != 2:
            raise ValueError(f"image_points must have shape (N, 2), got {src.shape!r}")
        if dst.ndim != 2 or dst.shape[1] != 2:
            raise ValueError(f"world_points must have shape (N, 2), got {dst.shape!r}")
        if src.shape[0] != dst.shape[0]:
            raise ValueError(
                f"Point count mismatch: image_points has {src.shape[0]}, "
                f"world_points has {dst.shape[0]}"
            )
        if src.shape[0] < 4:
            raise ValueError(f"At least 4 point correspondences are required, got {src.shape[0]}")

        homography_mat, _ = cv2.findHomography(src, dst, method=0)
        if homography_mat is None:
            raise ValueError("cv2.findHomography failed to estimate a valid matrix")
        return cls(homography_mat)
