"""Clean-room ByteTrack multi-object tracker using NumPy Kalman filtering and SciPy LAP."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from scipy.optimize import linear_sum_assignment

from edge_vision_bench.schemas import DetectionBox

_INVALID_COST = 1e5
_MIN_BOX_SIZE = 1e-3


class TrackState(StrEnum):
    """Lifecycle states of a tracked object."""

    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


def _xyxy_to_xyah(tlbr: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` box to ``[cx, cy, aspect_ratio, h]``."""
    w = max(float(tlbr[2] - tlbr[0]), _MIN_BOX_SIZE)
    h = max(float(tlbr[3] - tlbr[1]), _MIN_BOX_SIZE)
    cx = float(tlbr[0]) + 0.5 * w
    cy = float(tlbr[1]) + 0.5 * h
    aspect = w / h
    return np.array([cx, cy, aspect, h], dtype=np.float64)


def _xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, aspect_ratio, h]`` state back to ``[x1, y1, x2, y2]``."""
    cx, cy, aspect, h = (
        float(xyah[0]),
        float(xyah[1]),
        float(xyah[2]),
        float(xyah[3]),
    )
    h = max(h, _MIN_BOX_SIZE)
    w = max(aspect * h, _MIN_BOX_SIZE)
    half_w = 0.5 * w
    half_h = 0.5 * h
    return np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=np.float64)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix of shape ``(M, N)`` between ``[x1, y1, x2, y2]`` boxes."""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    a = boxes_a[:, np.newaxis, :]  # (M, 1, 4)
    b = boxes_b[np.newaxis, :, :]  # (1, N, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(0.0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])
    union_area = area_a + area_b - inter_area

    return np.where(union_area > 0.0, inter_area / union_area, 0.0)


def _linear_assignment(
    cost_matrix: np.ndarray,
    cost_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve minimum-cost bipartite matching with threshold gating via SciPy."""
    num_rows, num_cols = cost_matrix.shape
    if num_rows == 0 or num_cols == 0:
        return [], list(range(num_rows)), list(range(num_cols))

    gated = np.where(cost_matrix > cost_threshold, _INVALID_COST, cost_matrix)
    row_indices, col_indices = linear_sum_assignment(gated)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()

    for r_raw, c_raw in zip(row_indices, col_indices, strict=True):
        r, c = int(r_raw), int(c_raw)
        if float(cost_matrix[r, c]) <= cost_threshold:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_rows = [r for r in range(num_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(num_cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class KalmanFilter:
    """8-state constant-velocity Kalman filter for 2D bounding boxes.

    State vector: ``[cx, cy, aspect_ratio, h, vx, vy, va, vh]``.
    """

    def __init__(self) -> None:
        ndim = 4
        dt = 1.0
        self._motion_mat = np.eye(2 * ndim, dtype=np.float64)
        for idx in range(ndim):
            self._motion_mat[idx, ndim + idx] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float64)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Initialize state mean and covariance from an unassociated ``[cx, cy, a, h]`` box."""
        mean_pos = np.asarray(measurement, dtype=np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = max(float(mean_pos[3]), 1.0)
        std = np.array(
            [
                2.0 * self._std_weight_position * h,
                2.0 * self._std_weight_position * h,
                1e-2,
                2.0 * self._std_weight_position * h,
                10.0 * self._std_weight_velocity * h,
                10.0 * self._std_weight_velocity * h,
                1e-5,
                10.0 * self._std_weight_velocity * h,
            ],
            dtype=np.float64,
        )
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance state and covariance one time step under constant velocity."""
        h = max(float(mean[3]), 1.0)
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        next_mean = self._motion_mat @ mean
        next_cov = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return next_mean, next_cov

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution into 4D measurement space ``[cx, cy, a, h]``."""
        h = max(float(mean[3]), 1.0)
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float64,
        )
        innovation_cov = np.diag(np.square(std))
        proj_mean = self._update_mat @ mean
        proj_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return proj_mean, proj_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform Kalman measurement update using Joseph-form covariance stabilization."""
        proj_mean, proj_cov = self.project(mean, covariance)
        cross_cov = covariance @ self._update_mat.T  # (8, 4)
        kalman_gain = np.linalg.solve(proj_cov, cross_cov.T).T  # (8, 4)

        innovation = np.asarray(measurement, dtype=np.float64) - proj_mean
        new_mean = mean + kalman_gain @ innovation

        h = max(float(mean[3]), 1.0)
        r_std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float64,
        )
        r_mat = np.diag(np.square(r_std))
        eye = np.eye(covariance.shape[0], dtype=np.float64)
        i_kh = eye - kalman_gain @ self._update_mat
        new_cov = i_kh @ covariance @ i_kh.T + kalman_gain @ r_mat @ kalman_gain.T
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov


class STrack:
    """Single object track managed by ``ByteTracker``."""

    def __init__(
        self,
        tlbr: np.ndarray,
        score: float,
        class_id: int,
        class_name: str | None = None,
    ) -> None:
        self._tlbr: np.ndarray = np.asarray(tlbr, dtype=np.float64).copy()
        self.score: float = float(score)
        self.class_id: int = int(class_id)
        self.class_name: str | None = class_name

        self.track_id: int = 0
        self.state: TrackState = TrackState.NEW
        self.is_activated: bool = False
        self.frame_id: int = 0
        self.start_frame: int = 0
        self.tracklet_len: int = 0

        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

    @property
    def tlbr(self) -> np.ndarray:
        """Current bounding box in ``[x1, y1, x2, y2]`` format."""
        if self.mean is None:
            return self._tlbr.copy()
        return _xyah_to_xyxy(self.mean[:4])

    def predict(self, kalman_filter: KalmanFilter) -> None:
        """Predict next bounding box state via the Kalman filter."""
        if self.mean is None or self.covariance is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean_state[7] = 0.0
        self.mean, self.covariance = kalman_filter.predict(mean_state, self.covariance)

    def activate(
        self,
        kalman_filter: KalmanFilter,
        frame_id: int,
        track_id: int,
    ) -> None:
        """Start a brand-new track with a fresh Kalman state and assigned track ID."""
        self.track_id = int(track_id)
        self.mean, self.covariance = kalman_filter.initiate(_xyxy_to_xyah(self._tlbr))
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(
        self,
        kalman_filter: KalmanFilter,
        detection: STrack,
        frame_id: int,
    ) -> None:
        """Recover a previously lost track using a newly matched detection."""
        if self.mean is None or self.covariance is None:
            self.mean, self.covariance = kalman_filter.initiate(_xyxy_to_xyah(detection._tlbr))
        else:
            self.mean, self.covariance = kalman_filter.update(
                self.mean,
                self.covariance,
                _xyxy_to_xyah(detection._tlbr),
            )
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = detection.score
        self.class_id = detection.class_id
        self.class_name = detection.class_name

    def update(
        self,
        kalman_filter: KalmanFilter,
        detection: STrack,
        frame_id: int,
    ) -> None:
        """Update an active track with a matched detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        if self.mean is None or self.covariance is None:
            self.mean, self.covariance = kalman_filter.initiate(_xyxy_to_xyah(detection._tlbr))
        else:
            self.mean, self.covariance = kalman_filter.update(
                self.mean,
                self.covariance,
                _xyxy_to_xyah(detection._tlbr),
            )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = detection.score
        self.class_id = detection.class_id
        self.class_name = detection.class_name

    def mark_lost(self) -> None:
        """Transition track into ``LOST`` state."""
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        """Transition track into terminal ``REMOVED`` state."""
        self.state = TrackState.REMOVED


class ByteTracker:
    """Two-stage ByteTrack multi-object tracker with low-confidence occlusion recovery."""

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
    ) -> None:
        if not (0.0 <= low_thresh <= high_thresh <= 1.0):
            raise ValueError(
                f"Expected 0 <= low_thresh <= high_thresh <= 1, "
                f"got low_thresh={low_thresh}, high_thresh={high_thresh}"
            )
        if not (0.0 <= match_thresh <= 1.0):
            raise ValueError(f"match_thresh must lie in [0.0, 1.0], got {match_thresh}")
        if track_buffer < 1:
            raise ValueError(f"track_buffer must be >= 1, got {track_buffer}")

        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.match_thresh = float(match_thresh)
        self.track_buffer = int(track_buffer)

        self._kalman_filter = KalmanFilter()
        self.tracked_tracks: list[STrack] = []
        self.lost_tracks: list[STrack] = []
        self.removed_tracks: list[STrack] = []
        self.frame_id: int = 0
        self._next_id: int = 1

    def _allocate_id(self) -> int:
        track_id = self._next_id
        self._next_id += 1
        return track_id

    @staticmethod
    def _iou_cost(tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)), dtype=np.float64)
        track_boxes = np.stack([t.tlbr for t in tracks], axis=0)
        det_boxes = np.stack([d.tlbr for d in detections], axis=0)
        return 1.0 - _iou_matrix(track_boxes, det_boxes)

    def reset(self) -> None:
        """Clear all active, lost, and removed tracks and reset frame & ID counters."""
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.frame_id = 0
        self._next_id = 1

    def update(self, detections: list[DetectionBox]) -> list[DetectionBox]:
        """Advance the tracker by one frame and return active tracked detections."""
        self.frame_id += 1

        high_tracks: list[STrack] = []
        low_tracks: list[STrack] = []
        for det in detections:
            tlbr = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float64)
            candidate = STrack(
                tlbr=tlbr,
                score=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
            )
            if det.confidence >= self.high_thresh:
                high_tracks.append(candidate)
            elif det.confidence >= self.low_thresh:
                low_tracks.append(candidate)

        track_pool: list[STrack] = list(self.tracked_tracks) + list(self.lost_tracks)
        for track in track_pool:
            track.predict(self._kalman_filter)

        activated_tracks: list[STrack] = []
        refind_tracks: list[STrack] = []
        new_lost_tracks: list[STrack] = []

        # Stage 1: match high-confidence detections against active & lost tracks.
        cost_stage1 = self._iou_cost(track_pool, high_tracks)
        matches_s1, unmatched_pool_idx, unmatched_high_idx = _linear_assignment(
            cost_stage1,
            self.match_thresh,
        )

        for track_idx, det_idx in matches_s1:
            track = track_pool[track_idx]
            det_track = high_tracks[det_idx]
            if track.state == TrackState.TRACKED:
                track.update(self._kalman_filter, det_track, self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(self._kalman_filter, det_track, self.frame_id)
                refind_tracks.append(track)

        r_tracked_tracks = [
            track_pool[idx]
            for idx in unmatched_pool_idx
            if track_pool[idx].state == TrackState.TRACKED
        ]
        still_lost_tracks = [
            track_pool[idx]
            for idx in unmatched_pool_idx
            if track_pool[idx].state == TrackState.LOST
        ]

        # Stage 2: match remaining active tracks against low-confidence detections.
        stage2_thresh = min(self.match_thresh, 0.5)
        cost_stage2 = self._iou_cost(r_tracked_tracks, low_tracks)
        matches_s2, unmatched_r_idx, _ = _linear_assignment(cost_stage2, stage2_thresh)

        for track_idx, det_idx in matches_s2:
            track = r_tracked_tracks[track_idx]
            det_track = low_tracks[det_idx]
            track.update(self._kalman_filter, det_track, self.frame_id)
            activated_tracks.append(track)

        for track_idx in unmatched_r_idx:
            track = r_tracked_tracks[track_idx]
            track.mark_lost()
            new_lost_tracks.append(track)

        # Spawn new tracks from unmatched high-confidence detections.
        for det_idx in unmatched_high_idx:
            det_track = high_tracks[det_idx]
            det_track.activate(self._kalman_filter, self.frame_id, self._allocate_id())
            activated_tracks.append(det_track)

        # Prune lost tracks exceeding track_buffer frames.
        next_lost_tracks: list[STrack] = []
        for track in still_lost_tracks + new_lost_tracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                self.removed_tracks.append(track)
            else:
                next_lost_tracks.append(track)

        self.tracked_tracks = activated_tracks + refind_tracks
        self.lost_tracks = next_lost_tracks

        outputs: list[DetectionBox] = []
        for track in self.tracked_tracks:
            box = track.tlbr
            outputs.append(
                DetectionBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    confidence=float(np.clip(track.score, 0.0, 1.0)),
                    class_id=track.class_id,
                    class_name=track.class_name,
                    track_id=track.track_id,
                )
            )
        return outputs
