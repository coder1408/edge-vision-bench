"""Unit tests for preprocessing, YOLOX postprocessing, ByteTrack, homography, and OOD transforms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from edge_vision_bench.bench.perturbations import (
    CutoutOcclusionTransform,
    LowLightGammaTransform,
    MotionBlurTransform,
    OODPerturbation,
    get_default_ood_suite,
)
from edge_vision_bench.pipeline.homography import PlanarHomography
from edge_vision_bench.pipeline.postproc import (
    COCO_CLASSES,
    batched_nms,
    decode_yolox_outputs,
    postprocess_detections,
)
from edge_vision_bench.pipeline.preproc import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LetterboxResult,
    dinov2_preprocess,
    letterbox_preprocess,
    preprocess_batch,
)
from edge_vision_bench.pipeline.tracker import ByteTracker, KalmanFilter, TrackState
from edge_vision_bench.schemas import DetectionBox


class TestPreprocessing:
    """Tests for YOLOX letterbox, DINOv2 preprocessing, and batch helpers."""

    @pytest.mark.parametrize(
        ("orig_h", "orig_w", "input_size"),
        [
            (720, 1280, (640, 640)),
            (1080, 600, (640, 640)),
            (480, 854, (320, 640)),
        ],
    )
    def test_letterbox_roundtrip_geometry(
        self,
        orig_h: int,
        orig_w: int,
        input_size: tuple[int, int],
    ) -> None:
        img = np.full((orig_h, orig_w, 3), 200, dtype=np.uint8)
        lb = letterbox_preprocess(img, input_size=input_size, pad_value=114)

        assert isinstance(lb, LetterboxResult)
        assert lb.tensor_nchw.shape == (1, 3, input_size[0], input_size[1])
        assert lb.tensor_nchw.dtype == np.float32
        assert lb.tensor_nchw.flags["C_CONTIGUOUS"]
        assert lb.orig_h == orig_h
        assert lb.orig_w == orig_w

        # Verify border padding uses exact pad_value in [0, 255] BGR range.
        if lb.pad_y > 0:
            assert np.all(lb.tensor_nchw[0, :, 0, :] == 114.0)
        if lb.pad_x > 0:
            assert np.all(lb.tensor_nchw[0, :, :, 0] == 114.0)

        # Map an original-space bounding box into letterbox coords and invert via postproc.
        orig_box = (orig_w * 0.2, orig_h * 0.25, orig_w * 0.7, orig_h * 0.8)
        lb_x1 = orig_box[0] * lb.scale + lb.pad_x
        lb_y1 = orig_box[1] * lb.scale + lb.pad_y
        lb_x2 = orig_box[2] * lb.scale + lb.pad_x
        lb_y2 = orig_box[3] * lb.scale + lb.pad_y

        cx = (lb_x1 + lb_x2) * 0.5
        cy = (lb_y1 + lb_y2) * 0.5
        bw = lb_x2 - lb_x1
        bh = lb_y2 - lb_y1

        raw = np.zeros((1, 1, 85), dtype=np.float32)
        raw[0, 0, :4] = [cx, cy, bw, bh]
        raw[0, 0, 4] = 0.9
        raw[0, 0, 5 + 2] = 0.95  # class 2 = car

        dets = postprocess_detections(
            raw_output=raw,
            letterbox=lb,
            conf_threshold=0.25,
            nms_threshold=0.45,
            input_size=input_size,
            already_decoded=True,
        )
        assert len(dets) == 1
        det = dets[0]
        assert det.class_id == 2
        assert det.class_name == "car"
        assert det.x1 == pytest.approx(orig_box[0], abs=1e-4)
        assert det.y1 == pytest.approx(orig_box[1], abs=1e-4)
        assert det.x2 == pytest.approx(orig_box[2], abs=1e-4)
        assert det.y2 == pytest.approx(orig_box[3], abs=1e-4)

    def test_letterbox_bgr_no_swap_vs_imagenet_norm(self) -> None:
        # Constant BGR pixel: B=10, G=100, R=200.
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 10
        img[:, :, 1] = 100
        img[:, :, 2] = 200

        lb_raw = letterbox_preprocess(img, input_size=(100, 100), normalize_imagenet=False)
        assert lb_raw.tensor_nchw[0, 0, 50, 50] == pytest.approx(10.0)
        assert lb_raw.tensor_nchw[0, 1, 50, 50] == pytest.approx(100.0)
        assert lb_raw.tensor_nchw[0, 2, 50, 50] == pytest.approx(200.0)

        lb_norm = letterbox_preprocess(img, input_size=(100, 100), normalize_imagenet=True)
        rgb_scaled = np.array([200.0, 100.0, 10.0], dtype=np.float32) / 255.0
        expected_rgb = (rgb_scaled - IMAGENET_MEAN) / IMAGENET_STD
        np.testing.assert_allclose(lb_norm.tensor_nchw[0, :, 50, 50], expected_rgb, atol=1e-5)

    def test_dinov2_preprocess_normalization_stats(self) -> None:
        # Constant BGR image: B=51, G=153, R=204 -> RGB=[0.8, 0.6, 0.2]
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:, :, 0] = 51
        img[:, :, 1] = 153
        img[:, :, 2] = 204

        tensor = dinov2_preprocess(img, input_size=(518, 518))
        assert tensor.shape == (1, 3, 518, 518)
        assert tensor.dtype == np.float32
        assert tensor.flags["C_CONTIGUOUS"]

        rgb_scaled = np.array([204.0, 153.0, 51.0], dtype=np.float32) / 255.0
        expected_rgb = (rgb_scaled - IMAGENET_MEAN) / IMAGENET_STD
        channel_means = tensor[0].mean(axis=(1, 2))
        channel_stds = tensor[0].std(axis=(1, 2))
        np.testing.assert_allclose(channel_means, expected_rgb, atol=1e-4)
        np.testing.assert_allclose(channel_stds, np.zeros(3, dtype=np.float32), atol=1e-5)

    def test_preprocess_batch_both_tracks(self) -> None:
        imgs = [
            np.full((120, 160, 3), 50, dtype=np.uint8),
            np.full((240, 180, 3), 150, dtype=np.uint8),
        ]
        det_batch, metas = preprocess_batch(imgs, track="detection", input_size=(320, 320))
        assert det_batch.shape == (2, 3, 320, 320)
        assert det_batch.dtype == np.float32
        assert det_batch.flags["C_CONTIGUOUS"]
        assert len(metas) == 2

        emb_batch, emb_metas = preprocess_batch(imgs, track="embedding", input_size=(224, 224))
        assert emb_batch.shape == (2, 3, 224, 224)
        assert emb_batch.dtype == np.float32
        assert emb_metas == []

        with pytest.raises(ValueError, match="empty"):
            preprocess_batch([], track="detection", input_size=(320, 320))

        with pytest.raises(ValueError, match="Unknown track"):
            preprocess_batch(imgs, track="segmentation", input_size=(320, 320))


class TestPostprocessing:
    """Tests for YOLOX decoding, class-aware NMS, and detection box mapping."""

    def test_coco_classes_standard_length_and_indices(self) -> None:
        assert len(COCO_CLASSES) == 80
        assert COCO_CLASSES[0] == "person"
        assert COCO_CLASSES[2] == "car"
        assert COCO_CLASSES[7] == "truck"
        assert COCO_CLASSES[79] == "toothbrush"

    def test_decode_yolox_raw_and_already_decoded(self) -> None:
        input_size = (640, 640)
        strides = (8, 16, 32)
        total_anchors = (80 * 80) + (40 * 40) + (20 * 20)
        assert total_anchors == 8400

        raw = np.full((1, total_anchors, 85), -10.0, dtype=np.float32)
        raw[..., :4] = 0.0

        # Pick anchor at stride=8, gy=3, gx=5 -> index = 3 * 80 + 5 = 245.
        idx_s8 = 3 * 80 + 5
        raw[0, idx_s8, 0] = 0.5  # cx = (0.5 + 5) * 8 = 44.0
        raw[0, idx_s8, 1] = 0.25  # cy = (0.25 + 3) * 8 = 26.0
        raw[0, idx_s8, 2] = math.log(4.0)  # w = 4 * 8 = 32.0
        raw[0, idx_s8, 3] = math.log(6.0)  # h = 6 * 8 = 48.0
        raw[0, idx_s8, 4] = 4.0  # sigmoid(4.0) ~ 0.9820
        raw[0, idx_s8, 5 + 0] = 3.0  # person, sigmoid(3.0) ~ 0.9526

        # Pick anchor at stride=32, gy=10, gx=7 -> index = 6400 + 1600 + 10 * 20 + 7 = 8207.
        idx_s32 = 6400 + 1600 + 10 * 20 + 7
        raw[0, idx_s32, 0] = 0.0  # cx = 7 * 32 = 224.0
        raw[0, idx_s32, 1] = 0.5  # cy = 10.5 * 32 = 336.0
        raw[0, idx_s32, 2] = math.log(3.0)  # w = 96.0
        raw[0, idx_s32, 3] = math.log(2.0)  # h = 64.0
        raw[0, idx_s32, 4] = 5.0
        raw[0, idx_s32, 5 + 5] = 4.0  # bus

        raw_copy = raw.copy()
        decoded = decode_yolox_outputs(
            raw,
            input_size=input_size,
            strides=strides,
            already_decoded=False,
        )
        # Input must not be mutated.
        np.testing.assert_array_equal(raw, raw_copy)

        assert decoded.shape == (1, 8400, 85)
        np.testing.assert_allclose(decoded[0, idx_s8, :4], [44.0, 26.0, 32.0, 48.0], atol=1e-4)
        np.testing.assert_allclose(decoded[0, idx_s32, :4], [224.0, 336.0, 96.0, 64.0], atol=1e-4)
        assert decoded[0, idx_s8, 4] == pytest.approx(1.0 / (1.0 + math.exp(-4.0)), abs=1e-5)
        assert decoded[0, idx_s8, 5] == pytest.approx(1.0 / (1.0 + math.exp(-3.0)), abs=1e-5)

        # Already-decoded 2D input is promoted cleanly to 3D.
        decoded_2d = decode_yolox_outputs(decoded[0], already_decoded=True)
        assert decoded_2d.shape == (1, 8400, 85)
        np.testing.assert_allclose(decoded_2d, decoded)

    def test_class_aware_nms_suppresses_same_class_and_keeps_different_classes(self) -> None:
        # 3 overlapping detections + 1 low-confidence detection across batch size 2.
        decoded = np.zeros((2, 4, 85), dtype=np.float32)

        # Batch 0, det 0: person, high score at [100, 100, 50, 50]
        decoded[0, 0, :4] = [100.0, 100.0, 50.0, 50.0]
        decoded[0, 0, 4] = 0.9
        decoded[0, 0, 5 + 0] = 0.9  # score 0.81

        # Batch 0, det 1: person, heavily overlapping duplicate with lower score -> suppressed!
        decoded[0, 1, :4] = [102.0, 101.0, 50.0, 50.0]
        decoded[0, 1, 4] = 0.8
        decoded[0, 1, 5 + 0] = 0.8  # score 0.64

        # Batch 0, det 2: car, IDENTICAL box to det 0 -> different class, MUST be kept!
        decoded[0, 2, :4] = [100.0, 100.0, 50.0, 50.0]
        decoded[0, 2, 4] = 0.85
        decoded[0, 2, 5 + 2] = 0.85  # score 0.7225

        # Batch 0, det 3: below confidence threshold -> filtered out.
        decoded[0, 3, :4] = [300.0, 300.0, 40.0, 40.0]
        decoded[0, 3, 4] = 0.3
        decoded[0, 3, 5 + 1] = 0.3  # score 0.09

        # Batch 1: no boxes above threshold -> empty (0, 6).
        nms_out = batched_nms(decoded, conf_threshold=0.25, nms_threshold=0.45)
        assert len(nms_out) == 2
        assert nms_out[0].shape == (2, 6)
        assert nms_out[0].dtype == np.float32
        assert nms_out[1].shape == (0, 6)

        kept_classes = {int(round(float(row[5]))) for row in nms_out[0]}
        assert kept_classes == {0, 2}
        # Highest confidence box comes first.
        assert nms_out[0][0, 4] == pytest.approx(0.81, abs=1e-5)

    def test_postprocess_clamps_bounds_and_handles_unknown_class_id(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        lb = letterbox_preprocess(img, input_size=(640, 640))

        # 82 classes so class_id=81 is outside COCO_CLASSES.
        raw = np.zeros((1, 1, 5 + 82), dtype=np.float32)
        raw[0, 0, :4] = [5.0, 90.0, 40.0, 40.0]  # extends past left edge x1 < 0
        raw[0, 0, 4] = 1.2  # intentionally > 1.0 to test clamping
        raw[0, 0, 5 + 81] = 1.0

        dets = postprocess_detections(raw, lb, conf_threshold=0.25, nms_threshold=0.45)
        assert len(dets) == 1
        assert dets[0].x1 == 0.0
        assert 0.0 <= dets[0].y1 < dets[0].y2 <= 480.0
        assert dets[0].confidence == 1.0
        assert dets[0].class_id == 81
        assert dets[0].class_name is None


class TestByteTracker:
    """Tests for clean-room ByteTrack Kalman filter, occlusion recovery, and reset."""

    def test_kalman_filter_constant_velocity_prediction(self) -> None:
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 100.0]))
        for step in range(1, 6):
            mean, cov = kf.predict(mean, cov)
            meas = np.array([100.0 + 10.0 * step, 200.0, 0.5, 100.0])
            mean, cov = kf.update(mean, cov, meas)
        # Velocity along x should converge close to +10 px/frame.
        assert mean[4] > 7.0
        pred_mean, _ = kf.predict(mean, cov)
        assert pred_mean[0] > 155.0

    def test_bytetrack_id_persistence_across_three_frame_low_conf_occlusion(self) -> None:
        tracker = ByteTracker(
            high_thresh=0.5,
            low_thresh=0.1,
            match_thresh=0.8,
            track_buffer=30,
        )

        # Frame schedule: 3 high-conf frames, 3 low-conf occlusion dip frames, 3 recovery frames.
        confidences = [0.90, 0.88, 0.85, 0.25, 0.18, 0.15, 0.82, 0.87, 0.91]
        observed_ids_obj1: list[int] = []
        observed_ids_obj2: list[int] = []

        for frame_idx, conf in enumerate(confidences):
            dx = frame_idx * 6.0
            det_a = DetectionBox(
                x1=100.0 + dx,
                y1=120.0,
                x2=160.0 + dx,
                y2=240.0,
                confidence=conf,
                class_id=0,
                class_name="person",
            )
            # Second object moves independently at constant high confidence.
            det_b = DetectionBox(
                x1=420.0 - dx * 0.5,
                y1=200.0,
                x2=490.0 - dx * 0.5,
                y2=310.0,
                confidence=0.85,
                class_id=2,
                class_name="car",
            )
            tracked = tracker.update([det_a, det_b])
            assert len(tracked) == 2

            by_class = {d.class_id: d for d in tracked}
            assert by_class[0].track_id is not None
            assert by_class[2].track_id is not None
            assert by_class[0].class_name == "person"
            assert by_class[2].class_name == "car"
            assert by_class[0].confidence == pytest.approx(conf, abs=1e-6)

            observed_ids_obj1.append(by_class[0].track_id)
            observed_ids_obj2.append(by_class[2].track_id)

        # Zero ID switches across the entire 9-frame sequence including the 3-frame low-conf dip!
        assert len(set(observed_ids_obj1)) == 1
        assert len(set(observed_ids_obj2)) == 1
        assert observed_ids_obj1[0] != observed_ids_obj2[0]

    def test_bytetrack_lost_recovery_expiration_and_reset(self) -> None:
        tracker = ByteTracker(high_thresh=0.5, low_thresh=0.1, match_thresh=0.8, track_buffer=2)

        box = DetectionBox(x1=50.0, y1=50.0, x2=110.0, y2=150.0, confidence=0.9, class_id=0)
        out1 = tracker.update([box])
        assert len(out1) == 1
        initial_id = out1[0].track_id
        assert initial_id == 1

        # Drop detection for 2 frames (within track_buffer=2) -> track becomes LOST, then recovers.
        assert tracker.update([]) == []
        assert len(tracker.lost_tracks) == 1
        assert tracker.lost_tracks[0].state == TrackState.LOST
        assert tracker.update([]) == []
        recovered = tracker.update([box])
        assert len(recovered) == 1
        assert recovered[0].track_id == initial_id

        # Drop detection for 3 frames (> track_buffer=2) -> track is REMOVED.
        for _ in range(3):
            tracker.update([])
        assert len(tracker.lost_tracks) == 0
        assert len(tracker.removed_tracks) == 1
        assert tracker.removed_tracks[0].state == TrackState.REMOVED

        respawned = tracker.update([box])
        assert len(respawned) == 1
        assert respawned[0].track_id == 2

        # Full reset restarts ID counter and clears all pools.
        tracker.reset()
        assert tracker.frame_id == 0
        assert tracker.tracked_tracks == []
        assert tracker.lost_tracks == []
        assert tracker.removed_tracks == []

        fresh = tracker.update([box])
        assert len(fresh) == 1
        assert fresh[0].track_id == 1


class TestPlanarHomography:
    """Tests for projective planar homography estimation and detection projection."""

    def test_from_correspondences_exact_reprojection_error(self) -> None:
        # Known non-trivial perspective homography mapping pixel plane to world meters.
        true_h = np.array(
            [
                [0.08, -0.015, -12.0],
                [0.005, 0.11, -8.5],
                [0.0002, 0.00035, 1.0],
            ],
            dtype=np.float64,
        )
        image_pts = np.array(
            [
                [100.0, 120.0],
                [540.0, 140.0],
                [600.0, 440.0],
                [80.0, 420.0],
            ],
            dtype=np.float64,
        )

        def _apply_h(pts: np.ndarray) -> np.ndarray:
            ones = np.ones((pts.shape[0], 1), dtype=np.float64)
            homog = (true_h @ np.hstack([pts, ones]).T).T
            return homog[:, :2] / homog[:, 2:3]

        world_pts = _apply_h(image_pts)
        homography = PlanarHomography.from_correspondences(image_pts, world_pts)

        # Verify exact reprojection error < 1e-4 on correspondences AND interior test points.
        test_pts = np.vstack([image_pts, [[320.0, 260.0], [210.0, 350.0], [450.0, 220.0]]])
        expected_world = _apply_h(test_pts)

        for img_pt, exp_pt in zip(test_pts, expected_world, strict=True):
            proj_x, proj_y = homography.project_point(float(img_pt[0]), float(img_pt[1]))
            err = math.hypot(proj_x - float(exp_pt[0]), proj_y - float(exp_pt[1]))
            assert err < 1e-4

    def test_project_detections_uses_footpoint_and_does_not_mutate(self) -> None:
        # Pure scale + translation homography: X = 0.1 * x, Y = 0.2 * y.
        homography = PlanarHomography([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]])
        det = DetectionBox(
            x1=100.0,
            y1=50.0,
            x2=200.0,
            y2=250.0,
            confidence=0.9,
            class_id=0,
            class_name="person",
            track_id=7,
        )
        projected = homography.project_detections([det])
        assert len(projected) == 1
        assert det.world_xy is None  # input untouched
        assert projected[0].world_xy is not None
        # Bottom-center footpoint = (150.0, 250.0) -> (15.0, 50.0).
        assert projected[0].world_xy[0] == pytest.approx(15.0, abs=1e-6)
        assert projected[0].world_xy[1] == pytest.approx(50.0, abs=1e-6)
        assert projected[0].track_id == 7

    def test_homography_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            PlanarHomography([[1.0, 0.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="singular"):
            PlanarHomography([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        with pytest.raises(ValueError, match="At least 4"):
            PlanarHomography.from_correspondences(
                np.zeros((3, 2), dtype=np.float64),
                np.zeros((3, 2), dtype=np.float64),
            )

        # Matrix where w = x - 10 vanishes at x=10.
        vanishing = PlanarHomography([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -10.0]])
        with pytest.raises(ValueError, match="near zero"):
            vanishing.project_point(10.0, 5.0)


class TestOODPerturbations:
    """Tests for deterministic seeded OOD corruptions."""

    def test_default_suite_keys_and_contract(self) -> None:
        suite = get_default_ood_suite()
        assert set(suite.keys()) == {"blur", "occlusion", "lowlight"}
        assert all(isinstance(transform, OODPerturbation) for transform in suite.values())

        rng = np.random.default_rng(123)
        img = rng.integers(30, 225, size=(120, 160, 3), dtype=np.uint8)
        original = img.copy()

        for key, transform in suite.items():
            out_a = transform.apply(img, seed=42)
            out_b = transform.apply(img, seed=42)

            # Input must never be mutated.
            np.testing.assert_array_equal(img, original)
            # Shape & dtype preserved.
            assert out_a.shape == img.shape, key
            assert out_a.dtype == np.uint8, key
            # Seeded determinism.
            np.testing.assert_array_equal(out_a, out_b)

    def test_motion_blur_reduces_edge_gradient_and_supports_angle(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255
        blur_h = MotionBlurTransform(kernel_size=15, angle_deg=0.0)
        blur_diag = MotionBlurTransform(kernel_size=15, angle_deg=45.0)

        out_h = blur_h.apply(img)
        out_diag = blur_diag.apply(img)

        orig_max_grad = float(np.max(np.abs(np.diff(img.astype(np.float32), axis=1))))
        blur_max_grad = float(np.max(np.abs(np.diff(out_h.astype(np.float32), axis=1))))
        assert blur_max_grad < orig_max_grad * 0.25
        assert not np.array_equal(out_h, out_diag)

    def test_cutout_occlusion_seed_sensitivity_and_area(self) -> None:
        img = np.full((200, 200, 3), 180, dtype=np.uint8)
        cutout = CutoutOcclusionTransform(occlusion_ratio=0.30, num_patches=4)

        out_s1 = cutout.apply(img, seed=7)
        out_s2 = cutout.apply(img, seed=99)

        assert not np.array_equal(out_s1, out_s2)
        masked_fraction = float(np.mean(np.all(out_s1 == 0, axis=2)))
        assert 0.10 <= masked_fraction <= 0.35

    def test_lowlight_gamma_darkens_and_is_seed_sensitive(self) -> None:
        img = np.full((100, 100, 3), 150, dtype=np.uint8)
        lowlight = LowLightGammaTransform(gamma=2.4, noise_std=4.0)

        out_s1 = lowlight.apply(img, seed=11)
        out_s2 = lowlight.apply(img, seed=22)

        assert not np.array_equal(out_s1, out_s2)
        # Expected noiseless value: (150/255)^2.4 * 255 ~ 71.4
        expected_mean = ((150.0 / 255.0) ** 2.4) * 255.0
        assert float(out_s1.mean()) == pytest.approx(expected_mean, abs=2.0)
        assert float(out_s1.mean()) < float(img.mean()) * 0.6
