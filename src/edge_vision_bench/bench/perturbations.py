"""Deterministic out-of-distribution (OOD) visual perturbations for robustness evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


def _validate_uint8_bgr(image_bgr: np.ndarray) -> None:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image of shape (H, W, 3), got {image_bgr.shape!r}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image array, got dtype {image_bgr.dtype!r}")
    if image_bgr.shape[0] <= 0 or image_bgr.shape[1] <= 0:
        raise ValueError(f"Image spatial dimensions must be positive, got {image_bgr.shape[:2]!r}")


class OODPerturbation(ABC):
    """Abstract base class for deterministic, non-mutating BGR image corruptions."""

    name: str

    @abstractmethod
    def apply(self, image_bgr: np.ndarray, seed: int = 42) -> np.ndarray:
        """Apply perturbation without mutating ``image_bgr`` and return uint8 ``(H, W, 3)``."""


class MotionBlurTransform(OODPerturbation):
    """Directional linear motion blur via a rotated point-spread function (PSF) kernel."""

    def __init__(self, kernel_size: int = 15, angle_deg: float = 0.0) -> None:
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        self.kernel_size = int(kernel_size)
        self.angle_deg = float(angle_deg)
        self.name = "motion_blur"

    def _build_kernel(self) -> np.ndarray:
        k = self.kernel_size
        kernel: np.ndarray = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0

        if self.angle_deg % 360.0 != 0.0 and k > 1:
            center = ((k - 1) * 0.5, (k - 1) * 0.5)
            rot_mat = cv2.getRotationMatrix2D(center, self.angle_deg, 1.0)
            kernel = np.asarray(
                cv2.warpAffine(kernel, rot_mat, (k, k), flags=cv2.INTER_LINEAR),
                dtype=np.float32,
            )

        total = float(kernel.sum())
        if total > 0.0:
            kernel /= total
        else:
            kernel[k // 2, k // 2] = 1.0
        return kernel

    def apply(self, image_bgr: np.ndarray, seed: int = 42) -> np.ndarray:
        """Apply directional motion blur deterministically."""
        del seed  # Deterministic PSF independent of RNG seed.
        _validate_uint8_bgr(image_bgr)
        if self.kernel_size == 1:
            return image_bgr.copy()
        kernel = self._build_kernel()
        blurred = cv2.filter2D(
            image_bgr,
            ddepth=-1,
            kernel=kernel,
            borderType=cv2.BORDER_REPLICATE,
        )
        return np.ascontiguousarray(blurred, dtype=np.uint8)


class CutoutOcclusionTransform(OODPerturbation):
    """Deterministic seeded rectangular cutout patches masking a target image area ratio."""

    def __init__(
        self,
        occlusion_ratio: float = 0.30,
        num_patches: int = 4,
        fill_value: int = 0,
    ) -> None:
        if not (0.0 <= occlusion_ratio <= 1.0):
            raise ValueError(f"occlusion_ratio must lie in [0.0, 1.0], got {occlusion_ratio}")
        if num_patches < 1:
            raise ValueError(f"num_patches must be >= 1, got {num_patches}")
        if not (0 <= fill_value <= 255):
            raise ValueError(f"fill_value must lie in [0, 255], got {fill_value}")
        self.occlusion_ratio = float(occlusion_ratio)
        self.num_patches = int(num_patches)
        self.fill_value = int(fill_value)
        self.name = "cutout_occlusion"

    def apply(self, image_bgr: np.ndarray, seed: int = 42) -> np.ndarray:
        """Mask rectangular patches using a seeded NumPy generator."""
        _validate_uint8_bgr(image_bgr)
        out = image_bgr.copy()
        if self.occlusion_ratio == 0.0:
            return out

        h, w = int(out.shape[0]), int(out.shape[1])
        rng = np.random.default_rng(seed)
        area_per_patch = (h * w * self.occlusion_ratio) / self.num_patches

        for _ in range(self.num_patches):
            aspect = float(rng.uniform(0.5, 2.0))
            patch_h = max(1, min(h, int(round(np.sqrt(area_per_patch / aspect)))))
            patch_w = max(1, min(w, int(round(np.sqrt(area_per_patch * aspect)))))
            y0 = int(rng.integers(0, max(1, h - patch_h + 1)))
            x0 = int(rng.integers(0, max(1, w - patch_w + 1)))
            out[y0 : y0 + patch_h, x0 : x0 + patch_w] = self.fill_value

        return out


class LowLightGammaTransform(OODPerturbation):
    """Low-light gamma attenuation combined with seeded Gaussian sensor read noise."""

    def __init__(self, gamma: float = 2.4, noise_std: float = 4.0) -> None:
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")
        self.gamma = float(gamma)
        self.noise_std = float(noise_std)
        self.name = "lowlight_gamma"

    def apply(self, image_bgr: np.ndarray, seed: int = 42) -> np.ndarray:
        """Apply gamma darkening and seeded zero-mean Gaussian read noise."""
        _validate_uint8_bgr(image_bgr)
        rng = np.random.default_rng(seed)
        normalized = image_bgr.astype(np.float32) / 255.0
        attenuated = np.power(normalized, self.gamma) * 255.0

        if self.noise_std > 0.0:
            noise = rng.normal(loc=0.0, scale=self.noise_std, size=image_bgr.shape).astype(
                np.float32
            )
            attenuated = attenuated + noise

        return np.clip(np.rint(attenuated), 0.0, 255.0).astype(np.uint8)


def get_default_ood_suite() -> dict[str, OODPerturbation]:
    """Return the default 3-corruption OOD benchmark suite."""
    return {
        "blur": MotionBlurTransform(kernel_size=15, angle_deg=0.0),
        "occlusion": CutoutOcclusionTransform(occlusion_ratio=0.30, num_patches=4),
        "lowlight": LowLightGammaTransform(gamma=2.4, noise_std=4.0),
    }
