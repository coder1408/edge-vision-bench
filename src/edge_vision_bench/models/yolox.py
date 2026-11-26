"""Self-contained Apache-2.0 PyTorch YOLOX implementation compatible with Megvii checkpoints."""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

_MEGVII_RELEASE_BASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"

YOLOX_VARIANTS: dict[str, tuple[float, float]] = {
    "yolox-s": (0.33, 0.50),
    "yolox-m": (0.67, 0.75),
    "yolox-l": (1.00, 1.00),
}

YOLOX_WEIGHT_URLS: dict[str, str] = {
    "yolox-s": f"{_MEGVII_RELEASE_BASE}/yolox_s.pth",
    "yolox-m": f"{_MEGVII_RELEASE_BASE}/yolox_m.pth",
    "yolox-l": f"{_MEGVII_RELEASE_BASE}/yolox_l.pth",
}


def _get_activation(name: str = "silu", inplace: bool = True) -> nn.Module:
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    raise ValueError(f"Unsupported activation function: {name!r}")


class BaseConv(nn.Module):
    """Conv2d + BatchNorm2d + activation block matching Megvii YOLOX parameter layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        # Megvii YOLOX sets eps=1e-3, momentum=0.03 in init_yolo for all BatchNorm2d layers.
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = _get_activation(act, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.act(self.bn(self.conv(x)))
        return out


class Focus(nn.Module):
    """Focus spatial width and height information into channel space."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int = 1,
        stride: int = 1,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        concat = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right),
            dim=1,
        )
        out: torch.Tensor = self.conv(concat)
        return out


class Bottleneck(nn.Module):
    """Standard residual bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, ksize=3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling layer used at the end of CSPDarknet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (5, 9, 13),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, ksize=1, stride=1, act=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        pooled = [x, *(pool(x) for pool in self.m)]
        out: torch.Tensor = self.conv2(torch.cat(pooled, dim=1))
        return out


class CSPLayer(nn.Module):
    """Cross-Stage Partial (C3) block with three 1x1 convolutions and N bottlenecks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, ksize=1, stride=1, act=act)
        self.m = nn.Sequential(
            *(
                Bottleneck(
                    hidden_channels,
                    hidden_channels,
                    shortcut=shortcut,
                    expansion=1.0,
                    act=act,
                )
                for _ in range(n)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.m(self.conv1(x))
        x_2 = self.conv2(x)
        out: torch.Tensor = self.conv3(torch.cat((x_1, x_2), dim=1))
        return out


class CSPDarknet(nn.Module):
    """CSPDarknet backbone producing multi-scale feature maps (dark3, dark4, dark5)."""

    def __init__(
        self,
        dep_mul: float,
        wid_mul: float,
        out_features: tuple[str, ...] = ("dark3", "dark4", "dark5"),
        act: str = "silu",
    ) -> None:
        super().__init__()
        if not out_features:
            raise ValueError("out_features must contain at least one feature stage name")
        self.out_features = out_features
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = Focus(3, base_channels, ksize=3, act=act)

        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, ksize=3, stride=2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, act=act),
        )

        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, ksize=3, stride=2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, act=act),
        )

        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, ksize=3, stride=2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, act=act),
        )

        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, ksize=3, stride=2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                act=act,
            ),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class YOLOPAFPN(nn.Module):
    """CSPDarknet backbone combined with top-down FPN and bottom-up PAN neck."""

    def __init__(
        self,
        depth: float = 1.0,
        width: float = 1.0,
        in_features: tuple[str, ...] = ("dark3", "dark4", "dark5"),
        in_channels: tuple[int, ...] = (256, 512, 1024),
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.backbone = CSPDarknet(depth, width, out_features=in_features, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        neck_depth = max(round(3 * depth), 1)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width),
            int(in_channels[1] * width),
            ksize=1,
            stride=1,
            act=act,
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            n=neck_depth,
            shortcut=False,
            act=act,
        )

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width),
            int(in_channels[0] * width),
            ksize=1,
            stride=1,
            act=act,
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            n=neck_depth,
            shortcut=False,
            act=act,
        )

        self.bu_conv2 = BaseConv(
            int(in_channels[0] * width),
            int(in_channels[0] * width),
            ksize=3,
            stride=2,
            act=act,
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            n=neck_depth,
            shortcut=False,
            act=act,
        )

        self.bu_conv1 = BaseConv(
            int(in_channels[1] * width),
            int(in_channels[1] * width),
            ksize=3,
            stride=2,
            act=act,
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            n=neck_depth,
            shortcut=False,
            act=act,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_features = self.backbone(input_tensor)
        x2, x1, x0 = (out_features[name] for name in self.in_features)

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat((f_out0, x1), dim=1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat((f_out1, x2), dim=1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat((p_out1, fpn_out1), dim=1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat((p_out0, fpn_out0), dim=1)
        pan_out0 = self.C3_n4(p_out0)

        return (pan_out2, pan_out1, pan_out0)


class YOLOXHead(nn.Module):
    """Decoupled anchor-free YOLOX detection head with ONNX-friendly output decoding."""

    def __init__(
        self,
        num_classes: int = 80,
        width: float = 1.0,
        strides: tuple[int, ...] = (8, 16, 32),
        in_channels: tuple[int, ...] = (256, 512, 1024),
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        hidden_channels = int(256 * width)
        for in_ch in in_channels:
            self.stems.append(
                BaseConv(
                    in_channels=int(in_ch * width),
                    out_channels=hidden_channels,
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    BaseConv(hidden_channels, hidden_channels, ksize=3, stride=1, act=act),
                    BaseConv(hidden_channels, hidden_channels, ksize=3, stride=1, act=act),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    BaseConv(hidden_channels, hidden_channels, ksize=3, stride=1, act=act),
                    BaseConv(hidden_channels, hidden_channels, ksize=3, stride=1, act=act),
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        decoded_levels: list[torch.Tensor] = []
        for idx, feat in enumerate(inputs):
            stem_feat = self.stems[idx](feat)
            cls_feat = self.cls_convs[idx](stem_feat)
            cls_output = self.cls_preds[idx](cls_feat)

            reg_feat = self.reg_convs[idx](stem_feat)
            reg_output = self.reg_preds[idx](reg_feat)
            obj_output = self.obj_preds[idx](reg_feat)

            stride = float(self.strides[idx])
            height, width = reg_output.shape[2], reg_output.shape[3]
            yv, xv = torch.meshgrid(
                torch.arange(height, device=reg_output.device, dtype=reg_output.dtype),
                torch.arange(width, device=reg_output.device, dtype=reg_output.dtype),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), dim=2).reshape(1, height * width, 2)

            reg = reg_output.flatten(2).transpose(1, 2)
            obj = torch.sigmoid(obj_output).flatten(2).transpose(1, 2)
            cls = torch.sigmoid(cls_output).flatten(2).transpose(1, 2)

            xy = (reg[..., 0:2] + grid) * stride
            wh = torch.exp(reg[..., 2:4]) * stride
            decoded_levels.append(torch.cat((xy, wh, obj, cls), dim=-1))

        return torch.cat(decoded_levels, dim=1)


class YOLOXModel(nn.Module):
    """Complete anchor-free YOLOX object detector returning decoded ``(B, N, 5 + C)`` tensors."""

    def __init__(
        self,
        depth: float = 0.33,
        width: float = 0.50,
        num_classes: int = 80,
        in_channels: tuple[int, ...] = (256, 512, 1024),
        strides: tuple[int, ...] = (8, 16, 32),
    ) -> None:
        super().__init__()
        self.depth = depth
        self.width = width
        self.num_classes = num_classes
        self.strides = strides
        self.backbone = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels)
        self.head = YOLOXHead(
            num_classes=num_classes,
            width=width,
            strides=strides,
            in_channels=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fpn_outs = self.backbone(x)
        out: torch.Tensor = self.head(fpn_outs)
        return out


def _normalize_variant(variant: str) -> str:
    normalized = variant.strip().lower().replace("_", "-")
    if normalized not in YOLOX_VARIANTS:
        supported = ", ".join(sorted(YOLOX_VARIANTS))
        raise ValueError(f"Unknown YOLOX variant {variant!r}. Supported variants: {supported}")
    return normalized


def _download_weights(url: str, dest: Path, timeout: float = 60.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    logger.info("Downloading YOLOX weights from %s to %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "edge-vision-bench/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return dest


def _load_checkpoint_state_dict(weight_path: Path) -> dict[str, torch.Tensor]:
    try:
        ckpt: Any = torch.load(weight_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError):
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict: dict[str, torch.Tensor] = ckpt["model"]
        return state_dict
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unexpected checkpoint format in {weight_path}: {type(ckpt).__name__}")


def load_yolox(
    variant: str = "yolox-s",
    pretrained: bool = True,
    weights_dir: str | Path = "models/weights",
    num_classes: int = 80,
) -> YOLOXModel:
    """Instantiate a YOLOX variant and optionally load official Megvii COCO weights.

    Args:
        variant: One of ``"yolox-s"``, ``"yolox-m"``, or ``"yolox-l"``.
        pretrained: If True, downloads (if not cached) and loads official Megvii ``.pth`` weights
            with ``strict=True``.
        weights_dir: Local cache directory for downloaded ``.pth`` checkpoints.
        num_classes: Number of detection classes (must be 80 when ``pretrained=True``).

    Returns:
        ``YOLOXModel`` in evaluation mode.
    """
    canonical = _normalize_variant(variant)
    depth, width = YOLOX_VARIANTS[canonical]
    model = YOLOXModel(depth=depth, width=width, num_classes=num_classes)

    if pretrained:
        if num_classes != 80:
            raise ValueError(
                "Official Megvii YOLOX checkpoints require num_classes=80 for strict weight loading"
            )
        url = YOLOX_WEIGHT_URLS[canonical]
        filename = url.rsplit("/", maxsplit=1)[-1]
        weight_path = Path(weights_dir) / filename
        if not weight_path.exists():
            _download_weights(url, weight_path)
        state_dict = _load_checkpoint_state_dict(weight_path)
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model
