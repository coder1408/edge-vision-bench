"""Dataset loaders for COCO val2017, VisDrone2019 DET/MOT, MOTChallenge, and synthetic MOT."""

from __future__ import annotations

import configparser
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from edge_vision_bench.schemas import DetectionBox

# Cross-domain mapping from COCO 80-class contiguous indices to VisDrone2019 category IDs:
#   COCO person(0)     -> VisDrone pedestrian(1)
#   COCO bicycle(1)    -> VisDrone bicycle(3)
#   COCO car(2)        -> VisDrone car(4)
#   COCO motorcycle(3) -> VisDrone motor(10)
#   COCO bus(5)        -> VisDrone bus(9)
#   COCO truck(7)      -> VisDrone truck(6)
COCO_TO_VISDRONE_MAP: dict[int, int] = {
    0: 1,
    1: 3,
    2: 4,
    3: 10,
    5: 9,
    7: 6,
}

VISDRONE_CLASSES: dict[int, str] = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def discover_coco_paths(coco_dir: str | Path) -> tuple[Path, Path]:
    """Locate COCO val2017 images directory and annotations JSON inside a root directory."""
    root = Path(coco_dir)
    image_candidates = [
        root / "val2017",
        root / "images" / "val2017",
        root / "images",
        root,
    ]
    ann_candidates = [
        root / "annotations" / "instances_val2017.json",
        root / "instances_val2017.json",
    ]

    images_dir: Path | None = None
    for candidate in image_candidates:
        if candidate.is_dir() and any(
            p.suffix.lower() in _IMAGE_EXTENSIONS for p in candidate.iterdir() if p.is_file()
        ):
            images_dir = candidate
            break

    ann_json: Path | None = None
    for candidate in ann_candidates:
        if candidate.is_file():
            ann_json = candidate
            break
    if ann_json is None:
        json_files = sorted(root.rglob("*.json"))
        if json_files:
            ann_json = json_files[0]

    if images_dir is None or ann_json is None:
        raise FileNotFoundError(
            f"Could not locate COCO images and annotation JSON under {root}. "
            "Expected e.g. val2017/ and annotations/instances_val2017.json."
        )
    return images_dir, ann_json


class COCOSubsetDataset:
    """COCO val2017 dataset loader with strictly disjoint calibration and eval partitions."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_json: str | Path,
        split: Literal["calibration", "eval", "all"] = "eval",
        calibration_size: int = 500,
        eval_size: int = 500,
        seed: int = 42,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_json = Path(annotations_json)
        self.split = split
        self.calibration_size = calibration_size
        self.eval_size = eval_size
        self.seed = seed

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"COCO images directory not found: {self.images_dir}")
        if not self.annotations_json.is_file():
            raise FileNotFoundError(f"COCO annotation JSON not found: {self.annotations_json}")

        with self.annotations_json.open("r", encoding="utf-8") as handle:
            coco_data = json.load(handle)

        id_to_path: dict[int, Path] = {}
        for img_entry in coco_data.get("images", []):
            img_id = int(img_entry["id"])
            file_name = str(img_entry["file_name"])
            candidate = self.images_dir / file_name
            if not candidate.is_file():
                candidate = self.images_dir / Path(file_name).name
            if candidate.is_file():
                id_to_path[img_id] = candidate

        all_ids = sorted(id_to_path.keys())
        rng = np.random.default_rng(seed)
        permuted_indices = rng.permutation(len(all_ids))
        shuffled_ids = [all_ids[int(i)] for i in permuted_indices]

        calib_end = min(calibration_size, len(shuffled_ids))
        eval_end = min(calib_end + eval_size, len(shuffled_ids))

        self.calibration_image_ids: list[int] = shuffled_ids[:calib_end]
        self.eval_image_ids: list[int] = shuffled_ids[calib_end:eval_end]

        if not set(self.calibration_image_ids).isdisjoint(set(self.eval_image_ids)):
            raise AssertionError(
                "COCO calibration and evaluation splits must be strictly disjoint."
            )

        if split == "calibration":
            selected_ids = self.calibration_image_ids
        elif split == "eval":
            selected_ids = self.eval_image_ids
        elif split == "all":
            selected_ids = shuffled_ids
        else:
            raise ValueError(f"Unsupported COCO split: {split!r}")

        self.image_ids: list[int] = selected_ids
        self.image_paths: list[Path] = [id_to_path[img_id] for img_id in selected_ids]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[int, np.ndarray, Path]:
        img_id = self.image_ids[idx]
        img_path = self.image_paths[idx]
        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to decode image: {img_path}")
        return img_id, frame, img_path

    def load_frames(self, limit: int | None = None) -> tuple[list[int], list[np.ndarray]]:
        """Load up to ``limit`` BGR images along with their COCO image IDs."""
        count = len(self) if limit is None else min(limit, len(self))
        ids: list[int] = []
        frames: list[np.ndarray] = []
        for idx in range(count):
            img_id, frame, _ = self[idx]
            ids.append(img_id)
            frames.append(frame)
        return ids, frames


@dataclass
class MOTSequence:
    """Single Multi-Object Tracking sequence with ordered frame paths and ground-truth boxes."""

    name: str
    frame_paths: list[Path]
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]]
    is_visdrone: bool = False
    fps: float | None = None

    def load_frames(
        self, limit: int | None = None
    ) -> tuple[list[np.ndarray], dict[int, list[tuple[int, float, float, float, float]]]]:
        """Load up to ``limit`` BGR frames and matching 0-based ground-truth annotations."""
        selected_paths = self.frame_paths if limit is None else self.frame_paths[:limit]
        frames: list[np.ndarray] = []
        gt_subset: dict[int, list[tuple[int, float, float, float, float]]] = {}
        for idx, path in enumerate(selected_paths):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to decode MOT frame: {path}")
            frames.append(img)
            if idx in self.gt_by_frame:
                gt_subset[idx] = self.gt_by_frame[idx]
        return frames, gt_subset


def _list_sorted_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _parse_seqinfo(seqinfo_path: Path) -> tuple[str, str, float | None]:
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if "Sequence" not in parser:
        return seqinfo_path.parent.name, "img1", None
    section = parser["Sequence"]
    name = section.get("name", seqinfo_path.parent.name).strip() or seqinfo_path.parent.name
    im_dir = section.get("imDir", "img1").strip() or "img1"
    fps_str = section.get("frameRate", "").strip()
    fps = float(fps_str) if fps_str else None
    return name, im_dir, fps


def _parse_mot_gt_file(
    gt_path: Path,
    *,
    is_visdrone: bool = False,
    allowed_categories: set[int] | None = None,
) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Parse MOTChallenge gt.txt or VisDrone-MOT annotations/<seq>.txt into 0-based frames."""
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
    if not gt_path.is_file():
        return gt_by_frame

    with gt_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            frame_num = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            if w <= 0.0 or h <= 0.0:
                continue

            score = float(parts[6]) if len(parts) >= 7 and parts[6] != "" else 1.0
            category = int(float(parts[7])) if len(parts) >= 8 and parts[7] != "" else 1

            if score == 0.0:
                continue

            if is_visdrone and category in (0, 11):
                continue

            if allowed_categories is not None and category not in allowed_categories:
                continue

            frame_idx = max(0, frame_num - 1)
            gt_by_frame.setdefault(frame_idx, []).append((track_id, x1, y1, x1 + w, y1 + h))

    return gt_by_frame


class MOTSequenceDataset:
    """Unified loader supporting both MOTChallenge and VisDrone2019-MOT directory layouts."""

    def __init__(
        self,
        root_dir: str | Path,
        allowed_categories: set[int] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.allowed_categories = allowed_categories
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"MOT dataset root directory not found: {self.root_dir}")

        self.sequences: list[MOTSequence] = self._discover_sequences()
        if not self.sequences:
            raise FileNotFoundError(
                f"No MOTChallenge or VisDrone-MOT sequences found under {self.root_dir}"
            )

    def _load_visdrone_mot_dir(self, base_dir: Path) -> list[MOTSequence]:
        vd_seq_dir = base_dir / "sequences"
        vd_ann_dir = base_dir / "annotations"
        if not vd_seq_dir.is_dir():
            return []

        found: list[MOTSequence] = []
        for seq_dir in sorted(p for p in vd_seq_dir.iterdir() if p.is_dir()):
            frames = _list_sorted_images(seq_dir)
            if not frames:
                continue
            gt_path = vd_ann_dir / f"{seq_dir.name}.txt"
            gt_by_frame = _parse_mot_gt_file(
                gt_path,
                is_visdrone=True,
                allowed_categories=self.allowed_categories,
            )
            found.append(
                MOTSequence(
                    name=seq_dir.name,
                    frame_paths=frames,
                    gt_by_frame=gt_by_frame,
                    is_visdrone=True,
                )
            )
        return found

    def _discover_sequences(self) -> list[MOTSequence]:
        # Layout 1a: VisDrone2019-MOT at root (`<root>/sequences/<seq>/*.jpg`)
        vd_direct = self._load_visdrone_mot_dir(self.root_dir)
        if vd_direct:
            return vd_direct

        # Layout 1b: Nested VisDrone2019-MOT split (`<root>/VisDrone2019-MOT-val/sequences/...`)
        vd_nested: list[MOTSequence] = []
        for child in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            vd_nested.extend(self._load_visdrone_mot_dir(child))
        if vd_nested:
            return vd_nested

        # Layout 2: Single MOTChallenge sequence (`<root>/img1` or `<root>/seqinfo.ini`)
        single_seq = self._try_load_motchallenge_dir(self.root_dir)
        if single_seq is not None:
            return [single_seq]

        # Layout 3: Multi-sequence MOTChallenge root (`<root>/<seq>/img1`)
        sequences: list[MOTSequence] = []
        for child in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            seq = self._try_load_motchallenge_dir(child)
            if seq is not None:
                sequences.append(seq)

        return sequences

    def _try_load_motchallenge_dir(self, seq_dir: Path) -> MOTSequence | None:
        seqinfo_path = seq_dir / "seqinfo.ini"
        name = seq_dir.name
        im_dir_name = "img1"
        fps: float | None = None

        if seqinfo_path.is_file():
            name, im_dir_name, fps = _parse_seqinfo(seqinfo_path)

        img_dir = seq_dir / im_dir_name
        if not img_dir.is_dir():
            return None

        frames = _list_sorted_images(img_dir)
        if not frames:
            return None

        gt_path = seq_dir / "gt" / "gt.txt"
        gt_by_frame = _parse_mot_gt_file(
            gt_path,
            is_visdrone=False,
            allowed_categories=self.allowed_categories,
        )
        return MOTSequence(
            name=name,
            frame_paths=frames,
            gt_by_frame=gt_by_frame,
            is_visdrone=False,
            fps=fps,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> MOTSequence:
        return self.sequences[idx]

    def get_sequence(self, name: str) -> MOTSequence:
        """Look up a loaded sequence by name."""
        for seq in self.sequences:
            if seq.name == name:
                return seq
        raise KeyError(f"Sequence {name!r} not found in {self.root_dir}")


class VisDroneMOTDataset(MOTSequenceDataset):
    """VisDrone2019-MOT loader with COCO->VisDrone cross-domain category mapping support."""

    def __init__(
        self,
        root_dir: str | Path,
        coco_compatible_only: bool = True,
    ) -> None:
        allowed = set(COCO_TO_VISDRONE_MAP.values()) if coco_compatible_only else None
        super().__init__(root_dir=root_dir, allowed_categories=allowed)

    @staticmethod
    def remap_coco_detections(detections: list[DetectionBox]) -> list[DetectionBox]:
        """Filter and remap COCO 80-class detections to VisDrone2019 category IDs."""
        remapped: list[DetectionBox] = []
        for box in detections:
            vd_class_id = COCO_TO_VISDRONE_MAP.get(box.class_id)
            if vd_class_id is None:
                continue
            remapped.append(
                box.model_copy(
                    update={
                        "class_id": vd_class_id,
                        "class_name": VISDRONE_CLASSES.get(vd_class_id, box.class_name),
                    }
                )
            )
        return remapped


class VisDroneDETDataset:
    """VisDrone2019-DET loader parsing `images/*.jpg` and `annotations/<stem>.txt`."""

    def __init__(
        self,
        root_dir: str | Path,
        coco_compatible_only: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.annotations_dir = self.root_dir / "annotations"
        self.allowed_categories = (
            set(COCO_TO_VISDRONE_MAP.values()) if coco_compatible_only else None
        )

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"VisDrone-DET images directory not found: {self.images_dir}")
        self.image_paths: list[Path] = _list_sorted_images(self.images_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_annotations(self, idx: int) -> list[tuple[int, float, float, float, float]]:
        """Return list of `(visdrone_category_id, x1, y1, x2, y2)` boxes for image `idx`."""
        img_path = self.image_paths[idx]
        ann_path = self.annotations_dir / f"{img_path.stem}.txt"
        boxes: list[tuple[int, float, float, float, float]] = []
        if not ann_path.is_file():
            return boxes

        with ann_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    continue
                x1, y1, w, h = (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
                score = float(parts[4])
                category = int(float(parts[5]))
                if w <= 0.0 or h <= 0.0 or score == 0.0 or category in (0, 11):
                    continue
                if self.allowed_categories is not None and category not in self.allowed_categories:
                    continue
                boxes.append((category, x1, y1, x1 + w, y1 + h))
        return boxes

    def __getitem__(
        self, idx: int
    ) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
        img_path = self.image_paths[idx]
        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to decode VisDrone-DET image: {img_path}")
        return frame, self.load_annotations(idx)


@dataclass
class _SyntheticObject:
    track_id: int
    x: float
    y: float
    w: float
    h: float
    vx: float
    vy: float
    color: tuple[int, int, int]


def generate_synthetic_mot_sequence(
    num_frames: int = 30,
    width: int = 640,
    height: int = 480,
    num_objects: int = 4,
    seed: int = 42,
) -> tuple[list[np.ndarray], dict[int, list[tuple[int, float, float, float, float]]]]:
    """Generate deterministic synthetic BGR frames and MOT ground-truth boxes/track IDs."""
    rng = np.random.default_rng(seed)

    objects: list[_SyntheticObject] = []
    for obj_idx in range(num_objects):
        box_w = float(rng.integers(max(24, width // 16), max(48, width // 8)))
        box_h = float(rng.integers(max(24, height // 12), max(48, height // 6)))
        x = float(rng.uniform(10.0, max(11.0, width - box_w - 10.0)))
        y = float(rng.uniform(10.0, max(11.0, height - box_h - 10.0)))
        vx = float(rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 8.0))
        vy = float(rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 6.0))
        c_arr = rng.integers(60, 240, size=3)
        color = (int(c_arr[0]), int(c_arr[1]), int(c_arr[2]))
        objects.append(
            _SyntheticObject(
                track_id=obj_idx + 1,
                x=x,
                y=y,
                w=box_w,
                h=box_h,
                vx=vx,
                vy=vy,
                color=color,
            )
        )

    # Subtle textured background so preprocessing / perturbations behave realistically.
    base_bg = rng.integers(15, 45, size=(height, width, 3), dtype=np.uint8)

    frames: list[np.ndarray] = []
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}

    for frame_idx in range(num_frames):
        frame = base_bg.copy()
        frame_gt: list[tuple[int, float, float, float, float]] = []

        for obj in objects:
            x1 = float(np.clip(obj.x, 0.0, width - obj.w))
            y1 = float(np.clip(obj.y, 0.0, height - obj.h))
            x2 = float(np.clip(x1 + obj.w, x1 + 1.0, float(width)))
            y2 = float(np.clip(y1 + obj.h, y1 + 1.0, float(height)))

            cv2.rectangle(
                frame,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                obj.color,
                thickness=-1,
            )
            frame_gt.append((obj.track_id, x1, y1, x2, y2))

            next_x = x1 + obj.vx
            next_y = y1 + obj.vy
            if next_x < 0.0 or next_x + obj.w > width:
                obj.vx = -obj.vx
                next_x = float(np.clip(next_x, 0.0, width - obj.w))
            if next_y < 0.0 or next_y + obj.h > height:
                obj.vy = -obj.vy
                next_y = float(np.clip(next_y, 0.0, height - obj.h))

            obj.x = next_x
            obj.y = next_y

        frames.append(frame)
        gt_by_frame[frame_idx] = frame_gt

    return frames, gt_by_frame
