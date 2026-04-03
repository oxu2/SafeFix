import random
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

COCO_PERSON_KEYPOINT_FLIP_INDICES = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _has_valid_person_annotation(annotations: List[Dict]) -> bool:
    for ann in annotations:
        if ann.get("category_id") != 1:
            continue
        if ann.get("iscrowd", 0) == 1:
            continue

        bbox = ann.get("bbox", [0, 0, 0, 0])
        if bbox[2] <= 1 or bbox[3] <= 1:
            continue

        keypoints = ann.get("keypoints", [])
        if len(keypoints) != 51:
            continue
        visible = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
        if visible > 0:
            return True
    return False


class CocoPersonKeypointsDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        train: bool = False,
        hflip_prob: float = 0.5,
        filter_empty: bool = True,
    ) -> None:
        self.image_dir = image_dir
        self.train = train
        self.hflip_prob = hflip_prob
        self.coco = COCO(annotation_file)
        self.image_ids = sorted(self.coco.getImgIds())

        if filter_empty:
            valid_ids: List[int] = []
            for image_id in self.image_ids:
                ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=[1], iscrowd=None)
                anns = self.coco.loadAnns(ann_ids)
                if _has_valid_person_annotation(anns):
                    valid_ids.append(image_id)
            self.image_ids = valid_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.image_dir}/{image_info['file_name']}"
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=[1], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []
        keypoints: List[List[List[float]]] = []

        for ann in anns:
            if ann.get("category_id") != 1:
                continue
            if ann.get("iscrowd", 0) == 1:
                continue

            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue

            kp = ann.get("keypoints", [])
            if len(kp) != 51:
                continue
            if sum(1 for i in range(2, len(kp), 3) if kp[i] > 0) <= 0:
                continue

            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = max(x1 + 1.0, float(x + w))
            y2 = max(y1 + 1.0, float(y + h))
            boxes.append([x1, y1, x2, y2])
            labels.append(1)
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))
            keypoints.append([kp[i : i + 3] for i in range(0, len(kp), 3)])

        if not boxes:
            # Dataset is filtered, but this keeps sampling robust when annotations are malformed.
            fallback_idx = (idx + 1) % len(self.image_ids)
            return self.__getitem__(fallback_idx)

        target: Dict[str, torch.Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            "keypoints": torch.tensor(keypoints, dtype=torch.float32),
        }

        image_tensor = F.pil_to_tensor(image).float() / 255.0
        if self.train and random.random() < self.hflip_prob:
            image_tensor, target = horizontal_flip(image_tensor, target)

        return image_tensor, target


def horizontal_flip(
    image: torch.Tensor, target: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    flipped_image = F.hflip(image)
    width = image.shape[-1]

    boxes = target["boxes"].clone()
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    boxes[:, 0] = width - x2
    boxes[:, 2] = width - x1
    target["boxes"] = boxes

    keypoints = target["keypoints"].clone()
    keypoints[..., 0] = width - keypoints[..., 0]
    keypoints = keypoints[:, COCO_PERSON_KEYPOINT_FLIP_INDICES, :]
    invisible = keypoints[..., 2] == 0
    keypoints[..., 0][invisible] = 0
    keypoints[..., 1][invisible] = 0
    target["keypoints"] = keypoints

    return flipped_image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def default_coco_paths(data_root: str) -> Dict[str, str]:
    return {
        "train_images": f"{data_root}/train2017",
        "val_images": f"{data_root}/val2017",
        "train_annotations": f"{data_root}/annotations/person_keypoints_train2017.json",
        "val_annotations": f"{data_root}/annotations/person_keypoints_val2017.json",
    }


def build_datasets(
    data_root: str,
    train_images: Optional[str] = None,
    val_images: Optional[str] = None,
    train_annotations: Optional[str] = None,
    val_annotations: Optional[str] = None,
    hflip_prob: float = 0.5,
) -> Tuple[CocoPersonKeypointsDataset, CocoPersonKeypointsDataset]:
    paths = default_coco_paths(data_root)
    train_dataset = CocoPersonKeypointsDataset(
        image_dir=train_images or paths["train_images"],
        annotation_file=train_annotations or paths["train_annotations"],
        train=True,
        hflip_prob=hflip_prob,
        filter_empty=True,
    )
    val_dataset = CocoPersonKeypointsDataset(
        image_dir=val_images or paths["val_images"],
        annotation_file=val_annotations or paths["val_annotations"],
        train=False,
        hflip_prob=0.0,
        filter_empty=True,
    )
    return train_dataset, val_dataset
