from typing import Dict, Iterable, List, Tuple

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

COCO_KEYPOINT_CONNECTIONS: List[Tuple[int, int]] = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
]


def draw_pose_predictions(
    image: Image.Image,
    prediction: Dict[str, torch.Tensor],
    score_threshold: float = 0.5,
    keypoint_threshold: float = 0.1,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    keypoints = prediction["keypoints"].detach().cpu()

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    keypoints = keypoints[keep]

    for box, score, kpt in zip(boxes, scores, keypoints):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=2)
        draw.text((x1, max(0, y1 - 12)), f"{float(score):.2f}", fill=(255, 140, 0))

        points: List[Tuple[float, float, float]] = [(float(x), float(y), float(v)) for x, y, v in kpt.tolist()]
        for x, y, v in points:
            if v < keypoint_threshold:
                continue
            r = 2.0
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 127))

        for u, v in COCO_KEYPOINT_CONNECTIONS:
            x1p, y1p, v1 = points[u]
            x2p, y2p, v2 = points[v]
            if v1 >= keypoint_threshold and v2 >= keypoint_threshold:
                draw.line([(x1p, y1p), (x2p, y2p)], fill=(0, 220, 255), width=2)

    return canvas


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return F.pil_to_tensor(image).float() / 255.0
