import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from pycocotools.cocoeval import COCOeval


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    print_freq: int = 50,
) -> Dict[str, float]:
    model.train()
    start = time.time()
    running_loss = 0.0
    num_steps = 0

    for step, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        running_loss += float(total_loss.detach().cpu())
        num_steps += 1

        if (step + 1) % print_freq == 0:
            loss_parts = ", ".join(f"{k}={float(v.detach().cpu()):.4f}" for k, v in loss_dict.items())
            print(f"[Epoch {epoch}] step {step + 1}/{len(data_loader)}: total={float(total_loss):.4f} ({loss_parts})")

    elapsed = time.time() - start
    avg_loss = running_loss / max(1, num_steps)
    print(f"[Epoch {epoch}] done in {elapsed:.1f}s, avg_train_loss={avg_loss:.4f}")
    return {"train_loss": avg_loss, "epoch_time_sec": elapsed}


def _prediction_to_coco_results(
    image_id: int,
    prediction: Dict[str, torch.Tensor],
    score_threshold: float,
) -> List[Dict]:
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    keypoints = prediction["keypoints"].detach().cpu()

    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    keypoints = keypoints[keep]

    results: List[Dict] = []
    for i in range(len(scores)):
        kpt = keypoints[i].clone()
        visibility = torch.where(kpt[:, 2] > 0, torch.tensor(2.0), torch.tensor(0.0))
        kpt[:, 2] = visibility
        results.append(
            {
                "image_id": image_id,
                "category_id": 1,
                "keypoints": kpt.reshape(-1).tolist(),
                "score": float(scores[i]),
            }
        )
    return results


@torch.no_grad()
def evaluate_coco_keypoints(
    model: torch.nn.Module,
    data_loader: Iterable,
    coco_gt,
    device: torch.device,
    score_threshold: float = 0.05,
    result_json_path: Optional[Path] = None,
) -> Dict[str, float]:
    model.eval()
    results: List[Dict] = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"])
            results.extend(_prediction_to_coco_results(image_id, output, score_threshold=score_threshold))

    if result_json_path is not None:
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        with result_json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f)

    if not results:
        print("No predictions produced above score threshold; returning zeros.")
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
    eval_image_ids = getattr(getattr(data_loader, "dataset", None), "image_ids", None)
    if eval_image_ids is not None:
        coco_eval.params.imgIds = list(eval_image_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AR": float(stats[5]),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    epoch: int,
    extra: Optional[Dict] = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler=None,
    map_location: str = "cpu",
) -> int:
    payload = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if lr_scheduler is not None and payload.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
    start_epoch = int(payload.get("epoch", -1)) + 1
    print(f"Loaded checkpoint: {checkpoint_path} (resume from epoch {start_epoch})")
    return start_epoch
