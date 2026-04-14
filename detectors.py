"""Face detector wrappers and detector evaluation utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from facenet_pytorch import MTCNN
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install facenet-pytorch to use MTCNN detector.") from exc


BBox = Tuple[int, int, int, int]


@dataclass
class DetectionResult:
    """Single detector output as bounding boxes and confidences."""

    boxes: List[BBox]
    scores: List[float]


class HaarDetector:
    """OpenCV Haar Cascade face detector."""

    def __init__(self) -> None:
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, bgr_img: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return DetectionResult([], [])
        return DetectionResult([tuple(map(int, f)) for f in faces], [1.0] * len(faces))


class MTCNNDetector:
    """Deep MTCNN detector from facenet-pytorch."""

    def __init__(self, device: str, keep_all: bool = True):
        self.detector = MTCNN(keep_all=keep_all, device=device, post_process=False)

    def detect(self, bgr_img: np.ndarray) -> DetectionResult:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        boxes, probs = self.detector.detect(Image.fromarray(rgb))
        if boxes is None:
            return DetectionResult([], [])

        out_boxes: List[BBox] = []
        out_scores: List[float] = []
        for box, score in zip(boxes, probs):
            x1, y1, x2, y2 = box
            out_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            out_scores.append(float(score))
        return DetectionResult(out_boxes, out_scores)


class RetinaFaceDetector:
    """RetinaFace detector wrapper (optional dependency)."""

    def __init__(self) -> None:
        try:
            from retinaface import RetinaFace  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install retina-face package to use RetinaFaceDetector."
            ) from exc
        self._retinaface = RetinaFace

    def detect(self, bgr_img: np.ndarray) -> DetectionResult:
        detections = self._retinaface.detect_faces(bgr_img)
        if not detections:
            return DetectionResult([], [])

        boxes: List[BBox] = []
        scores: List[float] = []
        for _, det in detections.items():
            x1, y1, x2, y2 = det["facial_area"]
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            scores.append(float(det["score"]))
        return DetectionResult(boxes, scores)


def iou(box_a: BBox, box_b: BBox) -> float:
    """Compute IoU for boxes in (x, y, w, h) format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    det_boxes: Sequence[BBox],
    gt_boxes: Sequence[BBox],
    iou_thresh: float = 0.5,
) -> Tuple[int, int, int]:
    """Greedy one-to-one matching, returning (TP, FP, FN)."""
    matched_gt = set()
    tp = 0
    fp = 0

    for db in det_boxes:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            score = iou(db, gb)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_iou >= iou_thresh and best_j not in matched_gt:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def build_pseudo_gt(
    images: Sequence[np.ndarray],
    detector: MTCNNDetector,
    conf_threshold: float = 0.99,
) -> List[List[BBox]]:
    """Generate pseudo-ground-truth boxes from high-confidence MTCNN detections."""
    gt_list = []
    for img in images:
        result = detector.detect(img)
        gt = [b for b, s in zip(result.boxes, result.scores) if s >= conf_threshold]
        gt_list.append(gt)
    return gt_list


def evaluate_detector(
    detect_fn: Callable[[np.ndarray], DetectionResult],
    images: Sequence[np.ndarray],
    gt_list: Sequence[Sequence[BBox]],
    label: str,
    iou_thresh: float = 0.5,
) -> Dict[str, object]:
    """Evaluate detector using pseudo-GT and return precision/recall/F1/IoU/time."""
    total_tp = total_fp = total_fn = 0
    all_scores: List[float] = []
    all_ious: List[float] = []
    times_ms: List[float] = []

    for img, gt_boxes in zip(images, gt_list):
        t0 = time.perf_counter()
        result = detect_fn(img)
        times_ms.append((time.perf_counter() - t0) * 1000)

        tp, fp, fn = match_detections(result.boxes, gt_boxes, iou_thresh=iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for box, score in zip(result.boxes, result.scores):
            best = max((iou(box, gt) for gt in gt_boxes), default=0.0)
            all_ious.append(best)
            all_scores.append(score)

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "label": label,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "mean_time_ms": float(np.mean(times_ms)) if times_ms else 0.0,
        "scores": np.array(all_scores),
        "ious": np.array(all_ious),
    }
