"""Face alignment and data augmentation utilities."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from facenet_pytorch import MTCNN


ARCFACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class LandmarkAligner:
    """MTCNN landmark detector + ArcFace-style similarity alignment."""

    def __init__(self, device: str, output_size: Tuple[int, int] = (112, 112)):
        self.detector = MTCNN(keep_all=False, device=device, post_process=False)
        self.output_size = output_size

    def get_landmarks(self, bgr_img: np.ndarray) -> Optional[np.ndarray]:
        """Return 5-point landmark array or None if no face was found."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        _, _, landmarks = self.detector.detect(Image.fromarray(rgb), landmarks=True)
        if landmarks is None or len(landmarks) == 0:
            return None
        return landmarks[0].astype(np.float32)

    def align(self, bgr_img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align one face to canonical ArcFace template using similarity transform."""
        M, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_SRC, method=cv2.RANSAC)
        if M is None:
            return cv2.resize(bgr_img, self.output_size)
        return cv2.warpAffine(bgr_img, M, self.output_size, flags=cv2.INTER_LINEAR)

    def align_or_resize(self, bgr_img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Align image if landmarks exist; otherwise fallback to resize."""
        lm = self.get_landmarks(bgr_img)
        if lm is None:
            return cv2.resize(bgr_img, self.output_size), False
        return self.align(bgr_img, lm), True


def build_aligned_dataset(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    aligner: LandmarkAligner,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], int]:
    """Build aligned and unaligned image lists using the same successful detections."""
    aligned_imgs: List[np.ndarray] = []
    unaligned_imgs: List[np.ndarray] = []
    out_labels: List[str] = []
    failed = 0

    iterator = zip(images, labels)
    if max_samples is not None:
        iterator = zip(images[:max_samples], labels[:max_samples])

    for img, label in iterator:
        lm = aligner.get_landmarks(img)
        if lm is None:
            failed += 1
            continue
        aligned_imgs.append(aligner.align(img, lm))
        unaligned_imgs.append(cv2.resize(img, aligner.output_size))
        out_labels.append(label)

    return aligned_imgs, unaligned_imgs, out_labels, failed


def build_transforms() -> Tuple[Callable, Callable]:
    """Return train-time augmentation transform and deterministic eval transform."""
    train_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t + torch.randn_like(t) * 0.015),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    return train_tf, eval_tf
