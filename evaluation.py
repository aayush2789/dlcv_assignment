"""Verification, identification, and challenge-condition evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class VerificationResult:
    """Outputs from 1:1 face verification evaluation."""

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    eer: float
    tar_far_001: float
    tar_far_0001: float
    scores: np.ndarray
    labels: np.ndarray


@dataclass
class IdentificationResult:
    """Outputs from 1:N face identification evaluation."""

    cmc: np.ndarray
    rank1: float
    rank5: float
    gallery_size: int
    probe_size: int


@dataclass
class ChallengeSplit:
    """Index subsets for condition-oriented challenge analysis."""

    frontal: np.ndarray
    profile: np.ndarray
    normal_light: np.ndarray
    poor_light: np.ndarray
    unoccluded: np.ndarray
    occluded: np.ndarray


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for pre-normalized embeddings."""
    return float(np.dot(a, b))


def build_verification_pairs(
    labels: np.ndarray,
    n_same: int = 3000,
    n_diff: int = 3000,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Build 6000 LFW-style verification pairs (same + different identity)."""
    rng = np.random.RandomState(seed)
    id_to_idx: Dict[int, List[int]] = {}
    for i, label in enumerate(labels):
        id_to_idx.setdefault(int(label), []).append(i)

    genuine_ids = [k for k, idxs in id_to_idx.items() if len(idxs) >= 2]
    all_ids = list(id_to_idx.keys())

    pairs: List[Tuple[int, int]] = []
    pair_labels: List[int] = []

    for _ in range(n_same):
        identity = int(rng.choice(genuine_ids))
        i, j = rng.choice(id_to_idx[identity], 2, replace=False)
        pairs.append((int(i), int(j)))
        pair_labels.append(1)

    for _ in range(n_diff):
        id_a, id_b = rng.choice(all_ids, 2, replace=False)
        i = int(rng.choice(id_to_idx[int(id_a)]))
        j = int(rng.choice(id_to_idx[int(id_b)]))
        pairs.append((i, j))
        pair_labels.append(0)

    return pairs, np.array(pair_labels, dtype=np.int64)


def evaluate_verification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_same: int = 3000,
    n_diff: int = 3000,
    seed: int = 42,
) -> VerificationResult:
    """Compute ROC/AUC/EER/TAR at target FARs for 1:1 verification."""
    pairs, pair_labels = build_verification_pairs(labels, n_same=n_same, n_diff=n_diff, seed=seed)
    scores = np.array([cosine_similarity(embeddings[i], embeddings[j]) for i, j in pairs])

    fpr, tpr, thresholds = roc_curve(pair_labels, scores)
    auc = float(roc_auc_score(pair_labels, scores))

    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) * 0.5)

    def tar_at_far(target_far: float) -> float:
        idx = int(np.searchsorted(fpr, target_far, side="left"))
        idx = min(idx, len(tpr) - 1)
        return float(tpr[idx])

    return VerificationResult(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc=auc,
        eer=eer,
        tar_far_001=tar_at_far(0.01),
        tar_far_0001=tar_at_far(0.001),
        scores=scores,
        labels=pair_labels,
    )


def build_identification_protocol(
    labels: np.ndarray,
    gallery_identities: int = 100,
    gallery_per_id: int = 5,
    probe_identities: int = 50,
    probe_per_id: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create fixed gallery/probe indices from a test set.

    Probe identities are sampled from gallery identities so closed-set rank metrics are valid.
    """
    rng = np.random.RandomState(seed)
    id_to_idx: Dict[int, List[int]] = {}
    for i, label in enumerate(labels):
        id_to_idx.setdefault(int(label), []).append(i)

    needed = gallery_per_id + probe_per_id
    eligible_ids = [k for k, idxs in id_to_idx.items() if len(idxs) >= needed]
    if len(eligible_ids) < gallery_identities:
        raise ValueError(
            f"Need at least {gallery_identities} eligible identities, got {len(eligible_ids)}"
        )

    selected_gallery_ids = rng.choice(eligible_ids, size=gallery_identities, replace=False)
    selected_probe_ids = rng.choice(selected_gallery_ids, size=probe_identities, replace=False)

    gallery_indices: List[int] = []
    probe_indices: List[int] = []

    for identity in selected_gallery_ids:
        picks = rng.choice(id_to_idx[int(identity)], size=needed, replace=False)
        gallery_indices.extend(picks[:gallery_per_id].tolist())
        if int(identity) in set(map(int, selected_probe_ids)):
            probe_indices.extend(picks[gallery_per_id : gallery_per_id + probe_per_id].tolist())

    return np.array(gallery_indices, dtype=np.int64), np.array(probe_indices, dtype=np.int64)


def evaluate_identification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    gallery_indices: np.ndarray,
    probe_indices: np.ndarray,
    max_rank: int = 10,
) -> IdentificationResult:
    """Compute Rank-1, Rank-5, and CMC for 1:N identification."""
    gallery_emb = embeddings[gallery_indices]
    gallery_lbl = labels[gallery_indices]
    probe_emb = embeddings[probe_indices]
    probe_lbl = labels[probe_indices]

    sim = probe_emb @ gallery_emb.T
    ranked = np.argsort(-sim, axis=1)

    cmc = np.zeros(max_rank, dtype=np.float32)
    for row, true_label in enumerate(probe_lbl):
        hit_rank = None
        for r in range(min(max_rank, ranked.shape[1])):
            if gallery_lbl[ranked[row, r]] == true_label:
                hit_rank = r
                break
        if hit_rank is not None:
            cmc[hit_rank:] += 1.0
    cmc /= max(1, len(probe_lbl))

    rank1 = float(cmc[0]) if len(cmc) > 0 else 0.0
    rank5 = float(cmc[4]) if len(cmc) >= 5 else float(cmc[-1])

    return IdentificationResult(
        cmc=cmc,
        rank1=rank1,
        rank5=rank5,
        gallery_size=len(gallery_indices),
        probe_size=len(probe_indices),
    )


def classify_challenge_conditions(
    images_bgr: Sequence[np.ndarray],
    landmarks_list: Sequence[Optional[np.ndarray]],
    face_boxes: Sequence[Optional[Tuple[int, int, int, int]]],
    frontal_angle_deg: float = 8.0,
    profile_angle_deg: float = 18.0,
) -> ChallengeSplit:
    """Split samples into natural LFW pose/lighting/occlusion subsets.

    Heuristics:
    - Pose: eye-line tilt angle from landmarks.
    - Lighting: low luminance mean or low contrast in face ROI.
    - Occlusion: low edge energy around mouth relative to upper-face region.
    """
    frontal_idx: List[int] = []
    profile_idx: List[int] = []
    normal_light_idx: List[int] = []
    poor_light_idx: List[int] = []
    unoccluded_idx: List[int] = []
    occluded_idx: List[int] = []

    for i, (img, lm, box) in enumerate(zip(images_bgr, landmarks_list, face_boxes)):
        # Pose bucket from eye-line angle when landmarks are available.
        if lm is not None:
            left_eye, right_eye = lm[0], lm[1]
            dx = float(right_eye[0] - left_eye[0])
            dy = float(right_eye[1] - left_eye[1])
            angle = abs(np.degrees(np.arctan2(dy, dx + 1e-9)))
            if angle <= frontal_angle_deg:
                frontal_idx.append(i)
            if angle >= profile_angle_deg:
                profile_idx.append(i)

        # Use detected face ROI if possible, else fallback to full image.
        if box is not None:
            x, y, w, h = box
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            face = img[y : y + h, x : x + w]
        else:
            face = img

        if face.size == 0:
            continue

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        mean_luma = float(gray.mean())
        contrast = float(gray.std())

        if mean_luma < 85 or contrast < 32:
            poor_light_idx.append(i)
        else:
            normal_light_idx.append(i)

        h, w = gray.shape[:2]
        top = gray[: max(1, int(h * 0.4)), :]
        mouth_band = gray[int(h * 0.62) : min(h, int(h * 0.9)), :]
        if mouth_band.size == 0:
            continue

        top_edges = cv2.Canny(top, 60, 120).mean()
        mouth_edges = cv2.Canny(mouth_band, 60, 120).mean()
        ratio = mouth_edges / (top_edges + 1e-6)

        if ratio < 0.55:
            occluded_idx.append(i)
        else:
            unoccluded_idx.append(i)

    return ChallengeSplit(
        frontal=np.array(frontal_idx, dtype=np.int64),
        profile=np.array(profile_idx, dtype=np.int64),
        normal_light=np.array(normal_light_idx, dtype=np.int64),
        poor_light=np.array(poor_light_idx, dtype=np.int64),
        unoccluded=np.array(unoccluded_idx, dtype=np.int64),
        occluded=np.array(occluded_idx, dtype=np.int64),
    )


def evaluate_rank1_on_indices(
    embeddings: np.ndarray,
    labels: np.ndarray,
    gallery_indices: np.ndarray,
    probe_indices: np.ndarray,
) -> float:
    """Compute Rank-1 on a probe subset against a fixed gallery."""
    if len(probe_indices) == 0:
        return 0.0

    gallery_emb = embeddings[gallery_indices]
    gallery_lbl = labels[gallery_indices]
    probe_emb = embeddings[probe_indices]
    probe_lbl = labels[probe_indices]

    sim = probe_emb @ gallery_emb.T
    best = np.argmax(sim, axis=1)
    pred = gallery_lbl[best]
    return float((pred == probe_lbl).mean())


def evaluate_challenge_drop(
    embeddings: np.ndarray,
    labels: np.ndarray,
    gallery_indices: np.ndarray,
    probe_indices: np.ndarray,
    challenge_split: ChallengeSplit,
) -> Dict[str, Dict[str, float]]:
    """Measure Rank-1 drop between easy vs hard subsets for each challenge."""

    def valid_probe_subset(indices: np.ndarray) -> np.ndarray:
        probe_set = set(map(int, probe_indices.tolist()))
        return np.array([i for i in indices.tolist() if int(i) in probe_set], dtype=np.int64)

    frontal_probe = valid_probe_subset(challenge_split.frontal)
    profile_probe = valid_probe_subset(challenge_split.profile)
    normal_probe = valid_probe_subset(challenge_split.normal_light)
    poor_probe = valid_probe_subset(challenge_split.poor_light)
    unocc_probe = valid_probe_subset(challenge_split.unoccluded)
    occ_probe = valid_probe_subset(challenge_split.occluded)

    results: Dict[str, Dict[str, float]] = {}

    pairs = {
        "pose": (frontal_probe, profile_probe, "frontal", "profile"),
        "lighting": (normal_probe, poor_probe, "normal", "poor"),
        "occlusion": (unocc_probe, occ_probe, "unoccluded", "occluded"),
    }

    for key, (easy_idx, hard_idx, easy_name, hard_name) in pairs.items():
        easy_acc = evaluate_rank1_on_indices(embeddings, labels, gallery_indices, easy_idx)
        hard_acc = evaluate_rank1_on_indices(embeddings, labels, gallery_indices, hard_idx)
        drop_pct = ((easy_acc - hard_acc) / (easy_acc + 1e-9)) * 100.0
        results[key] = {
            "easy_acc": easy_acc,
            "hard_acc": hard_acc,
            "drop_pct": float(drop_pct),
            "easy_count": float(len(easy_idx)),
            "hard_count": float(len(hard_idx)),
            "easy_name": easy_name,
            "hard_name": hard_name,
        }

    return results


def topk_gallery_matches(
    embeddings: np.ndarray,
    labels: np.ndarray,
    gallery_indices: np.ndarray,
    probe_indices: np.ndarray,
    k: int = 5,
    n_examples: int = 5,
    seed: int = 42,
) -> List[Dict[str, object]]:
    """Return top-k gallery matches for probe visualization."""
    rng = np.random.RandomState(seed)
    chosen = rng.choice(probe_indices, size=min(n_examples, len(probe_indices)), replace=False)

    gallery_emb = embeddings[gallery_indices]
    gallery_lbl = labels[gallery_indices]

    examples: List[Dict[str, object]] = []
    for probe_idx in chosen:
        sim = embeddings[int(probe_idx)] @ gallery_emb.T
        ranked = np.argsort(-sim)[:k]
        examples.append(
            {
                "probe_index": int(probe_idx),
                "probe_label": int(labels[int(probe_idx)]),
                "matches": [
                    {
                        "gallery_index": int(gallery_indices[r]),
                        "gallery_label": int(gallery_lbl[r]),
                        "score": float(sim[r]),
                    }
                    for r in ranked
                ],
            }
        )
    return examples
