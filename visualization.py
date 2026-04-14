"""Visualization helpers for detector, alignment, and recognition evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


def plot_detector_comparison(
    image_bgr: np.ndarray,
    detections: Dict[str, List[tuple]],
    output_path: Optional[str] = None,
) -> None:
    """Plot one image with detector-specific bounding boxes."""
    n = len(detections) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, (name, boxes) in zip(axes[1:], detections.items()):
        canvas = image_bgr.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 220, 0), 2)
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120)
    plt.show()


def plot_alignment_examples(
    unaligned_imgs: Sequence[np.ndarray],
    aligned_imgs: Sequence[np.ndarray],
    n_examples: int = 5,
    output_path: Optional[str] = None,
) -> None:
    """Visualize unaligned vs aligned face pairs."""
    n = min(n_examples, len(unaligned_imgs), len(aligned_imgs))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for i in range(n):
        axes[0, i].imshow(cv2.cvtColor(unaligned_imgs[i], cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Unaligned {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(cv2.cvtColor(aligned_imgs[i], cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Aligned {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120)
    plt.show()


def plot_augmentations(
    image_rgb: np.ndarray,
    train_transform,
    n_examples: int = 3,
    output_path: Optional[str] = None,
) -> None:
    """Show augmented samples from the training transform."""
    fig, axes = plt.subplots(1, n_examples + 1, figsize=(3 * (n_examples + 1), 3.5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i in range(n_examples):
        t = train_transform(image_rgb)
        # De-normalize for plotting.
        vis = (t.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
        axes[i + 1].imshow(vis)
        axes[i + 1].set_title(f"Aug {i+1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120)
    plt.show()


def plot_roc_curves(
    roc_data: Sequence[Dict[str, object]],
    output_path: Optional[str] = None,
) -> None:
    """Plot ROC curves for multiple methods."""
    plt.figure(figsize=(7, 6))
    for item in roc_data:
        label = str(item["label"])
        fpr = np.asarray(item["fpr"])
        tpr = np.asarray(item["tpr"])
        auc = float(item["auc"])
        eer = float(item["eer"])
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.4f}, EER={eer:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Face Verification")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120)
    plt.show()


def plot_cmc_curves(
    cmc_data: Sequence[Dict[str, object]],
    max_rank: int = 10,
    output_path: Optional[str] = None,
) -> None:
    """Plot CMC curves from rank-1 to rank-N."""
    ranks = np.arange(1, max_rank + 1)
    plt.figure(figsize=(7, 5))

    for item in cmc_data:
        label = str(item["label"])
        cmc = np.asarray(item["cmc"])[:max_rank]
        plt.plot(ranks[: len(cmc)], cmc, marker="o", lw=2, label=label)

    plt.xlabel("Rank")
    plt.ylabel("Cumulative Match Characteristic")
    plt.title("CMC Curve - Face Identification")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120)
    plt.show()


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_identities: int = 20,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> None:
    """Visualize embeddings using t-SNE for top-N identities by frequency."""
    from collections import Counter

    counts = Counter(labels.tolist())
    top_ids = [idx for idx, _ in counts.most_common(n_identities)]
    mask = np.isin(labels, top_ids)

    emb_sub = embeddings[mask]
    lbl_sub = labels[mask]

    tsne = TSNE(n_components=2, perplexity=30, random_state=seed, n_iter=1000)
    proj = tsne.fit_transform(emb_sub)

    palette = sns.color_palette("tab20", len(top_ids))
    color_map = {lab: palette[i] for i, lab in enumerate(top_ids)}

    plt.figure(figsize=(10, 8))
    for lab in top_ids:
        m = lbl_sub == lab
        plt.scatter(proj[m, 0], proj[m, 1], s=30, alpha=0.7, color=color_map[lab], label=str(lab))

    plt.title(f"t-SNE Embedding Visualization ({len(top_ids)} identities)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=130)
    plt.show()


def plot_challenge_drop(
    challenge_results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
) -> None:
    """Plot easy-vs-hard accuracy and percentage drop for each challenge."""
    names = list(challenge_results.keys())
    easy = [challenge_results[k]["easy_acc"] for k in names]
    hard = [challenge_results[k]["hard_acc"] for k in names]
    drop = [challenge_results[k]["drop_pct"] for k in names]

    x = np.arange(len(names))
    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.18, easy, width=0.36, label="Easy", color="steelblue")
    plt.bar(x + 0.18, hard, width=0.36, label="Hard", color="salmon")
    for i, d in enumerate(drop):
        plt.text(i, max(easy[i], hard[i]) + 0.02, f"-{d:.1f}%", ha="center", color="red")

    plt.xticks(x, [n.capitalize() for n in names])
    plt.ylim(0, 1.1)
    plt.ylabel("Rank-1 Accuracy")
    plt.title("Performance Under Natural Challenges")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120)
    plt.show()


def plot_top5_matches(
    examples: List[Dict[str, object]],
    images_bgr: Sequence[np.ndarray],
    output_path: Optional[str] = None,
) -> None:
    """Visualize probe and its top-5 gallery matches."""
    if not examples:
        return

    n_rows = len(examples)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, example in enumerate(examples):
        p_idx = int(example["probe_index"])
        p_lbl = int(example["probe_label"])

        axes[r, 0].imshow(cv2.cvtColor(images_bgr[p_idx], cv2.COLOR_BGR2RGB))
        axes[r, 0].set_title(f"Probe\nID={p_lbl}")
        axes[r, 0].axis("off")

        for c, match in enumerate(example["matches"], start=1):
            g_idx = int(match["gallery_index"])
            g_lbl = int(match["gallery_label"])
            score = float(match["score"])
            axes[r, c].imshow(cv2.cvtColor(images_bgr[g_idx], cv2.COLOR_BGR2RGB))
            axes[r, c].set_title(f"Top-{c}\nID={g_lbl}\n{score:.3f}")
            axes[r, c].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=120)
    plt.show()
