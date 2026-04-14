"""Training and embedding extraction helpers for face recognition models."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def train_model(
    model: torch.nn.Module,
    loss_module: torch.nn.Module,
    train_ds,
    val_ds,
    device: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    num_workers: int = 0,
    label: str = "Model",
) -> Tuple[torch.nn.Module, Dict[str, list]]:
    """Train embedding model and return checkpointed history."""
    model = model.to(device)
    loss_module = loss_module.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_module.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        loss_module.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(imgs)
            loss = loss_module(embeddings, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(imgs)

        train_loss = running_loss / max(1, len(train_ds))

        model.eval()
        loss_module.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                embeddings = model(imgs)
                val_loss += loss_module(embeddings, labels).item() * len(imgs)
        val_loss /= max(1, len(val_ds))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[{label}] epoch {epoch:03d}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

    return model, history


def extract_embeddings(
    model: torch.nn.Module,
    dataset,
    device: str,
    batch_size: int = 128,
    num_workers: int = 0,
):
    """Extract L2-normalized embeddings and labels for an entire dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings, labels = [], []

    model.eval()
    with torch.no_grad():
        for imgs, batch_labels in loader:
            emb = model(imgs.to(device)).cpu().numpy()
            embeddings.append(emb)
            labels.extend(batch_labels.numpy().tolist())

    return np.vstack(embeddings), np.array(labels, dtype=np.int64)
