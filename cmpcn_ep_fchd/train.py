from __future__ import annotations

import argparse
import json
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig, add_common_args, update_config_from_args
from .datasets import (
    EPFCHDImageTextDataset,
    encode_texts,
    stratified_labeled_unlabeled_split,
)
from .metrics import compute_classification_metrics
from .model import build_cmpcn
from .text_descriptions import (
    DEFAULT_UNLABELED_TEXT,
    EP_FCHD_CLASS_DESCRIPTIONS,
    EP_FCHD_CLASS_NAMES,
)
from .transforms import get_strong_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved = dict(batch)
    moved["image"] = batch["image"].to(device, non_blocking=True)
    moved["label"] = batch["label"].to(device, non_blocking=True)
    moved["text_encodings"] = {
        key: value.to(device, non_blocking=True)
        for key, value in batch["text_encodings"].items()
    }
    return moved


def make_loader(dataset, config: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def initialize_prototypes(model, loader: DataLoader, device: torch.device, use_multi_layer: bool) -> None:
    image_features = {class_idx: [] for class_idx in range(model.num_classes)}
    text_features = {class_idx: [] for class_idx in range(model.num_classes)}
    model.eval()
    for batch in tqdm(loader, desc="Initializing prototypes", leave=False):
        batch = batch_to_device(batch, device)
        outputs = model(
            batch["image"],
            batch["text_encodings"],
            return_all_layers=use_multi_layer,
        )
        image_embeddings, text_embeddings = outputs[-1] if use_multi_layer else outputs
        for image_embedding, text_embedding, label in zip(
            image_embeddings,
            text_embeddings,
            batch["label"],
        ):
            label_idx = int(label.item())
            if label_idx >= 0:
                image_features[label_idx].append(image_embedding.detach().cpu())
                text_features[label_idx].append(text_embedding.detach().cpu())
    model.initialize_prototypes_from_features(image_features, text_features)


@torch.no_grad()
def predict_candidate_logits(
    model,
    images: torch.Tensor,
    tokenizer,
    config: TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    """Score each image against every EP_FCHD class description."""

    batch_size = images.size(0)
    num_classes = len(EP_FCHD_CLASS_NAMES)
    candidate_texts = [
        EP_FCHD_CLASS_DESCRIPTIONS[class_name]
        for _ in range(batch_size)
        for class_name in EP_FCHD_CLASS_NAMES
    ]
    repeated_images = images.repeat_interleave(num_classes, dim=0)
    candidate_text_encodings = encode_texts(
        tokenizer,
        candidate_texts,
        max_length=config.text_max_length,
        device=device,
    )
    image_embeddings, text_embeddings = model(repeated_images, candidate_text_encodings)
    candidate_logits = model.classify(image_embeddings, text_embeddings)
    candidate_logits = candidate_logits.view(batch_size, num_classes, config.num_classes)
    class_indices = torch.arange(num_classes, device=device)
    return candidate_logits[:, class_indices, class_indices]


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    tokenizer,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch_to_device(batch, device)
        logits = predict_candidate_logits(model, batch["image"], tokenizer, config, device)
        total_loss += F.cross_entropy(logits, batch["label"]).item()
        probs = torch.softmax(logits, dim=-1)
        labels.extend(batch["label"].cpu().tolist())
        predictions.extend(torch.argmax(probs, dim=-1).cpu().tolist())
        probabilities.extend(probs.cpu().tolist())

    metrics = compute_classification_metrics(labels, predictions, probabilities, config.num_classes)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def build_unlabeled_loader(
    config: TrainConfig,
    tokenizer,
    image_processor,
    split_unlabeled,
) -> DataLoader | None:
    if split_unlabeled is not None and len(split_unlabeled) > 0:
        return make_loader(split_unlabeled, config, shuffle=True)

    if not config.use_unlabel_folder:
        return None

    unlabel_dir = config.unlabel_dir
    if not unlabel_dir.exists():
        return None

    unlabel_dataset = EPFCHDImageTextDataset(
        unlabel_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config.text_max_length,
        unlabeled=True,
    )
    return make_loader(unlabel_dataset, config, shuffle=True)


def train(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    device = torch.device(config.device if torch.cuda.is_available() and "cuda" in config.device else "cpu")
    model, tokenizer, image_processor = build_cmpcn(config)
    model.to(device)

    train_dataset = EPFCHDImageTextDataset(
        config.train_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config.text_max_length,
    )
    test_dataset = EPFCHDImageTextDataset(
        config.test_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config.text_max_length,
    )

    labeled_subset, split_unlabeled = stratified_labeled_unlabeled_split(
        train_dataset,
        labeled_ratio=config.labeled_ratio,
        seed=config.seed,
    )
    train_loader = make_loader(labeled_subset, config, shuffle=True)
    prototype_loader = make_loader(train_dataset, config, shuffle=False)
    test_loader = make_loader(test_dataset, config, shuffle=False)
    unlabeled_loader = build_unlabeled_loader(config, tokenizer, image_processor, split_unlabeled)
    strong_transform = get_strong_transform(config.image_size)

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    if config.use_prototypes:
        initialize_prototypes(model, prototype_loader, device, config.use_multi_layer_loss)

    best_accuracy = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_steps = len(train_loader)
        unlabeled_iter = cycle(unlabeled_loader) if unlabeled_loader is not None else None

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for step, labeled_batch in enumerate(progress, start=1):
            labeled_batch = batch_to_device(labeled_batch, device)
            layer_outputs = model(
                labeled_batch["image"],
                labeled_batch["text_encodings"],
                return_all_layers=config.use_multi_layer_loss,
            )
            if config.use_multi_layer_loss:
                image_embeddings, text_embeddings = layer_outputs[-1]
                multi_layer_loss = model.compute_multi_layer_loss(
                    layer_outputs,
                    labeled_batch["label"],
                )
            else:
                image_embeddings, text_embeddings = layer_outputs
                multi_layer_loss = image_embeddings.new_tensor(0.0)

            logits = model.classify(image_embeddings, text_embeddings)
            classification_loss = model.compute_focal_loss(logits, labeled_batch["label"])
            contrastive_loss = model.compute_info_nce_loss(image_embeddings, text_embeddings)
            prototype_loss = model.compute_prototype_loss(
                image_embeddings,
                text_embeddings,
                labeled_batch["label"],
            )
            consistency_loss = image_embeddings.new_tensor(0.0)

            if unlabeled_iter is not None:
                unlabeled_batch = batch_to_device(next(unlabeled_iter), device)
                consistency_loss = compute_consistency_loss(
                    model,
                    tokenizer,
                    unlabeled_batch,
                    strong_transform,
                    config,
                    device,
                    train_dataset.idx_to_class,
                )

            loss = (
                config.class_loss_weight * classification_loss
                + config.contrastive_loss_weight * contrastive_loss
                + config.prototype_loss_weight * prototype_loss
                + config.consistency_loss_weight * consistency_loss
                + config.multi_layer_loss_weight * multi_layer_loss
            )
            loss = loss / max(config.gradient_accumulation_steps, 1)
            loss.backward()

            if step % config.gradient_accumulation_steps == 0 or step == total_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                model.update_prototypes(
                    image_embeddings.detach(),
                    text_embeddings.detach(),
                    labeled_batch["label"].detach(),
                )

            total_loss += loss.item() * max(config.gradient_accumulation_steps, 1)
            progress.set_postfix(loss=f"{total_loss / step:.4f}")

        metrics = evaluate(model, test_loader, tokenizer, device, config)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(total_steps, 1)
        history.append(metrics)
        with (config.output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        accuracy = float(metrics["accuracy"])
        scheduler.step(accuracy)
        print(
            f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f}, "
            f"test_loss={metrics['loss']:.4f}, accuracy={accuracy:.4f}, "
            f"f1_weighted={metrics['f1_weighted']:.4f}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            save_checkpoint(config.output_dir / "best_model.pt", model, optimizer, config, epoch, metrics)
        else:
            epochs_without_improvement += 1

        if config.save_every and epoch % config.save_every == 0:
            save_checkpoint(config.output_dir / f"checkpoint_epoch_{epoch}.pt", model, optimizer, config, epoch, metrics)

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    save_checkpoint(config.output_dir / "last_model.pt", model, optimizer, config, epoch, history[-1])
    return {"best_accuracy": best_accuracy, "history": history}


def compute_consistency_loss(
    model,
    tokenizer,
    unlabeled_batch: dict[str, object],
    strong_transform,
    config: TrainConfig,
    device: torch.device,
    idx_to_class: dict[int, str],
) -> torch.Tensor:
    images = unlabeled_batch["image"]
    batch_size = images.size(0)
    default_text = encode_texts(
        tokenizer,
        [DEFAULT_UNLABELED_TEXT] * batch_size,
        max_length=config.text_max_length,
        device=device,
    )

    with torch.no_grad():
        weak_image_embeddings, weak_text_embeddings = model(images, default_text)
        weak_logits = model.classify(weak_image_embeddings, weak_text_embeddings)
        weak_probs = torch.softmax(weak_logits, dim=-1)
        confidence, pseudo_labels = torch.max(weak_probs, dim=-1)
        confident = confidence >= config.pseudo_label_threshold

    if confident.sum() == 0:
        return images.new_tensor(0.0)

    pseudo_texts = [
        EP_FCHD_CLASS_DESCRIPTIONS[idx_to_class[int(label.item())]]
        for label in pseudo_labels
    ]
    pseudo_text_encodings = encode_texts(
        tokenizer,
        pseudo_texts,
        max_length=config.text_max_length,
        device=device,
    )
    strong_images = strong_transform(images)
    strong_image_embeddings, strong_text_embeddings = model(strong_images, pseudo_text_encodings)
    strong_logits = model.classify(strong_image_embeddings, strong_text_embeddings)
    return F.kl_div(
        F.log_softmax(strong_logits[confident], dim=-1),
        weak_probs[confident],
        reduction="batchmean",
    )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, object],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "config": config.to_dict(),
            "class_names": EP_FCHD_CLASS_NAMES,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train CMPCN on EP_FCHD.")
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--labeled-ratio", type=float, default=TrainConfig.labeled_ratio)
    parser.add_argument("--use-unlabel-folder", action="store_true")
    parser.add_argument("--pseudo-label-threshold", type=float, default=TrainConfig.pseudo_label_threshold)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=TrainConfig.gradient_accumulation_steps)
    parser.add_argument("--prototypes-per-class", type=int, default=TrainConfig.prototypes_per_class)
    parser.add_argument("--prototype-momentum", type=float, default=TrainConfig.prototype_momentum)
    parser.add_argument("--cross-attention-depth", type=int, default=TrainConfig.cross_attention_depth)
    parser.add_argument("--cross-attention-heads", type=int, default=TrainConfig.cross_attention_heads)
    parser.add_argument("--no-prototypes", dest="use_prototypes", action="store_false")
    parser.add_argument("--no-cross-attention", dest="use_cross_attention", action="store_false")
    parser.add_argument("--no-multi-layer-loss", dest="use_multi_layer_loss", action="store_false")
    parser.add_argument("--freeze-image-encoder", action="store_true")
    parser.add_argument("--freeze-text-encoder", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=TrainConfig.early_stopping_patience)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    args = parser.parse_args()
    return update_config_from_args(TrainConfig(), args)


def main() -> None:
    config = parse_args()
    result = train(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
