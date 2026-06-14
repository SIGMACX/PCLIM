from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig


class CrossModalBlock(nn.Module):
    """Self-attention plus bidirectional image-text cross-attention."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.image_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.text_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.image_to_text_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.text_to_image_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.image_self_norm = nn.LayerNorm(embed_dim)
        self.text_self_norm = nn.LayerNorm(embed_dim)
        self.image_cross_norm = nn.LayerNorm(embed_dim)
        self.text_cross_norm = nn.LayerNorm(embed_dim)
        self.image_ffn_norm = nn.LayerNorm(embed_dim)
        self.text_ffn_norm = nn.LayerNorm(embed_dim)

        self.image_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.text_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_self, _ = self.image_self_attn(image_tokens, image_tokens, image_tokens)
        image_tokens = self.image_self_norm(image_tokens + image_self)

        text_self, _ = self.text_self_attn(
            text_tokens,
            text_tokens,
            text_tokens,
            key_padding_mask=text_key_padding_mask,
        )
        text_tokens = self.text_self_norm(text_tokens + text_self)

        image_cross, _ = self.image_to_text_attn(
            image_tokens,
            text_tokens,
            text_tokens,
            key_padding_mask=text_key_padding_mask,
        )
        image_tokens = self.image_cross_norm(image_tokens + image_cross)

        text_cross, _ = self.text_to_image_attn(text_tokens, image_tokens, image_tokens)
        text_tokens = self.text_cross_norm(text_tokens + text_cross)

        image_tokens = self.image_ffn_norm(image_tokens + self.image_ffn(image_tokens))
        text_tokens = self.text_ffn_norm(text_tokens + self.text_ffn(text_tokens))
        return image_tokens, text_tokens


class CMPCNModel(nn.Module):
    """Cross-Modal Prototype-based Contrastive Network.

    The model follows the main project design: a ViT-style image encoder, a
    BERT-style text encoder, bidirectional cross-modal attention, class
    prototypes for both modalities, and contrastive/classification objectives.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        image_hidden_size: int,
        text_hidden_size: int,
        num_classes: int = 4,
        prototypes_per_class: int = 5,
        prototype_momentum: float = 0.9,
        memory_size: int = 200,
        use_prototypes: bool = True,
        use_cross_attention: bool = True,
        cross_attention_heads: int = 8,
        cross_attention_depth: int = 2,
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.embed_dim = embed_dim or image_hidden_size
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.prototype_momentum = prototype_momentum
        self.memory_size = memory_size
        self.use_prototypes = use_prototypes
        self.use_cross_attention = use_cross_attention

        self.image_projection = (
            nn.Identity()
            if image_hidden_size == self.embed_dim
            else nn.Linear(image_hidden_size, self.embed_dim)
        )
        self.text_projection = (
            nn.Identity()
            if text_hidden_size == self.embed_dim
            else nn.Linear(text_hidden_size, self.embed_dim)
        )

        self.cross_modal_blocks = nn.ModuleList(
            [
                CrossModalBlock(self.embed_dim, cross_attention_heads)
                for _ in range(cross_attention_depth if use_cross_attention else 0)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, num_classes),
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        if use_prototypes:
            self.image_prototypes = nn.Parameter(
                F.normalize(torch.randn(num_classes, prototypes_per_class, self.embed_dim), dim=-1)
            )
            self.text_prototypes = nn.Parameter(
                F.normalize(torch.randn(num_classes, prototypes_per_class, self.embed_dim), dim=-1)
            )
            self.register_buffer(
                "image_memory_pool",
                torch.zeros(num_classes, memory_size, self.embed_dim),
            )
            self.register_buffer(
                "text_memory_pool",
                torch.zeros(num_classes, memory_size, self.embed_dim),
            )
            self.register_buffer("memory_ptr", torch.zeros(num_classes, dtype=torch.long))
            self.register_buffer("memory_count", torch.zeros(num_classes, dtype=torch.long))

    def freeze_image_encoder(self) -> None:
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False

    def freeze_text_encoder(self) -> None:
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

    def _encode_tokens(
        self,
        images: torch.Tensor,
        text_encodings: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        image_output = self.image_encoder(pixel_values=images)
        image_tokens = self.image_projection(image_output.last_hidden_state)

        attention_mask = text_encodings.get("attention_mask")
        text_output = self.text_encoder(
            input_ids=text_encodings["input_ids"],
            attention_mask=attention_mask,
        )
        text_tokens = self.text_projection(text_output.last_hidden_state)
        text_padding_mask = None
        if attention_mask is not None:
            text_padding_mask = ~attention_mask.bool()
        return image_tokens, text_tokens, text_padding_mask

    def _pool_text(
        self,
        text_tokens: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_padding_mask is None:
            return text_tokens.mean(dim=1)
        valid_mask = (~text_padding_mask).unsqueeze(-1).to(text_tokens.dtype)
        return (text_tokens * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

    def _pool_embeddings(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embedding = F.normalize(image_tokens.mean(dim=1), dim=-1)
        text_embedding = F.normalize(self._pool_text(text_tokens, text_padding_mask), dim=-1)
        return image_embedding, text_embedding

    def forward(
        self,
        images: torch.Tensor,
        text_encodings: dict[str, torch.Tensor],
        return_all_layers: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor]:
        image_tokens, text_tokens, text_padding_mask = self._encode_tokens(images, text_encodings)

        if not self.cross_modal_blocks:
            embeddings = self._pool_embeddings(image_tokens, text_tokens, text_padding_mask)
            return [embeddings] if return_all_layers else embeddings

        layer_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.cross_modal_blocks:
            image_tokens, text_tokens = block(image_tokens, text_tokens, text_padding_mask)
            layer_outputs.append(self._pool_embeddings(image_tokens, text_tokens, text_padding_mask))

        return layer_outputs if return_all_layers else layer_outputs[-1]

    def classify(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([image_embeddings, text_embeddings], dim=-1))

    def compute_focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gamma: float = 2.0,
        alpha: float | None = 0.25,
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        if alpha is None:
            alpha_weight = 1.0
        else:
            alpha_weight = torch.full_like(labels, fill_value=alpha, dtype=logits.dtype)
            alpha_weight[labels == 0] = 1.0 - alpha
        return (alpha_weight * (1.0 - pt).pow(gamma) * ce_loss).mean()

    def compute_info_nce_loss(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        logits = image_embeddings @ text_embeddings.T
        logits = logits * self.logit_scale.exp().clamp(max=100)
        targets = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
        )

    def compute_multi_layer_loss(
        self,
        layer_outputs: Iterable[tuple[torch.Tensor, torch.Tensor]],
        labels: torch.Tensor,
        prototype_weight: float = 0.5,
    ) -> torch.Tensor:
        losses = []
        for image_embeddings, text_embeddings in layer_outputs:
            loss = self.compute_info_nce_loss(image_embeddings, text_embeddings)
            if self.use_prototypes:
                loss = loss + prototype_weight * self.compute_prototype_loss(
                    image_embeddings,
                    text_embeddings,
                    labels,
                )
            losses.append(loss)
        return torch.stack(losses).mean()

    def compute_prototype_loss(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.1,
    ) -> torch.Tensor:
        if not self.use_prototypes or self.num_classes <= 1:
            return image_embeddings.new_tensor(0.0)

        image_prototypes = F.normalize(self.image_prototypes, dim=-1)
        text_prototypes = F.normalize(self.text_prototypes, dim=-1)
        losses = []

        for image_embedding, text_embedding, label in zip(image_embeddings, text_embeddings, labels):
            label_idx = int(label.item())
            if label_idx < 0:
                continue
            pos_image = F.cosine_similarity(
                image_embedding.unsqueeze(0), image_prototypes[label_idx], dim=-1
            ).max()
            pos_text = F.cosine_similarity(
                text_embedding.unsqueeze(0), text_prototypes[label_idx], dim=-1
            ).max()

            neg_image_values = []
            neg_text_values = []
            for class_idx in range(self.num_classes):
                if class_idx == label_idx:
                    continue
                neg_image_values.append(
                    F.cosine_similarity(
                        image_embedding.unsqueeze(0), image_prototypes[class_idx], dim=-1
                    ).max()
                )
                neg_text_values.append(
                    F.cosine_similarity(
                        text_embedding.unsqueeze(0), text_prototypes[class_idx], dim=-1
                    ).max()
                )

            neg_image = torch.stack(neg_image_values).max()
            neg_text = torch.stack(neg_text_values).max()
            losses.append(0.5 * (F.relu(neg_image - pos_image + margin) + F.relu(neg_text - pos_text + margin)))

        if not losses:
            return image_embeddings.new_tensor(0.0)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def update_prototypes(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if not self.use_prototypes:
            return

        grouped_image: dict[int, list[torch.Tensor]] = defaultdict(list)
        grouped_text: dict[int, list[torch.Tensor]] = defaultdict(list)
        for image_embedding, text_embedding, label in zip(image_embeddings, text_embeddings, labels):
            label_idx = int(label.item())
            if label_idx >= 0:
                grouped_image[label_idx].append(image_embedding.detach())
                grouped_text[label_idx].append(text_embedding.detach())

        for class_idx in grouped_image:
            class_images = torch.stack(grouped_image[class_idx])
            class_texts = torch.stack(grouped_text[class_idx])
            class_images = _expand_to_count(class_images, self.prototypes_per_class)
            class_texts = _expand_to_count(class_texts, self.prototypes_per_class)

            self.image_prototypes[class_idx].mul_(self.prototype_momentum).add_(
                class_images,
                alpha=1.0 - self.prototype_momentum,
            )
            self.text_prototypes[class_idx].mul_(self.prototype_momentum).add_(
                class_texts,
                alpha=1.0 - self.prototype_momentum,
            )
            self.image_prototypes[class_idx].copy_(F.normalize(self.image_prototypes[class_idx], dim=-1))
            self.text_prototypes[class_idx].copy_(F.normalize(self.text_prototypes[class_idx], dim=-1))
            self._update_memory(class_idx, class_images, class_texts)

    @torch.no_grad()
    def _update_memory(
        self,
        class_idx: int,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> None:
        count = min(image_embeddings.size(0), self.memory_size)
        ptr = int(self.memory_ptr[class_idx].item())
        end = ptr + count
        if end <= self.memory_size:
            self.image_memory_pool[class_idx, ptr:end] = image_embeddings[:count]
            self.text_memory_pool[class_idx, ptr:end] = text_embeddings[:count]
        else:
            first = self.memory_size - ptr
            self.image_memory_pool[class_idx, ptr:] = image_embeddings[:first]
            self.text_memory_pool[class_idx, ptr:] = text_embeddings[:first]
            self.image_memory_pool[class_idx, : end % self.memory_size] = image_embeddings[first:count]
            self.text_memory_pool[class_idx, : end % self.memory_size] = text_embeddings[first:count]
        self.memory_ptr[class_idx] = end % self.memory_size
        self.memory_count[class_idx] = min(self.memory_count[class_idx] + count, self.memory_size)

    @torch.no_grad()
    def initialize_prototypes_from_features(
        self,
        image_features_by_class: dict[int, list[torch.Tensor]],
        text_features_by_class: dict[int, list[torch.Tensor]],
    ) -> None:
        if not self.use_prototypes:
            return

        device = self.image_prototypes.device
        for class_idx in range(self.num_classes):
            image_features = image_features_by_class.get(class_idx, [])
            text_features = text_features_by_class.get(class_idx, [])
            if not image_features or not text_features:
                continue
            image_stack = torch.stack(image_features).to(device)
            text_stack = torch.stack(text_features).to(device)
            image_centers = _prototype_centers(image_stack, self.prototypes_per_class)
            text_centers = _prototype_centers(text_stack, self.prototypes_per_class)
            self.image_prototypes[class_idx].copy_(F.normalize(image_centers, dim=-1))
            self.text_prototypes[class_idx].copy_(F.normalize(text_centers, dim=-1))


def _expand_to_count(features: torch.Tensor, target_count: int) -> torch.Tensor:
    if features.size(0) >= target_count:
        return F.normalize(features[:target_count], dim=-1)
    repeats = math.ceil(target_count / features.size(0))
    expanded = features.repeat(repeats, 1)[:target_count]
    expanded = expanded + torch.randn_like(expanded) * 0.01
    return F.normalize(expanded, dim=-1)


def _prototype_centers(features: torch.Tensor, target_count: int) -> torch.Tensor:
    if features.size(0) < target_count:
        return _expand_to_count(features, target_count)
    if target_count == 1:
        return features.mean(dim=0, keepdim=True)
    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=target_count, n_init=10)
        centers = kmeans.fit(features.detach().cpu().numpy()).cluster_centers_
        return torch.tensor(centers, device=features.device, dtype=features.dtype)
    except Exception:
        return _expand_to_count(features, target_count)


def _hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "hidden_size"):
        raise ValueError("Encoder config must expose hidden_size.")
    return int(config.hidden_size)


def build_cmpcn(config: TrainConfig):
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.text_encoder_name,
        local_files_only=config.local_files_only,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        config.image_encoder_name,
        local_files_only=config.local_files_only,
    )
    if hasattr(image_processor, "size"):
        image_processor.size = {"height": config.image_size, "width": config.image_size}
    if hasattr(image_processor, "crop_size"):
        image_processor.crop_size = {"height": config.image_size, "width": config.image_size}
    image_encoder = AutoModel.from_pretrained(
        config.image_encoder_name,
        local_files_only=config.local_files_only,
    )
    text_encoder = AutoModel.from_pretrained(
        config.text_encoder_name,
        local_files_only=config.local_files_only,
    )

    model = CMPCNModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        image_hidden_size=_hidden_size(image_encoder),
        text_hidden_size=_hidden_size(text_encoder),
        num_classes=config.num_classes,
        prototypes_per_class=config.prototypes_per_class,
        prototype_momentum=config.prototype_momentum,
        memory_size=config.memory_size,
        use_prototypes=config.use_prototypes,
        use_cross_attention=config.use_cross_attention,
        cross_attention_heads=config.cross_attention_heads,
        cross_attention_depth=config.cross_attention_depth,
    )
    if config.freeze_image_encoder:
        model.freeze_image_encoder()
    if config.freeze_text_encoder:
        model.freeze_text_encoder()
    return model, tokenizer, image_processor
