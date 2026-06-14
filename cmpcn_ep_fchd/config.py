from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """Runtime configuration for EP_FCHD CMPCN training."""

    data_root: Path = Path("data/EP_FCHD")
    output_dir: Path = Path("outputs/ep_fchd_cmpcn")
    image_encoder_name: str = "google/vit-base-patch16-224-in21k"
    text_encoder_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    local_files_only: bool = False

    image_size: int = 224
    text_max_length: int = 128
    num_classes: int = 4
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    seed: int = 42
    device: str = "cuda"

    labeled_ratio: float = 1.0
    use_unlabel_folder: bool = False
    pseudo_label_threshold: float = 0.6
    gradient_accumulation_steps: int = 1

    prototypes_per_class: int = 5
    prototype_momentum: float = 0.9
    memory_size: int = 200
    use_prototypes: bool = True
    use_cross_attention: bool = True
    cross_attention_heads: int = 8
    cross_attention_depth: int = 2
    use_multi_layer_loss: bool = True
    freeze_image_encoder: bool = False
    freeze_text_encoder: bool = False

    class_loss_weight: float = 2.0
    contrastive_loss_weight: float = 0.1
    prototype_loss_weight: float = 0.1
    consistency_loss_weight: float = 0.1
    multi_layer_loss_weight: float = 0.2

    early_stopping_patience: int = 50
    save_every: int = 0

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("data_root", "output_dir"):
            values[key] = str(values[key])
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        cleaned = dict(values)
        if "data_root" in cleaned:
            cleaned["data_root"] = Path(cleaned["data_root"])
        if "output_dir" in cleaned:
            cleaned["output_dir"] = Path(cleaned["output_dir"])
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in cleaned.items() if key in allowed})

    @property
    def train_dir(self) -> Path:
        return self.data_root / "training"

    @property
    def test_dir(self) -> Path:
        return self.data_root / "testing"

    @property
    def unlabel_dir(self) -> Path:
        return self.data_root / "unlabel"


def add_common_args(parser):
    parser.add_argument("--data-root", type=Path, default=TrainConfig.data_root)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--image-encoder-name", default=TrainConfig.image_encoder_name)
    parser.add_argument("--text-encoder-name", default=TrainConfig.text_encoder_name)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--text-max-length", type=int, default=TrainConfig.text_max_length)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    return parser


def update_config_from_args(config: TrainConfig, args) -> TrainConfig:
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    return config
