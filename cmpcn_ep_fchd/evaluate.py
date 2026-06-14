from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import TrainConfig
from .datasets import EPFCHDImageTextDataset
from .model import build_cmpcn
from .train import evaluate, make_loader


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device, overrides) -> tuple[object, object, object, TrainConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = TrainConfig.from_dict(checkpoint.get("config", {}))

    for key in (
        "data_root",
        "output_dir",
        "batch_size",
        "num_workers",
        "device",
        "local_files_only",
    ):
        value = getattr(overrides, key, None)
        if value is not None:
            setattr(config, key, value)

    model, tokenizer, image_processor = build_cmpcn(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, image_processor, config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CMPCN checkpoint on EP_FCHD testing split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_device = args.device or "cuda"
    device = torch.device(requested_device if torch.cuda.is_available() and "cuda" in requested_device else "cpu")
    model, tokenizer, image_processor, config = load_model_from_checkpoint(args.checkpoint, device, args)

    test_dataset = EPFCHDImageTextDataset(
        config.test_dir,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config.text_max_length,
    )
    test_loader = make_loader(test_dataset, config, shuffle=False)
    metrics = evaluate(model, test_loader, tokenizer, device, config)
    print(json.dumps(metrics, indent=2))

    output_dir = args.output_dir or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
