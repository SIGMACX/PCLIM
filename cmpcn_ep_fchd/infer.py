from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .evaluate import load_model_from_checkpoint
from .text_descriptions import EP_FCHD_CLASS_NAMES
from .train import predict_candidate_logits


def parse_args():
    parser = argparse.ArgumentParser(description="Run CMPCN inference for EP_FCHD images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_device = args.device or "cuda"
    device = torch.device(requested_device if torch.cuda.is_available() and "cuda" in requested_device else "cpu")
    model, tokenizer, image_processor, config = load_model_from_checkpoint(args.checkpoint, device, args)

    results = []
    for image_path in args.images:
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        logits = predict_candidate_logits(model, image_tensor, tokenizer, config, device)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
        results.append(
            {
                "image": str(image_path),
                "prediction": EP_FCHD_CLASS_NAMES[pred_idx],
                "probabilities": {
                    class_name: float(probs[idx].item())
                    for idx, class_name in enumerate(EP_FCHD_CLASS_NAMES)
                },
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
