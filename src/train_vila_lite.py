from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from src.vila.datasets import AVACaptionPairDataset, collate_vila_examples
from src.vila.model_loader import DEFAULT_MODEL_NAME, load_vision_language_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaffold entry point for future VILA-lite fine-tuning on AVA captions.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv")
    parser.add_argument("--image_root")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--init_model", action="store_true")
    parser.add_argument("--output_dir")
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def _preview_batch(loader: DataLoader) -> dict[str, object]:
    batch = next(iter(loader))
    texts = batch["texts"]
    return {
        "batch_size": len(texts),
        "image_example": batch["image_paths"][0] if batch["image_paths"] else None,
        "text_example": texts[0][:160] if texts else None,
        "mos_example": None if batch["mos"].numel() == 0 else float(batch["mos"][0].item()),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    train_dataset = AVACaptionPairDataset.from_csv(args.train_csv, image_root=args.image_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_vila_examples,
    )

    val_summary = None
    if args.val_csv:
        val_dataset = AVACaptionPairDataset.from_csv(args.val_csv, image_root=args.image_root)
        val_summary = {
            "num_examples": len(val_dataset),
        }

    model_summary = None
    if args.init_model:
        model = load_vision_language_model(
            model_name=args.model_name,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        model_summary = {
            "backend": model.backend,
            "model_name": model.original_name,
            "device": str(model.device),
        }

    summary = {
        "train_csv": str(Path(args.train_csv)),
        "val_csv": str(Path(args.val_csv)) if args.val_csv else None,
        "num_train_examples": len(train_dataset),
        "train_preview": _preview_batch(train_loader),
        "val_summary": val_summary,
        "model": model_summary,
        "todo": [
            "Add contrastive fine-tuning over image/comment pairs.",
            "Optionally swap full fine-tuning for LoRA or adapter tuning.",
            "Add a validation loop that tracks prompt-pair accuracy and MOS correlation.",
        ],
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "vila_lite_scaffold_summary.json"
        summary_path.write_text(json.dumps(summary, indent=args.json_indent, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
