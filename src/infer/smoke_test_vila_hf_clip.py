from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.vila.model_loader import DEFAULT_MODEL_NAME, load_vision_language_model
from src.vila.score_with_prompts import PromptBasedVILAScorer, load_image_rgb


def _default_image_path() -> Path:
    return Path(__file__).resolve().parents[2] / "test_samples" / "KakaoTalk_20260330_180646779.jpg"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Hugging Face CLIP path used by the VILA-lite scorer."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--image_path", default=str(_default_image_path()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prompts = [
        "a crisp, well-composed product photo",
        "a blurry, badly framed snapshot",
    ]
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Smoke-test image not found: {image_path}")

    model = load_vision_language_model(
        model_name=args.model_name,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    scorer = PromptBasedVILAScorer(model=model)

    text_embeddings = model.encode_texts(prompts)
    image = load_image_rgb(image_path)
    image_embeddings = model.encode_images([image])
    sample_score = scorer.score_loaded_images([image_path], [image])[0]

    summary = {
        "model_name": model.original_name,
        "model_backend": model.backend,
        "text_embeddings_shape": list(text_embeddings.shape),
        "image_embeddings_shape": list(image_embeddings.shape),
        "prompt_preset": scorer.prompt_preset,
        "num_prompt_pairs": len(scorer.prompt_pairs),
        "sample_image": str(image_path),
        "sample_vila_score": sample_score["vila_score"],
    }
    print(json.dumps(summary, indent=args.json_indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
