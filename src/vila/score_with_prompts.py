from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch

from src.vila.explain_selection import explain_prompt_scores
from src.vila.prompt_sets import DEFAULT_PROMPT_PRESET, PromptPair, get_prompt_preset, prompt_score_column


def load_image_rgb(image_path: str | Path) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


def _batched(items: Sequence[Path], batch_size: int) -> Sequence[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_value = math.exp(-value)
        return float(1.0 / (1.0 + exp_value))
    exp_value = math.exp(value)
    return float(exp_value / (1.0 + exp_value))


def _require_embedding_batch(
    embeddings: object,
    *,
    source: str,
    expected_rows: int | None = None,
) -> torch.Tensor:
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(f"{source} must return a torch.Tensor with shape [batch, dim], got {type(embeddings).__name__}.")
    if embeddings.ndim != 2:
        raise ValueError(f"{source} must return a 2D tensor with shape [batch, dim], got {tuple(embeddings.shape)}.")
    if expected_rows is not None and embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"{source} returned {embeddings.shape[0]} embeddings for {expected_rows} inputs."
        )
    return embeddings


class PromptBasedVILAScorer:
    def __init__(
        self,
        model,
        prompt_preset: str = DEFAULT_PROMPT_PRESET,
        prompt_pairs: Sequence[PromptPair] | None = None,
    ):
        self.model = model
        self.prompt_preset = prompt_preset
        self.prompt_pairs = tuple(prompt_pairs or get_prompt_preset(prompt_preset))
        self.logit_scale = float(model.similarity_scale())
        self._prompt_embeddings = self._encode_prompt_pairs()

    def _encode_prompt_pairs(self) -> dict[str, dict[str, object]]:
        texts = []
        for pair in self.prompt_pairs:
            texts.extend([pair.positive_prompt, pair.negative_prompt])

        embeddings = _require_embedding_batch(
            self.model.encode_texts(texts),
            source=f"{self.model.original_name} encode_texts()",
            expected_rows=len(texts),
        )
        encoded = {}
        for index, pair in enumerate(self.prompt_pairs):
            base_index = index * 2
            encoded[pair.key] = {
                "pair": pair,
                "positive_embedding": embeddings[base_index],
                "negative_embedding": embeddings[base_index + 1],
            }
        return encoded

    def _score_embedding(self, image_path: str | Path, image_embedding) -> dict[str, object]:
        per_prompt_scores = {}
        per_prompt_details = {}
        weighted_total = 0.0
        total_weight = 0.0

        for pair in self.prompt_pairs:
            entry = self._prompt_embeddings[pair.key]
            positive_similarity = float((image_embedding * entry["positive_embedding"]).sum().item())
            negative_similarity = float((image_embedding * entry["negative_embedding"]).sum().item())
            margin = self.logit_scale * (positive_similarity - negative_similarity)
            positive_probability = _sigmoid(margin)

            per_prompt_scores[pair.key] = positive_probability
            per_prompt_details[pair.key] = {
                "positive_prompt": pair.positive_prompt,
                "negative_prompt": pair.negative_prompt,
                "positive_similarity": self.logit_scale * positive_similarity,
                "negative_similarity": self.logit_scale * negative_similarity,
                "margin": margin,
                "positive_probability": positive_probability,
                "weight": pair.weight,
            }
            weighted_total += pair.weight * positive_probability
            total_weight += pair.weight

        vila_score = float(weighted_total / total_weight) if total_weight else None
        explanation = explain_prompt_scores(
            prompt_scores=per_prompt_scores,
            vila_score=vila_score,
            selected=None,
            prompt_preset=self.prompt_preset,
        )
        result = {
            "image_path": str(image_path),
            "vila_score": vila_score,
            "model_name": self.model.original_name,
            "model_backend": self.model.backend,
            "prompt_preset": self.prompt_preset,
            "per_prompt_scores": per_prompt_scores,
            "per_prompt_details": per_prompt_details,
            "explanation_signals": explanation["signals"],
            "vila_explanation": explanation["text"],
        }
        for pair in self.prompt_pairs:
            result[prompt_score_column(pair)] = per_prompt_scores.get(pair.key)
        return result

    def score_loaded_images(self, image_paths: Sequence[str | Path], images: Sequence[Image.Image]) -> list[dict[str, object]]:
        if len(image_paths) != len(images):
            raise ValueError("image_paths and images must have the same length.")
        image_embeddings = _require_embedding_batch(
            self.model.encode_images(images),
            source=f"{self.model.original_name} encode_images()",
            expected_rows=len(images),
        )
        return [
            self._score_embedding(image_path=image_path, image_embedding=image_embeddings[index])
            for index, image_path in enumerate(image_paths)
        ]

    def score_image(self, image_path: str | Path) -> dict[str, object]:
        image = load_image_rgb(image_path)
        return self.score_loaded_images([str(image_path)], [image])[0]

    def score_paths(self, image_paths: Sequence[str | Path], batch_size: int = 8) -> list[dict[str, object]]:
        paths = [Path(path) for path in image_paths]
        results: list[dict[str, object]] = []
        for batch_paths in _batched(paths, batch_size=max(1, batch_size)):
            images = [load_image_rgb(path) for path in batch_paths]
            results.extend(self.score_loaded_images(batch_paths, images))
        return results


def flatten_vila_result(result: dict[str, object], prompt_preset: str | None = None) -> dict[str, object]:
    resolved_preset = prompt_preset or str(result.get("prompt_preset") or DEFAULT_PROMPT_PRESET)
    row = {
        "image_path": result.get("image_path"),
        "vila_score": result.get("vila_score"),
        "model_name": result.get("model_name"),
        "model_backend": result.get("model_backend"),
        "prompt_preset": resolved_preset,
        "vila_explanation": result.get("vila_explanation"),
        "explanation_signals": json.dumps(result.get("explanation_signals"), ensure_ascii=False),
    }
    prompt_scores = result.get("per_prompt_scores", {})
    for pair in get_prompt_preset(resolved_preset):
        row[prompt_score_column(pair)] = prompt_scores.get(pair.key)
    return row
