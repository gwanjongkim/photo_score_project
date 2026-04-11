from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch


DEFAULT_MODEL_NAME = "hf:openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    model_id: str
    pretrained: str | None
    original_name: str


class VisionLanguageBackend(Protocol):
    backend: str
    original_name: str
    device: torch.device

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        ...

    def encode_images(self, images) -> torch.Tensor:
        ...

    def similarity_scale(self) -> float:
        ...


def parse_model_name(model_name: str | None) -> ModelSpec:
    resolved = (model_name or DEFAULT_MODEL_NAME).strip()
    if resolved.startswith("hf:"):
        model_id = resolved.split(":", 1)[1].strip()
        if not model_id:
            raise ValueError("Expected Hugging Face model name after 'hf:'.")
        return ModelSpec(backend="hf", model_id=model_id, pretrained=None, original_name=resolved)
    if resolved.startswith("open_clip:"):
        parts = resolved.split(":")
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise ValueError(
                "OpenCLIP model names must use the form 'open_clip:<architecture>:<pretrained_tag>'."
            )
        return ModelSpec(
            backend="open_clip",
            model_id=parts[1],
            pretrained=parts[2],
            original_name=resolved,
        )
    return ModelSpec(backend="hf", model_id=resolved, pretrained=None, original_name=resolved)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


_DEFAULT_EMBEDDING_FIELDS = (
    "text_embeds",
    "image_embeds",
    "embeddings",
    "sentence_embedding",
    "pooler_output",
)


def _get_output_field(output: object, field_name: str) -> object | None:
    if isinstance(output, dict):
        return output.get(field_name)
    return getattr(output, field_name, None)


def _available_tensor_fields(output: object) -> list[str]:
    if isinstance(output, dict):
        items = output.items()
    elif hasattr(output, "keys"):
        keys = output.keys()
        items = ((key, _get_output_field(output, str(key))) for key in keys)
    else:
        return []
    return [str(key) for key, value in items if isinstance(value, torch.Tensor)]


def _extract_embedding_tensor(
    features: object,
    *,
    source: str,
    preferred_fields: Sequence[str] = (),
) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features

    candidate_fields = tuple(dict.fromkeys([*preferred_fields, *_DEFAULT_EMBEDDING_FIELDS]))
    for field_name in candidate_fields:
        value = _get_output_field(features, field_name)
        if isinstance(value, torch.Tensor):
            return value

    tuple_values = getattr(features, "to_tuple", None)
    if callable(tuple_values):
        tensor_items = [value for value in tuple_values() if isinstance(value, torch.Tensor)]
        if len(tensor_items) == 1:
            return tensor_items[0]

    tensor_fields = _available_tensor_fields(features)
    expected = ", ".join(candidate_fields)
    available = ", ".join(tensor_fields) if tensor_fields else "none"
    raise TypeError(
        f"{source} returned {type(features).__name__}, but no embedding tensor could be extracted. "
        f"Expected a torch.Tensor or one of the fields [{expected}]. "
        f"Available tensor fields: [{available}]."
    )


def _normalize_embeddings(
    features: object,
    *,
    source: str,
    preferred_fields: Sequence[str] = (),
) -> torch.Tensor:
    tensor = _extract_embedding_tensor(features, source=source, preferred_fields=preferred_fields)
    if tensor.ndim != 2:
        raise ValueError(
            f"{source} must return a 2D embedding tensor of shape [batch, dim]; got shape {tuple(tensor.shape)}."
        )
    return torch.nn.functional.normalize(tensor, dim=-1)


def _install_hint() -> str:
    return (
        "Install the PyTorch-side dependencies in a separate env, for example: "
        "pip install torch torchvision transformers open_clip_torch pillow pandas."
    )


class HuggingFaceCLIPBackend:
    backend = "hf"

    def __init__(self, spec: ModelSpec, device: torch.device, local_files_only: bool = False):
        self.original_name = spec.original_name
        self.device = device

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError(f"Missing transformers dependency. {_install_hint()}") from exc

        try:
            self.processor = AutoProcessor.from_pretrained(spec.model_id, local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained(spec.model_id, local_files_only=local_files_only)
        except OSError as exc:  # pragma: no cover - depends on local model cache/network
            mode = "local cache only" if local_files_only else "download or cache lookup"
            raise RuntimeError(
                f"Failed to load Hugging Face model '{spec.model_id}' via {mode}. "
                "If you are offline, pre-download the model or rerun with a locally cached model."
            ) from exc

        if not hasattr(self.model, "get_text_features") or not hasattr(self.model, "get_image_features"):
            raise TypeError(
                f"Model '{spec.model_id}' is not CLIP-compatible for prompt scoring because it lacks "
                "`get_text_features` and `get_image_features`."
            )

        self.model.to(self.device)
        self.model.eval()

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_text_features(**inputs)
        return _normalize_embeddings(
            features,
            source=f"{self.original_name} text encoder",
            preferred_fields=("text_embeds", "pooler_output"),
        )

    def encode_images(self, images) -> torch.Tensor:
        with torch.inference_mode():
            inputs = self.processor(images=list(images), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_image_features(**inputs)
        return _normalize_embeddings(
            features,
            source=f"{self.original_name} image encoder",
            preferred_fields=("image_embeds", "pooler_output"),
        )

    def similarity_scale(self) -> float:
        scale = getattr(self.model, "logit_scale", None)
        if scale is None:
            return 1.0
        return float(scale.exp().detach().cpu().item())


class OpenCLIPBackend:
    backend = "open_clip"

    def __init__(self, spec: ModelSpec, device: torch.device):
        self.original_name = spec.original_name
        self.device = device

        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError(f"Missing open_clip_torch dependency. {_install_hint()}") from exc

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                spec.model_id,
                pretrained=spec.pretrained,
                device=device,
            )
        except Exception as exc:  # pragma: no cover - depends on local model cache/network
            raise RuntimeError(
                f"Failed to load OpenCLIP model '{spec.original_name}'. "
                "If the weights are not already cached, load them once with network access or use a local path."
            ) from exc

        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(spec.model_id)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            tokens = self.tokenizer(texts).to(self.device)
            features = self.model.encode_text(tokens)
        return _normalize_embeddings(features, source=f"{self.original_name} text encoder")

    def encode_images(self, images) -> torch.Tensor:
        with torch.inference_mode():
            image_batch = torch.stack([self.preprocess(image) for image in images], dim=0).to(self.device)
            features = self.model.encode_image(image_batch)
        return _normalize_embeddings(features, source=f"{self.original_name} image encoder")

    def similarity_scale(self) -> float:
        scale = getattr(self.model, "logit_scale", None)
        if scale is None:
            return 1.0
        return float(scale.exp().detach().cpu().item())


def load_vision_language_model(
    model_name: str | None = None,
    device: str = "auto",
    local_files_only: bool = False,
) -> VisionLanguageBackend:
    spec = parse_model_name(model_name)
    resolved_device = resolve_device(device)
    if spec.backend == "hf":
        return HuggingFaceCLIPBackend(spec, device=resolved_device, local_files_only=local_files_only)
    if spec.backend == "open_clip":
        return OpenCLIPBackend(spec, device=resolved_device)
    raise ValueError(f"Unsupported backend '{spec.backend}'.")
