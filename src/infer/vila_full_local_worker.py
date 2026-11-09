from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback

DEFAULT_HF_HOME = "/tmp/photo_score_project_hf_cache"
DEFAULT_REMOTE_CACHE_ROOT = "/tmp/photo_score_project_vila_remote"


def _bootstrap_hf_cache_env() -> str:
    hf_home = str(os.environ.get("HF_HOME") or DEFAULT_HF_HOME)
    transformers_cache = str(Path(hf_home) / "transformers")
    hub_cache = str(Path(hf_home) / "hub")
    modules_cache = str(Path(hf_home) / "modules")
    for cache_dir in (hf_home, transformers_cache, hub_cache, modules_cache):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HF_MODULES_CACHE"] = modules_cache
    return hf_home


_bootstrap_hf_cache_env()

from PIL import Image
from transformers import StoppingCriteria

REASON_FIELDS = ("short_reason", "detailed_reason", "comparison_reason")
STRUCTURED_FIELDS = REASON_FIELDS + ("tags", "confidence")
REQUIRED_JSON_KEYS = STRUCTURED_FIELDS
# The local VILA prompt now asks for a full five-key JSON object with a compact
# score/rank explanation. Give generation enough room to finish the object
# before the stop criterion sees all required keys.
MAX_STRUCTURED_OUTPUT_TOKENS = 160
DEFAULT_NVILA_LOW_MEMORY_GPU_GB = 16.0
ALLOWED_TAGS = {
    "composition",
    "clarity",
    "subject_separation",
    "lighting",
    "background",
    "distraction",
    "blur",
    "exposure",
}
VILA_VERSION_CONV_MODE_MAP = {
    "vila1.5-3b": "vicuna_v1",
    "vila1.5-13b": "vicuna_v1",
    "vila1.5-8b": "llama_3",
    "llama-3-vila1.5-8b": "llama_3",
    "vila1.5-40b": "hermes-2",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the official local VILA explanation model over an input JSON batch.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser


def _ensure_hf_cache_env() -> str:
    return _bootstrap_hf_cache_env()


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower()).strip("_")
    return cleaned or "vila_model"


def _infer_vila_version(model_path: str) -> str | None:
    lower = model_path.lower()
    if "vila1.5" in lower:
        return "vila1.5"
    if "vila-m3" in lower:
        return "vila-m3"
    if "nvila" in lower:
        return "nvila"
    if "longvila" in lower:
        return "longvila"
    if "vila-u" in lower:
        return "vila-u"
    return None


def _infer_conv_mode(model_path: str) -> str | None:
    lower = Path(model_path).name.lower()
    for token, conv_mode in VILA_VERSION_CONV_MODE_MAP.items():
        if token in lower:
            return conv_mode
    inferred_version = _infer_vila_version(model_path)
    if inferred_version == "vila1.5":
        return "vicuna_v1"
    return None


def _converted_model_dir(model_path: str) -> Path:
    model_dir = Path(model_path).resolve()
    slug = _sanitize_token(model_dir.name)
    digest = hashlib.sha1(str(model_dir).encode("utf-8")).hexdigest()[:10]
    base_dir = Path(os.environ.get("ACUT_VILA_REMOTE_CACHE_DIR") or DEFAULT_REMOTE_CACHE_ROOT)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{slug}_{digest}"


def _converted_dir_ready(converted_dir: Path, conv_mode: str | None) -> bool:
    config_path = converted_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    auto_map = config.get("auto_map") or {}
    if config.get("model_type") != "vila":
        return False
    if "AutoModelForCausalLM" not in auto_map:
        return False
    if conv_mode:
        template_path = converted_dir / f"{conv_mode}.jinja"
        if not template_path.exists():
            return False
    return True


def _patch_converted_remote_code_for_attn(converted_dir: str, attn_implementation: str | None) -> None:
    def replace_once(path: Path, target: str, replacement: str) -> None:
        if not path.exists():
            return
        original_text = path.read_text(encoding="utf-8")
        if target not in original_text:
            return
        updated_text = original_text.replace(target, replacement)
        if updated_text != original_text:
            path.write_text(updated_text, encoding="utf-8")

    if attn_implementation is not None:
        replace_once(
            Path(converted_dir) / "siglip_encoder.py",
            '            attn_implementation="flash_attention_2",\n',
            "            attn_implementation=config._attn_implementation,\n",
        )
        replace_once(
            Path(converted_dir) / "builder.py",
            "    llm_cfg._attn_implementation = attn_implementation\n",
            '    llm_cfg._attn_implementation = attn_implementation or getattr(config, "_attn_implementation", None)\n',
        )
    replace_once(
        Path(converted_dir) / "modeling_vila.py",
        "        self.llm.resize_token_embeddings(len(self.tokenizer))\n",
        "        self.llm.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)\n",
    )
    replace_once(
        Path(converted_dir) / "modeling_vila.py",
        '        # TODO: need to check dtype with jason\n        self.llm = self.llm.to(torch.float16)\n        self.mm_projector = self.mm_projector.to(torch.float16)\n        self.vision_tower = self.vision_tower.to(torch.float16)\n',
        '        # TODO: need to check dtype with jason\n        target_dtype = eval(self.config.model_dtype)\n        if not hasattr(self.llm, "hf_device_map"):\n            self.llm = self.llm.to(target_dtype)\n        self.mm_projector = self.mm_projector.to(target_dtype)\n        self.vision_tower = self.vision_tower.to(target_dtype)\n',
    )
    replace_once(
        Path(converted_dir) / "modeling_vila.py",
        '        # TODO: need to check dtype with jason\n        if not hasattr(self.llm, "hf_device_map"):\n            if not hasattr(self.llm, "hf_device_map"):\n            self.llm = self.llm.to(torch.float16)\n        self.mm_projector = self.mm_projector.to(torch.float16)\n        self.vision_tower = self.vision_tower.to(torch.float16)\n',
        '        # TODO: need to check dtype with jason\n        target_dtype = eval(self.config.model_dtype)\n        if not hasattr(self.llm, "hf_device_map"):\n            self.llm = self.llm.to(target_dtype)\n        self.mm_projector = self.mm_projector.to(target_dtype)\n        self.vision_tower = self.vision_tower.to(target_dtype)\n',
    )
    replace_once(
        Path(converted_dir) / "modeling_vila.py",
        '        # TODO: need to check dtype with jason\n        if not hasattr(self.llm, "hf_device_map"):\n            self.llm = self.llm.to(torch.float16)\n        self.mm_projector = self.mm_projector.to(torch.float16)\n        self.vision_tower = self.vision_tower.to(torch.float16)\n',
        '        # TODO: need to check dtype with jason\n        target_dtype = eval(self.config.model_dtype)\n        if not hasattr(self.llm, "hf_device_map"):\n            self.llm = self.llm.to(target_dtype)\n        self.mm_projector = self.mm_projector.to(target_dtype)\n        self.vision_tower = self.vision_tower.to(target_dtype)\n',
    )
    replace_once(
        Path(converted_dir) / "modeling_vila.py",
        "        import xgrammar as xgr\n",
        "        xgr = None\n",
    )
    replace_once(
        Path(converted_dir) / "base_projector.py",
        "class MultimodalProjector(PreTrainedModel):\n    config_class = MultimodalProjectorConfig\n",
        "class MultimodalProjector(PreTrainedModel):\n    config_class = MultimodalProjectorConfig\n    all_tied_weights_keys = {}\n",
    )
    replace_once(
        Path(converted_dir) / "base_projector.py",
        "    all_tied_weights_keys = []\n",
        "    all_tied_weights_keys = {}\n",
    )
    replace_once(
        Path(converted_dir) / "modeling_vila.py",
        "class VILAForCausalLM(VILAPretrainedModel):\n    def __init__(self, config: VILAConfig, *args, **kwargs):\n        super().__init__(config, *args, **kwargs)\n\n    def merge_features_for_dynamic_s2(self, image_features, block_sizes):\n",
        """class VILAForCausalLM(VILAPretrainedModel):
    def __init__(self, config: VILAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    @staticmethod
    def merge_chessboard(x, num_split_h, num_split_w):
        \"\"\"
        x: b * n * c or b * h * w * c
        out: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        \"\"\"
        B = x.shape[0]
        if x.dim() == 3:
            N = x.shape[1]
            x = rearrange(x, \"b (h w) c -> b c h w\", h=int(N**0.5), w=int(N**0.5))

        assert B % (num_split_h * num_split_w) == 0
        b = B // (num_split_h * num_split_w)

        x_merge = torch.cat(
            [
                torch.cat(
                    [x[(i * num_split_w + j) * b : (i * num_split_w + j + 1) * b] for j in range(num_split_w)], dim=-1
                )
                for i in range(num_split_h)
            ],
            dim=-2,
        )

        return x_merge

    @staticmethod
    def split_chessboard(x, num_split_h, num_split_w):
        \"\"\"
        x: b * c * h * w
        out: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
        \"\"\"
        B, C, H, W = x.shape
        assert H % num_split_h == 0 and W % num_split_w == 0
        h, w = H // num_split_h, W // num_split_w
        x_split = torch.cat(
            [x[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w] for i in range(num_split_h) for j in range(num_split_w)],
            dim=0,
        )
        return x_split

    def merge_features_for_dynamic_s2(self, image_features, block_sizes):
""",
    )


def _clear_cached_remote_modules(converted_dir: str) -> None:
    modules_root = Path(os.environ.get("HF_MODULES_CACHE") or Path(DEFAULT_HF_HOME) / "modules")
    model_cache_dir = modules_root / "transformers_modules" / Path(converted_dir).name
    if model_cache_dir.exists():
        shutil.rmtree(model_cache_dir)


def _prepare_official_remote_checkpoint(model_path: str) -> tuple[str, str | None]:
    # llava's VILA remote-code package still imports no_init_weights from
    # transformers.modeling_utils. Newer Transformers exposes it from
    # transformers.initialization instead.
    try:
        import transformers.modeling_utils as modeling_utils

        if not hasattr(modeling_utils, "no_init_weights"):
            from transformers.initialization import no_init_weights

            modeling_utils.no_init_weights = no_init_weights
    except Exception:
        pass
    try:
        import transformers.image_utils as image_utils

        if not hasattr(image_utils, "VideoInput") and hasattr(image_utils, "ImageInput"):
            image_utils.VideoInput = image_utils.ImageInput
    except Exception:
        pass

    from llava.remote_code.modeling_vila import VILAForCausalLM

    converted_dir = _converted_model_dir(model_path)
    conv_mode = os.environ.get("ACUT_VILA_CONV_MODE") or _infer_conv_mode(model_path)
    vila_version = os.environ.get("ACUT_VILA_VERSION") or _infer_vila_version(model_path)

    if not _converted_dir_ready(converted_dir, conv_mode):
        if converted_dir.exists():
            shutil.rmtree(converted_dir)
        convert_kwargs = {
            "model_path": model_path,
            "output_dir": str(converted_dir),
            "copy": False,
        }
        if vila_version:
            convert_kwargs["vila_version"] = vila_version
        if conv_mode:
            convert_kwargs["conv_mode"] = conv_mode
        VILAForCausalLM.convert_vila_dev_ckpt_to_remote(**convert_kwargs)

    return str(converted_dir), conv_mode


def _resolve_requested_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_attn_implementation(model_path: str, requested_device: str) -> str | None:
    explicit = _clean_text(os.environ.get("ACUT_VILA_ATTN_IMPLEMENTATION"))
    if explicit is not None:
        return explicit
    if "nvila" not in model_path.lower():
        return None
    if requested_device == "cpu":
        return "eager"
    if _flash_attn_available():
        return None
    return "eager"


def _nvila_low_memory_mode_enabled(requested_device: str) -> bool:
    if requested_device != "cuda":
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    threshold_raw = _clean_text(os.environ.get("ACUT_NVILA_LOW_MEMORY_GPU_GB"))
    try:
        threshold_gb = float(threshold_raw) if threshold_raw is not None else DEFAULT_NVILA_LOW_MEMORY_GPU_GB
    except ValueError:
        threshold_gb = DEFAULT_NVILA_LOW_MEMORY_GPU_GB

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return total_gb < threshold_gb


def _apply_nvila_runtime_overrides(model, model_path: str, requested_device: str) -> None:
    if "nvila" not in model_path.lower():
        return
    explicit_aspect_ratio = _clean_text(os.environ.get("ACUT_VILA_IMAGE_ASPECT_RATIO"))
    if explicit_aspect_ratio is not None:
        model.config.image_aspect_ratio = explicit_aspect_ratio
        model.config.dynamic_s2 = explicit_aspect_ratio == "dynamic_s2"

    if explicit_aspect_ratio is not None or not _nvila_low_memory_mode_enabled(requested_device):
        return

    base_image_size = None
    image_processor = getattr(model.vision_tower, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, "size"):
        base_image_size = image_processor.size.get("height") or image_processor.size.get("shortest_edge")
    if base_image_size is None:
        base_image_size = 448
    compact_scales = f"{int(base_image_size)},{int(base_image_size) * 2},{int(base_image_size) * 2}"

    model.config.image_aspect_ratio = "dynamic_s2"
    model.config.dynamic_s2 = True
    model.config.s2_scales = compact_scales
    model.config.min_tiles = 1
    model.config.max_tiles = 4
    if hasattr(model.config, "video_max_tiles"):
        model.config.video_max_tiles = 4


def _repair_offloaded_resized_lm_head(model) -> None:
    import torch

    llm = getattr(model, "llm", None)
    tokenizer = getattr(model, "tokenizer", None)
    if llm is None or tokenizer is None:
        return

    target_vocab_size = len(tokenizer)
    output_embeddings = llm.get_output_embeddings()
    hook = getattr(output_embeddings, "_hf_hook", None)
    weights_map = getattr(hook, "weights_map", None) if hook is not None else None
    dataset = getattr(weights_map, "dataset", None) if weights_map is not None else None
    state_dict = getattr(dataset, "state_dict", None) if dataset is not None else None
    if not isinstance(state_dict, dict):
        return

    weight_key = None
    try:
        for key in weights_map.keys():
            if str(key).endswith(".weight"):
                weight_key = str(key)
                break
    except Exception:
        weight_key = None
    if weight_key is None or weight_key not in state_dict:
        return

    stored_weight = state_dict[weight_key]
    if not isinstance(stored_weight, torch.Tensor):
        return
    if stored_weight.shape[0] >= target_vocab_size:
        return

    extra_rows = target_vocab_size - int(stored_weight.shape[0])
    pad_rows = stored_weight[-1:].clone().repeat(extra_rows, 1)
    state_dict[weight_key] = torch.cat([stored_weight, pad_rows], dim=0)


def _primary_device(model) -> object:
    llm = getattr(model, "llm", None)
    if llm is not None and hasattr(llm, "device"):
        return llm.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cpu")


def _runtime_model_dtype(model):
    import torch

    primary_device = _primary_device(model)
    if getattr(primary_device, "type", None) == "cpu":
        return torch.float32

    model_dtype = getattr(model.config, "model_dtype", None)
    if isinstance(model_dtype, str):
        try:
            return eval(model_dtype, {"torch": torch})
        except Exception:
            pass
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        pass
    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is not None:
        try:
            return next(vision_tower.parameters()).dtype
        except StopIteration:
            pass
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _load_model(model_path: str, device: str):
    from transformers import AutoModelForCausalLM
    import torch

    converted_dir, conv_mode = _prepare_official_remote_checkpoint(model_path)
    requested_device = _resolve_requested_device(device)
    device_map = "cpu" if requested_device == "cpu" else "auto"
    attn_implementation = _resolve_attn_implementation(model_path, requested_device)
    _patch_converted_remote_code_for_attn(converted_dir, attn_implementation)
    _clear_cached_remote_modules(converted_dir)
    model = AutoModelForCausalLM.from_pretrained(
        converted_dir,
        trust_remote_code=True,
        device_map=device_map,
        local_files_only=True,
        attn_implementation=attn_implementation,
    )
    if requested_device == "cpu":
        model = model.to(dtype=torch.float32)
        try:
            model.config.model_dtype = "torch.float32"
        except Exception:
            pass
    _apply_nvila_runtime_overrides(model, model_path, requested_device)
    _repair_offloaded_resized_lm_head(model)
    return model, converted_dir, requested_device, conv_mode


def _module_import(model, suffix: str):
    module_base = model.__class__.__module__.rsplit(".", 1)[0]
    return importlib.import_module(f"{module_base}.{suffix}")


def _process_image_media(model, media: dict[str, object], media_config: dict[str, dict[str, object]]) -> dict[str, object]:
    mm_utils = _module_import(model, "mm_utils")
    config = model.config
    target_dtype = _runtime_model_dtype(model)

    if "image" not in media:
        return media

    if len(media["image"]) == 1 and config.image_aspect_ratio in {"dynamic", "dynamic_s2"}:
        config.image_processor = model.vision_tower.image_processor
        if getattr(config.image_processor, "crop_size", None) is None and hasattr(config.image_processor, "size"):
            config.image_processor.crop_size = config.image_processor.size
        if config.image_aspect_ratio == "dynamic":
            images = mm_utils.process_image(media["image"][0], config, None, enable_dynamic_res=True).to(dtype=target_dtype)
        else:
            if isinstance(config.s2_scales, str):
                config.s2_scales = list(map(int, config.s2_scales.split(",")))
            images, block_sizes = mm_utils.process_image(media["image"][0], config, None, enable_dynamic_s2=True)
            images = images.to(dtype=target_dtype)
            media_config["image"]["block_sizes"] = [block_sizes]
    else:
        if getattr(model.vision_tower.image_processor, "crop_size", None) is None and hasattr(
            model.vision_tower.image_processor,
            "size",
        ):
            model.vision_tower.image_processor.crop_size = model.vision_tower.image_processor.size
        images = mm_utils.process_images(media["image"], model.vision_tower.image_processor, config).to(dtype=target_dtype)

    media["image"] = [image for image in images]
    return media


def _normalize_generated_json_candidate(text: str) -> str:
    normalized = text.replace("\r\n", "\n").strip()
    normalized = re.sub(r"^\s*```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*```\s*$", "", normalized)
    normalized = re.sub(r"^\s*(assistant|response|output)\s*:\s*", "", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def _first_complete_json_object_text(text: str) -> str | None:
    normalized = _normalize_generated_json_candidate(text)
    start = normalized.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaping = False
    for index, char in enumerate(normalized[start:], start=start):
        if escaping:
            escaping = False
            continue
        if in_string:
            if char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                candidate = normalized[start : index + 1].strip()
                if all(f'"{key}"' in candidate for key in REQUIRED_JSON_KEYS):
                    return candidate
                return None
    return None


def _first_complete_three_line_text(text: str) -> str | None:
    normalized = _normalize_generated_json_candidate(text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    for index in range(max(0, len(lines) - 2)):
        candidate = lines[index : index + 3]
        if len(candidate) != 3:
            continue
        if not candidate[0].startswith("Strength:"):
            continue
        if not candidate[1].startswith("Weakness:"):
            continue
        if not candidate[2].startswith("Verdict:"):
            continue
        return "\n".join(candidate)
    return None


class _JsonObjectStopCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_length :]
        if getattr(generated_ids, "numel", lambda: 0)() <= 0:
            return False
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return _first_complete_json_object_text(generated_text) is not None or _first_complete_three_line_text(generated_text) is not None


def _generate_response_text(model, image_path: str, prompt: str, max_new_tokens: int) -> str:
    media_mod = _module_import(model, "media")
    tokenizer_utils = _module_import(model, "tokenizer_utils")
    from transformers import GenerationConfig, StoppingCriteriaList

    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")

    conversation = [{"from": "human", "value": [image_rgb, prompt]}]
    media = media_mod.extract_media(conversation, model.config)
    media_config: dict[str, dict[str, object]] = defaultdict(dict)
    media = _process_image_media(model, media, media_config)

    input_ids = tokenizer_utils.tokenize_conversation(
        conversation,
        model.tokenizer,
        add_generation_prompt=True,
    ).unsqueeze(0)
    input_ids = input_ids.to(_primary_device(model))
    prompt_length = int(input_ids.shape[1])
    effective_max_new_tokens = min(max(1, int(max_new_tokens)), MAX_STRUCTURED_OUTPUT_TOKENS)
    stopping_criteria = StoppingCriteriaList([_JsonObjectStopCriteria(model.tokenizer, prompt_length)])

    output_ids = model.generate(
        input_ids=input_ids,
        media=media,
        media_config=media_config,
        generation_config=GenerationConfig(
            max_new_tokens=effective_max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,
        ),
        stopping_criteria=stopping_criteria,
        return_output_ids_only=True,
    )
    return model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _normalize_response_text(response_text: str) -> str:
    return _normalize_generated_json_candidate(response_text)


def _extract_embedded_json_object(response_text: str) -> dict[str, object] | None:
    candidate = _first_complete_json_object_text(response_text)
    if candidate is not None:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload

    decoder = json.JSONDecoder()
    for index, char in enumerate(response_text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_adjacent_sentences(text: str) -> str:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    if not sentences:
        return text

    deduped: list[str] = []
    previous_normalized = None
    for sentence in sentences:
        normalized = sentence.rstrip(".!?").strip().lower()
        if normalized and normalized == previous_normalized:
            continue
        deduped.append(sentence)
        previous_normalized = normalized or previous_normalized
    return " ".join(deduped).strip() or text


def _limit_sentences(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return ""
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    if not sentences:
        return text
    return " ".join(sentences[:max_sentences]).strip()


def _trim_incomplete_tail(text: str) -> str:
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    last_punctuation = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_punctuation >= 0:
        candidate = text[: last_punctuation + 1].strip()
        if candidate:
            return candidate
    return text


def _sanitize_reason_value(field_name: str, value: object) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None

    normalized = text.strip()
    while True:
        updated = re.sub(
            rf'^\s*"?{re.escape(field_name)}"?\s*:\s*',
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        updated = updated.strip().strip(",")
        updated = updated.strip()
        if updated == normalized:
            break
        normalized = updated

    normalized = normalized.strip().strip(",")
    normalized = normalized.rstrip("}")
    normalized = normalized.strip().strip('"').strip()
    normalized = normalized.replace("\\n", " ").replace("\\t", " ")
    if normalized.lower() in {"null", "none", "n/a", "...", "…"}:
        return None
    cleaned = _collapse_whitespace(normalized)
    cleaned = cleaned.strip(" ;,")
    if not cleaned:
        return None
    if cleaned.startswith("{"):
        return None
    if cleaned.startswith("["):
        if cleaned.endswith("]"):
            inner = cleaned[1:-1].strip()
            if not inner:
                return None
            if inner.startswith("{") or inner.startswith("["):
                return None
            if re.search(
                r'"?(short_reason|detailed_reason|comparison_reason|detail_reason|short_word|detail_word|tags|confidence|long_reason)"?\s*:',
                inner,
                flags=re.IGNORECASE,
            ):
                return None
            cleaned = _collapse_whitespace(inner).strip(" ;,")
            if not cleaned:
                return None
        else:
            return None
    if re.search(
        r'"?(short_reason|detailed_reason|comparison_reason|detail_reason|short_word|detail_word)"?\s*:',
        cleaned,
        flags=re.IGNORECASE,
    ):
        return None
    cleaned = _dedupe_adjacent_sentences(cleaned)
    if field_name == "detailed_reason":
        cleaned = _limit_sentences(cleaned, max_sentences=2)
        cleaned = _trim_incomplete_tail(cleaned)
    elif field_name == "comparison_reason":
        cleaned = _limit_sentences(cleaned, max_sentences=1)
        cleaned = _trim_incomplete_tail(cleaned)
    cleaned = cleaned.strip(" ;,")
    return cleaned or None


def _parse_tags_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = _clean_text(value)
        if text is None:
            return []
        text = re.sub(r'^\s*"?tags"?\s*:\s*', "", text, flags=re.IGNORECASE)
        text = text.strip().rstrip(",}")
        if not text or text.lower() in {"null", "none", "[]"}:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                raw_items = parsed
            else:
                raw_items = re.split(r"[,;\n]+", text.strip("[]"))
        else:
            raw_items = re.split(r"[,;\n]+", text)

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        item_text = _clean_text(item)
        if item_text is None:
            continue
        item_text = item_text.strip('"').strip("'").strip().lower()
        if item_text in {"...", "…"}:
            continue
        if not item_text:
            continue
        if item_text not in ALLOWED_TAGS:
            continue
        if item_text in seen:
            continue
        seen.add(item_text)
        cleaned.append(item_text)
    return cleaned[:4]


def _parse_confidence_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = _clean_text(value)
        if text is None:
            return None
        text = re.sub(r'^\s*"?confidence"?\s*:\s*', "", text, flags=re.IGNORECASE)
        text = text.strip()
        text = text.splitlines()[0].strip()
        text = text.split(",", 1)[0].strip()
        text = text.rstrip("}")
        if text.lower() in {"null", "none", "n/a"}:
            return None
        fraction_match = re.match(r"([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)", text)
        if fraction_match is not None:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            if denominator <= 0:
                return None
            numeric = numerator / denominator
        else:
            match = re.match(r"[0-9]*\.?[0-9]+", text)
            if match is None:
                return None
            numeric = float(match.group(0))

    if numeric != numeric:
        return None
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return round(numeric, 4)


def _finalize_payload(payload: dict[str, object]) -> dict[str, object]:
    finalized = {
        "short_reason": _sanitize_reason_value("short_reason", payload.get("short_reason")),
        "detailed_reason": _sanitize_reason_value("detailed_reason", payload.get("detailed_reason")),
        "comparison_reason": _sanitize_reason_value("comparison_reason", payload.get("comparison_reason")),
        "tags": _parse_tags_value(payload.get("tags")),
        "confidence": _parse_confidence_value(payload.get("confidence")),
    }
    return finalized


def _payload_has_content(payload: dict[str, object]) -> bool:
    return any(payload.get(field_name) for field_name in REASON_FIELDS) or bool(payload.get("tags")) or payload.get("confidence") is not None


def _extract_field_blocks(response_text: str) -> dict[str, object] | None:
    pattern = re.compile(r'"?(short_reason|detailed_reason|comparison_reason|tags|confidence)"?\s*:\s*', flags=re.IGNORECASE)
    matches = list(pattern.finditer(response_text))
    if not matches:
        return None

    payload: dict[str, object] = {}
    prefix = response_text[: matches[0].start()].strip()
    if prefix:
        payload["short_reason"] = prefix
    for index, match in enumerate(matches):
        field_name = match.group(1).lower()
        if field_name in payload:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(response_text)
        value = response_text[start:end].strip()
        if value:
            payload[field_name] = value
    return payload or None


def _parse_response_payload(response_text: str) -> tuple[dict[str, object], str, str | None, str]:
    normalized_text = _normalize_response_text(response_text)
    if not normalized_text:
        raise ValueError("Local VILA output was empty.")

    try:
        payload = json.loads(normalized_text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        finalized = _finalize_payload(payload)
        if _payload_has_content(finalized):
            return finalized, "strict_json", None, normalized_text

    embedded_payload = _extract_embedded_json_object(normalized_text)
    if embedded_payload is not None:
        finalized = _finalize_payload(embedded_payload)
        if _payload_has_content(finalized):
            return finalized, "embedded_json", "Recovered JSON object from surrounding text.", normalized_text

    field_payload = _extract_field_blocks(normalized_text)
    if field_payload is not None:
        finalized = _finalize_payload(field_payload)
        if _payload_has_content(finalized):
            return finalized, "field_recovery", "Recovered labeled fields from non-JSON output.", normalized_text

    fallback_payload = _fallback_payload(normalized_text)
    if _payload_has_content(fallback_payload):
        return fallback_payload, "plain_text_fallback", "Recovered plain-text local output without full JSON.", normalized_text

    raise ValueError("Local VILA output did not contain parseable explanation fields.")


def _fallback_payload(response_text: str) -> dict[str, object]:
    cleaned = _clean_text(response_text) or ""
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", cleaned) if segment.strip()]
    short_reason = sentences[0] if sentences else None
    comparison_reason = None
    if len(sentences) > 1 and re.search(r"\b(compared|relative|versus|vs\.?|better|worse|rank)\b", sentences[1], flags=re.IGNORECASE):
        comparison_reason = sentences[1]
    return {
        "short_reason": _sanitize_reason_value("short_reason", short_reason),
        "detailed_reason": _collapse_whitespace(cleaned) or None,
        "comparison_reason": _sanitize_reason_value("comparison_reason", comparison_reason),
        "tags": [],
        "confidence": None,
    }


def main() -> None:
    args = build_parser().parse_args()
    _ensure_hf_cache_env()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Input JSON must contain a 'rows' list.")

    model, converted_dir, resolved_device, conv_mode = _load_model(
        model_path=args.model_path,
        device=args.device,
    )

    results = []
    for row in rows:
        image_path = row.get("image_path")
        prompt = row.get("prompt")
        if not isinstance(image_path, str) or not image_path.strip():
            results.append({"image_path": image_path, "error": "Missing image_path."})
            continue
        if not isinstance(prompt, str) or not prompt.strip():
            results.append({"image_path": image_path, "error": "Missing prompt."})
            continue

        try:
            response_text = _generate_response_text(
                model=model,
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )
            try:
                parsed, parse_mode, parse_warning, normalized_text = _parse_response_payload(response_text)
            except ValueError:
                parsed = _fallback_payload(response_text)
                parse_mode = "plain_text_fallback"
                parse_warning = "Recovered plain-text local output without full JSON."
                normalized_text = _normalize_response_text(response_text)
            results.append(
                {
                    "image_path": image_path,
                    "short_reason": parsed.get("short_reason"),
                    "detailed_reason": parsed.get("detailed_reason"),
                    "comparison_reason": parsed.get("comparison_reason"),
                    "tags": parsed.get("tags"),
                    "confidence": parsed.get("confidence"),
                    "parse_mode": parse_mode,
                    "parse_warning": parse_warning,
                    "raw_text": response_text,
                    "normalized_text": normalized_text,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "image_path": image_path,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=12),
                }
            )

    print(
        json.dumps(
            {
                "runtime": "local_vila_remote_code",
                "model_name": args.model_path,
                "loader": "llava.remote_code.AutoModelForCausalLM",
                "device": resolved_device,
                "converted_model_dir": converted_dir,
                "conv_mode": conv_mode,
                "results": results,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr, limit=12)
        print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
