from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPair:
    key: str
    positive_prompt: str
    negative_prompt: str
    display_name: str
    positive_reason: str
    negative_reason: str
    weight: float = 1.0


DEFAULT_PROMPT_PRESET = "a_cut_basic"


PROMPT_PRESETS: dict[str, tuple[PromptPair, ...]] = {
    DEFAULT_PROMPT_PRESET: (
        PromptPair(
            key="good_image",
            positive_prompt="a good aesthetically pleasing photograph",
            negative_prompt="a bad aesthetically weak photograph",
            display_name="overall image quality",
            positive_reason="overall image appeal is strong",
            negative_reason="overall image appeal is weak",
        ),
        PromptPair(
            key="good_composition",
            positive_prompt="a photograph with strong composition",
            negative_prompt="a photograph with poor composition",
            display_name="composition",
            positive_reason="composition is strong",
            negative_reason="composition is weak",
        ),
        PromptPair(
            key="good_lighting",
            positive_prompt="a photograph with good lighting",
            negative_prompt="a photograph with poor lighting",
            display_name="lighting",
            positive_reason="lighting is strong",
            negative_reason="lighting is weak",
        ),
        PromptPair(
            key="clear_subject",
            positive_prompt="a photograph with a clear sharp subject",
            negative_prompt="a photograph with a blurry unclear subject",
            display_name="subject clarity",
            positive_reason="subject clarity is high",
            negative_reason="blur reduces subject clarity",
        ),
        PromptPair(
            key="clean_background",
            positive_prompt="a photograph with a clean non-distracting background",
            negative_prompt="a photograph with a distracting cluttered background",
            display_name="background cleanliness",
            positive_reason="background distraction is low",
            negative_reason="background distraction is high",
        ),
    ),
}


def available_prompt_presets() -> list[str]:
    return sorted(PROMPT_PRESETS)


def get_prompt_preset(name: str = DEFAULT_PROMPT_PRESET) -> tuple[PromptPair, ...]:
    if name not in PROMPT_PRESETS:
        raise ValueError(
            f"Unknown prompt preset '{name}'. Available presets: {', '.join(available_prompt_presets())}"
        )
    return PROMPT_PRESETS[name]


def prompt_score_column(prompt_pair: PromptPair | str) -> str:
    key = prompt_pair.key if isinstance(prompt_pair, PromptPair) else str(prompt_pair)
    return f"prompt_{key}"
