from __future__ import annotations

import os
import json
import sys
from PIL import Image

from src.infer.composition_tags import extract_composition_tags

try:
    import google.generativeai as genai
except ImportError:
    genai = None


DEFAULT_GEMINI_MODEL_NAME = "models/gemini-2.5-flash-image"


def _parse_env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def generate_multimodal_explanations(
    rows: list[dict[str, object]],
    top_k: int,
    gemini_model_name: str | None = None,
    enable_gemini: bool | None = None,
) -> None:
    """
    Generates explanations for the top-k images using a multimodal LLM.
    """
    if enable_gemini is False:
        print("Gemini explanations disabled by explicit flag. Skipping multimodal explanations.", file=sys.stderr)
        return

    if enable_gemini is None:
        env_flag = _parse_env_flag("ENABLE_GEMINI")
        if env_flag is False:
            print("ENABLE_GEMINI=false. Skipping multimodal explanations.", file=sys.stderr)
            return

    raw_api_key = os.environ.get("GEMINI_API_KEY")
    api_key = (raw_api_key or "").strip()
    if not api_key:
        raw_google_api_key = os.environ.get("GOOGLE_API_KEY")
        google_api_key_present = bool((raw_google_api_key or "").strip())
        print(
            "GEMINI_API_KEY environment variable not set or empty. "
            f"(gemini_present={raw_api_key is not None}, google_api_key_present={google_api_key_present}) "
            "Skipping multimodal explanations.",
            file=sys.stderr,
        )
        return

    if genai is None:
        print("google-generativeai package is not installed. Skipping multimodal explanations.", file=sys.stderr)
        return

    resolved_model_name = (
        str(gemini_model_name).strip()
        if gemini_model_name is not None and str(gemini_model_name).strip()
        else str(os.environ.get("GEMINI_MODEL_NAME") or DEFAULT_GEMINI_MODEL_NAME)
    )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(resolved_model_name)

    for index, row in enumerate(rows):
        if index >= top_k:
            continue

        try:
            image_path = row.get("image_path")
            if not image_path or not os.path.exists(image_path):
                continue

            img = Image.open(image_path)

            explanation_structured = row.get("explanation_structured")
            if not isinstance(explanation_structured, dict):
                explanation_structured = row.get("acut_explanation_structured")
            if not isinstance(explanation_structured, dict):
                explanation_structured = None

            composition_tags = extract_composition_tags(row, explanation_structured)
            row["composition_tags"] = composition_tags
            
            prompt = f"""
            Analyze the following image for an automatic photo selection system (A-cut).
            Your task is to provide a concise and helpful explanation for why this image was or was not selected.
            You must respond with a JSON object with the following keys: "short_reason", "detailed_reason", "comparison_reason".

            Image Metadata:
            - Rank: {row.get('rank', 'N/A')}
            - Status: {'Selected' if row.get('selected') else 'Rejected'}
            - Aesthetic Score: {row.get('aesthetic_score', 'N/A')}
            - Technical Score: {row.get('technical_score', 'N/A')}
            - Composition Tags: {', '.join(composition_tags)}
            - Photo Type Mode: {row.get('photoTypeMode', 'N/A')}

            Based on the image and its metadata, provide the JSON response.
            - short_reason: A very brief, one-sentence summary.
            - detailed_reason: A more detailed explanation (2-3 sentences) covering strengths and weaknesses.
            - comparison_reason: A brief comparison to other photos (you can infer this based on its rank).
            """

            response = model.generate_content([prompt, img])
            
            # Clean the response text to extract the JSON part
            cleaned_response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            
            explanation = json.loads(cleaned_response_text)

            row["acut_short_reason"] = explanation.get("short_reason")
            row["acut_detailed_reason"] = explanation.get("detailed_reason")
            row["acut_comparison_reason"] = explanation.get("comparison_reason")

        except Exception as e:
            print(f"Error generating multimodal explanation for {row.get('image_path')}: {e}", file=sys.stderr)
            # On failure, do not overwrite the existing baseline explanations.
            pass
