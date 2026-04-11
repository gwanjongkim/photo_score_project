#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AESTHETIC_CONFIG = REPO_ROOT / "configs" / "stage5_reference.json"
DEFAULT_PYTHON_MIN = (3, 10)
DEFAULT_PYTHON_MAX_EXCLUSIVE = (3, 13)
PIPELINE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CORE_MODEL_FIELDS = (
    "aadb_model",
    "koniq_model",
    "flive_image_model",
    "flive_patch_model",
    "nima_model",
    "alamp_model",
    "musiq_model",
    "rgnet_model",
)
PAIRWISE_MODEL_FIELDS = (
    "pairwise_model",
    "pairwise_reference_csv",
)


class CheckCollector:
    def __init__(self) -> None:
        self.ok: list[str] = []
        self.warn: list[str] = []
        self.fail: list[str] = []

    def add_ok(self, message: str) -> None:
        self.ok.append(message)

    def add_warn(self, message: str) -> None:
        self.warn.append(message)

    def add_fail(self, message: str) -> None:
        self.fail.append(message)


class PreflightError(RuntimeError):
    pass


def parse_bool(value: str | None, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def check_python_version(checks: CheckCollector) -> None:
    current = sys.version_info[:3]
    if current < DEFAULT_PYTHON_MIN or current >= DEFAULT_PYTHON_MAX_EXCLUSIVE:
        checks.add_fail(
            "Python version is unsupported for this repo. "
            f"Current={current[0]}.{current[1]}.{current[2]}, "
            "recommended range is >=3.10 and <3.13."
        )
        return
    checks.add_ok(f"Python version OK: {current[0]}.{current[1]}.{current[2]}")


def check_import(checks: CheckCollector, module_name: str, *, required: bool, hint: str | None = None) -> None:
    try:
        importlib.import_module(module_name)
        checks.add_ok(f"Import OK: {module_name}")
    except Exception as exc:
        message = f"Import failed: {module_name} ({exc.__class__.__name__}: {exc})"
        if hint:
            message = f"{message}. {hint}"
        if required:
            checks.add_fail(message)
        else:
            checks.add_warn(message)


def load_aesthetic_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PreflightError(f"Aesthetic config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"Aesthetic config is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise PreflightError(f"Aesthetic config root must be a JSON object: {path}")
    bundle_args = payload.get("bundle_args")
    if not isinstance(bundle_args, dict):
        raise PreflightError("Aesthetic config is missing object key 'bundle_args'.")
    return payload


def check_model_paths(
    checks: CheckCollector,
    *,
    aesthetic_config: Path,
    expect_pairwise: bool,
) -> None:
    try:
        config = load_aesthetic_config(aesthetic_config)
    except PreflightError as exc:
        checks.add_fail(str(exc))
        return

    bundle_args = dict(config.get("bundle_args", {}))
    checks.add_ok(f"Loaded aesthetic config: {aesthetic_config}")

    for field_name in CORE_MODEL_FIELDS:
        raw = bundle_args.get(field_name)
        if not isinstance(raw, str) or not raw.strip():
            checks.add_fail(f"Missing required bundle_args['{field_name}'] in {aesthetic_config}.")
            continue
        resolved = resolve_project_path(raw)
        assert resolved is not None
        if not resolved.exists():
            checks.add_fail(f"Model file missing for {field_name}: {resolved}")
            continue
        checks.add_ok(f"Model path OK: {field_name} -> {resolved}")

    for field_name in PAIRWISE_MODEL_FIELDS:
        raw = bundle_args.get(field_name)
        if not isinstance(raw, str) or not raw.strip():
            if expect_pairwise:
                checks.add_fail(f"Pairwise is required, but bundle_args['{field_name}'] is missing.")
            else:
                checks.add_warn(
                    f"Pairwise field not configured: {field_name}. "
                    "This is acceptable unless pairwise rerank is required."
                )
            continue
        resolved = resolve_project_path(raw)
        assert resolved is not None
        if not resolved.exists():
            message = f"Pairwise path missing for {field_name}: {resolved}"
            if expect_pairwise:
                checks.add_fail(message)
            else:
                checks.add_warn(message)
            continue
        checks.add_ok(f"Pairwise path OK: {field_name} -> {resolved}")


def check_pythonpath(checks: CheckCollector) -> None:
    pythonpath = os.environ.get("PYTHONPATH", "")
    if not pythonpath:
        checks.add_warn("PYTHONPATH is empty. Use 'export PYTHONPATH=.' when running module-style commands.")
        return
    entries = [Path(part).resolve() for part in pythonpath.split(":") if part.strip()]
    if REPO_ROOT.resolve() in entries:
        checks.add_ok(f"PYTHONPATH includes repo root: {REPO_ROOT}")
    else:
        checks.add_warn(
            "PYTHONPATH does not include the repo root. "
            "Commands like 'python -m src.infer.run_acut_pipeline' may fail if run outside repo root."
        )


def check_local_paths(checks: CheckCollector, *, local_input_dir: str | None, local_output_dir: str | None) -> None:
    if local_input_dir:
        input_dir = resolve_project_path(local_input_dir)
        assert input_dir is not None
        if not input_dir.exists() or not input_dir.is_dir():
            checks.add_fail(f"LOCAL_INPUT_DIR does not exist or is not a directory: {input_dir}")
        else:
            images = [
                path for path in input_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in PIPELINE_IMAGE_SUFFIXES
            ]
            if images:
                checks.add_ok(f"LOCAL_INPUT_DIR has {len(images)} image(s): {input_dir}")
            else:
                checks.add_warn(
                    f"LOCAL_INPUT_DIR exists but no supported images were found: {input_dir} "
                    f"(extensions={','.join(sorted(PIPELINE_IMAGE_SUFFIXES))})"
                )

    if local_output_dir:
        output_dir = resolve_project_path(local_output_dir)
        assert output_dir is not None
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=output_dir, prefix="preflight_", delete=True):
                pass
            checks.add_ok(f"LOCAL_OUTPUT_DIR is writable: {output_dir}")
        except Exception as exc:
            checks.add_fail(
                f"LOCAL_OUTPUT_DIR is not writable: {output_dir} "
                f"({exc.__class__.__name__}: {exc})"
            )


def check_gemini(checks: CheckCollector, *, strict: bool) -> None:
    enable_gemini_env = parse_bool(os.environ.get("ENABLE_GEMINI"), default=None)
    if strict and enable_gemini_env is False:
        checks.add_fail("ENABLE_GEMINI=false but Gemini mode was requested.")
        return
    if not strict and enable_gemini_env is False:
        checks.add_ok("ENABLE_GEMINI=false, Gemini checks skipped by configuration.")
        return

    raw_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not raw_api_key.strip():
        if strict:
            checks.add_fail("GEMINI_API_KEY is required for Gemini explanation mode.")
        else:
            checks.add_warn("GEMINI_API_KEY is not set. Multimodal Gemini explanations will be skipped.")
    else:
        checks.add_ok("GEMINI_API_KEY is set.")

    gemini_model_name = (os.environ.get("GEMINI_MODEL_NAME") or "").strip()
    if gemini_model_name:
        checks.add_ok(f"GEMINI_MODEL_NAME is set: {gemini_model_name}")
    else:
        checks.add_warn(
            "GEMINI_MODEL_NAME is not set. Default model 'models/gemini-2.5-flash-image' will be used."
        )


def check_worker_env(checks: CheckCollector) -> None:
    bucket = (os.environ.get("FIREBASE_STORAGE_BUCKET") or "").strip()
    if bucket:
        checks.add_ok(f"FIREBASE_STORAGE_BUCKET is set: {bucket}")
    else:
        checks.add_warn(
            "FIREBASE_STORAGE_BUCKET is not set. Provide --bucket at runtime or configure default bucket in Firebase app."
        )

    project_id = (
        os.environ.get("FIREBASE_PROJECT_ID")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or ""
    ).strip()
    if project_id:
        checks.add_ok(f"Firebase project id is set: {project_id}")
    else:
        checks.add_warn(
            "FIREBASE_PROJECT_ID / GCLOUD_PROJECT / GOOGLE_CLOUD_PROJECT is not set. "
            "Set one explicitly for predictable worker startup."
        )

    firestore_emulator = (os.environ.get("FIRESTORE_EMULATOR_HOST") or "").strip()
    storage_emulator = (os.environ.get("FIREBASE_STORAGE_EMULATOR_HOST") or "").strip()
    if firestore_emulator or storage_emulator:
        checks.add_ok(
            "Emulator mode detected via FIRESTORE_EMULATOR_HOST or FIREBASE_STORAGE_EMULATOR_HOST. "
            "Service account credentials are not required in this mode."
        )
        return

    credentials_path_raw = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if not credentials_path_raw:
        checks.add_warn(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. Worker can still run if ADC is already configured "
            "(for example via gcloud), but explicit service-account JSON is recommended for reproducibility."
        )
        return

    credentials_path = Path(credentials_path_raw)
    if not credentials_path.is_absolute():
        credentials_path = (REPO_ROOT / credentials_path).resolve()

    if not credentials_path.exists() or not credentials_path.is_file():
        checks.add_fail(
            "GOOGLE_APPLICATION_CREDENTIALS points to a missing file: "
            f"{credentials_path}"
        )
        return

    try:
        with credentials_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("credential JSON root is not an object")
    except Exception as exc:
        checks.add_fail(
            f"GOOGLE_APPLICATION_CREDENTIALS is not readable JSON: {credentials_path} "
            f"({exc.__class__.__name__}: {exc})"
        )
        return

    checks.add_ok(f"GOOGLE_APPLICATION_CREDENTIALS is readable: {credentials_path}")


def check_vila_dependencies(checks: CheckCollector, *, force: bool) -> None:
    model_name = (os.environ.get("ACUT_VILA_MODEL_NAME") or "").strip()
    if not force and not model_name:
        checks.add_ok("VILA model is not configured (ACUT_VILA_MODEL_NAME unset), optional VILA checks skipped.")
        return

    check_import(
        checks,
        "torch",
        required=True,
        hint="Install requirements-vila.txt if you need optional VILA scoring.",
    )

    if model_name.startswith("open_clip:"):
        check_import(
            checks,
            "open_clip",
            required=True,
            hint="Install requirements-vila.txt for open_clip backend support.",
        )
    else:
        check_import(
            checks,
            "transformers",
            required=True,
            hint="Install requirements-vila.txt for Hugging Face CLIP backend support.",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate runtime prerequisites for local scoring, Gemini explanations, and Firebase worker modes."
    )
    parser.add_argument(
        "--mode",
        choices=("minimal", "gemini", "worker"),
        default="minimal",
        help="minimal: local scoring only, gemini: scoring + Gemini checks, worker: Firebase worker checks.",
    )
    parser.add_argument(
        "--aesthetic_config",
        default=os.environ.get("ACUT_AESTHETIC_CONFIG", str(DEFAULT_AESTHETIC_CONFIG)),
        help="Path to stage5 config used to resolve model files.",
    )
    parser.add_argument(
        "--disable_aesthetic_scoring",
        action="store_true",
        default=parse_bool(os.environ.get("ACUT_DISABLE_AESTHETIC_SCORING"), default=False),
        help="Skip model file checks for the TensorFlow aesthetic bundle.",
    )
    parser.add_argument(
        "--expect_pairwise",
        action="store_true",
        help="Treat pairwise model/reference files as required.",
    )
    parser.add_argument(
        "--check_vila",
        action="store_true",
        help="Force optional VILA dependency checks even when ACUT_VILA_MODEL_NAME is unset.",
    )
    parser.add_argument(
        "--local_input_dir",
        default=os.environ.get("LOCAL_INPUT_DIR"),
        help="Optional local input dir to validate (for mode1/mode2 quick checks).",
    )
    parser.add_argument(
        "--local_output_dir",
        default=os.environ.get("LOCAL_OUTPUT_DIR"),
        help="Optional local output dir to validate write permission.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    return parser


def run_checks(args: argparse.Namespace) -> CheckCollector:
    checks = CheckCollector()

    check_python_version(checks)
    check_pythonpath(checks)

    # Core imports for score calculation path.
    check_import(
        checks,
        "numpy",
        required=True,
        hint="Install base deps with: pip install -r requirements.txt",
    )
    check_import(
        checks,
        "pandas",
        required=True,
        hint="Install base deps with: pip install -r requirements.txt",
    )
    check_import(
        checks,
        "PIL",
        required=True,
        hint="Install base deps with: pip install -r requirements.txt",
    )
    check_import(
        checks,
        "tensorflow",
        required=True,
        hint="Install base deps with: pip install -r requirements.txt",
    )

    if not args.disable_aesthetic_scoring:
        aesthetic_config = resolve_project_path(args.aesthetic_config)
        assert aesthetic_config is not None
        check_model_paths(
            checks,
            aesthetic_config=aesthetic_config,
            expect_pairwise=bool(args.expect_pairwise),
        )
    else:
        checks.add_ok("Aesthetic model path checks skipped (--disable_aesthetic_scoring=true).")

    check_local_paths(
        checks,
        local_input_dir=args.local_input_dir,
        local_output_dir=args.local_output_dir,
    )

    if args.mode in {"gemini", "worker"}:
        check_import(
            checks,
            "google.generativeai",
            required=(args.mode == "gemini"),
            hint="Install Gemini deps with: pip install -r requirements-gemini.txt",
        )
        check_gemini(checks, strict=(args.mode == "gemini"))

    if args.mode == "worker":
        check_import(
            checks,
            "firebase_admin",
            required=True,
            hint="Install worker deps with: pip install -r requirements-firebase-worker.txt",
        )
        check_import(
            checks,
            "pillow_heif",
            required=False,
            hint="Install pillow-heif for HEIC/HEIF decoding support in worker mode.",
        )
        check_worker_env(checks)

    check_vila_dependencies(checks, force=bool(args.check_vila))

    return checks


def print_human_report(args: argparse.Namespace, checks: CheckCollector) -> None:
    print(f"[Preflight] mode={args.mode}")
    print(f"[Preflight] repo_root={REPO_ROOT}")

    for message in checks.ok:
        print(f"[OK] {message}")
    for message in checks.warn:
        print(f"[WARN] {message}")
    for message in checks.fail:
        print(f"[FAIL] {message}")

    print(
        "[Summary] "
        f"ok={len(checks.ok)} warn={len(checks.warn)} fail={len(checks.fail)}"
    )

    if checks.fail:
        print(
            "[Result] FAILED. Resolve the [FAIL] items above, then rerun preflight."
        )
    else:
        print("[Result] SUCCESS. Required checks passed for the selected mode.")


def main() -> None:
    args = build_parser().parse_args()
    checks = run_checks(args)

    if args.json:
        payload = {
            "mode": args.mode,
            "repo_root": str(REPO_ROOT),
            "ok": checks.ok,
            "warn": checks.warn,
            "fail": checks.fail,
            "summary": {
                "ok": len(checks.ok),
                "warn": len(checks.warn),
                "fail": len(checks.fail),
            },
            "passed": len(checks.fail) == 0,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_human_report(args, checks)

    raise SystemExit(1 if checks.fail else 0)


if __name__ == "__main__":
    main()
