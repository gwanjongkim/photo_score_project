from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable


DEFAULT_JOB_COLLECTION = "jobs"
DEFAULT_LOCAL_TOP_K = 5
DEFAULT_POLL_INTERVAL_SECONDS = 10.0
FIRESTORE_ERROR_CODE_PREFIX = "acut_worker"
COMMAND_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_KEEP_FAILED_JOB_DIRS = True
PIPELINE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_DEBUG_FILE_NAMES_IN_FIRESTORE = 200
MAX_DEBUG_FILE_FORENSICS_IN_FIRESTORE = 80
MAGIC_HEIF_BRANDS = {"heic", "heix", "hevc", "hevx", "heim", "heis", "mif1", "msf1"}


class CancellationRequested(Exception):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process Firebase-backed A-cut analysis jobs using the existing Python pipeline.",
    )
    parser.add_argument("--job_collection", default=DEFAULT_JOB_COLLECTION)
    parser.add_argument("--job_id")
    parser.add_argument("--bucket", default=os.environ.get("FIREBASE_STORAGE_BUCKET"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll_interval_seconds", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--pipeline_python", default=sys.executable)
    parser.add_argument("--vila_python", default=os.environ.get("ACUT_VILA_PYTHON"))
    parser.add_argument("--vila_model_name", default=os.environ.get("ACUT_VILA_MODEL_NAME"))
    parser.add_argument(
        "--vila_prompt_preset",
        default=os.environ.get("ACUT_VILA_PROMPT_PRESET", "a_cut_basic"),
    )
    parser.add_argument(
        "--vila_local_files_only",
        action="store_true",
        default=os.environ.get("ACUT_VILA_LOCAL_FILES_ONLY", "").lower() == "true",
    )
    parser.add_argument("--local_input_dir")
    parser.add_argument("--local_output_dir")
    parser.add_argument("--local_top_k", type=int, default=DEFAULT_LOCAL_TOP_K)
    parser.add_argument("--local_enable_diversity", action="store_true")
    parser.add_argument(
        "--aesthetic_config",
        default=os.environ.get("ACUT_AESTHETIC_CONFIG"),
        help="Optional config forwarded to run_acut_pipeline.py for automatic server-side aesthetic scoring.",
    )
    parser.add_argument(
        "--disable_aesthetic_scoring",
        action="store_true",
        default=os.environ.get("ACUT_DISABLE_AESTHETIC_SCORING", "").lower() == "true",
        help="Forward --disable_aesthetic_scoring to run_acut_pipeline.py.",
    )
    parser.add_argument(
        "--keep_failed_job_dirs",
        action=argparse.BooleanOptionalAction,
        default=parse_bool(os.environ.get("ACUT_KEEP_FAILED_JOB_DIRS"), default=DEFAULT_KEEP_FAILED_JOB_DIRS),
        help=(
            "Preserve per-job temp directories when processing fails. "
            "Enabled by default for failure forensics."
        ),
    )
    parser.add_argument("--json_indent", type=int, default=2)
    return parser


def run_command(
    command: list[str],
    *,
    cwd: Path,
    should_cancel: Callable[[], bool] | None = None,
) -> str:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root()) if not existing_pythonpath else f"{repo_root()}:{existing_pythonpath}"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while True:
        try:
            stdout, stderr = process.communicate(timeout=COMMAND_POLL_INTERVAL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            if should_cancel is not None and should_cancel():
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                raise CancellationRequested("Analysis was cancelled by the user.")

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )

    return stdout.strip()


def guess_content_type(path: Path) -> str | None:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed


def normalize_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def extract_file_name(path_value: str | None) -> str | None:
    if not path_value:
        return None
    normalized = path_value.replace("\\", "/").strip()
    if not normalized:
        return None
    return normalized.rsplit("/", maxsplit=1)[-1]


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def parse_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def emit_worker_event(event: str, payload: dict[str, object], *, json_indent: int | None = None) -> None:
    message = {
        "event": event,
        **payload,
    }
    print(json.dumps(message, indent=json_indent, ensure_ascii=False))


def relative_file_names(paths: list[Path], *, base_dir: Path) -> list[str]:
    names: list[str] = []
    for path in sorted(paths):
        try:
            names.append(str(path.relative_to(base_dir)).replace("\\", "/"))
        except ValueError:
            names.append(path.name)
    return names


def compute_total_bytes(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += int(path.stat().st_size)
        except OSError:
            continue
    return total


def count_pipeline_images(paths: list[Path]) -> int:
    return sum(1 for path in paths if path.suffix.lower() in PIPELINE_IMAGE_SUFFIXES)


def read_file_header_bytes(path: Path, num_bytes: int = 64) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(max(0, int(num_bytes)))
    except Exception:
        return b""


def detect_magic_format(header: bytes) -> str | None:
    if len(header) >= 12 and header[4:8] == b"ftyp":
        brand = header[8:12].decode("ascii", errors="replace").lower()
        if brand in MAGIC_HEIF_BRANDS or brand.startswith("he"):
            return f"image/heif({brand})"
        return f"application/isobmff({brand})"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith(b"BM"):
        return "image/bmp"
    return None


def header_hex_preview(header: bytes, max_bytes: int = 24) -> str:
    if not header:
        return ""
    return header[: max(0, int(max_bytes))].hex()


def collect_downloaded_file_forensics(
    paths: list[Path],
    *,
    base_dir: Path,
) -> list[dict[str, object]]:
    forensics: list[dict[str, object]] = []
    for path in sorted(paths):
        header = read_file_header_bytes(path, num_bytes=64)
        mime_guess, _ = mimetypes.guess_type(str(path))
        try:
            size_bytes = int(path.stat().st_size)
        except OSError:
            size_bytes = None
        try:
            relative_name = str(path.relative_to(base_dir)).replace("\\", "/")
        except ValueError:
            relative_name = path.name
        forensics.append(
            {
                "file_name": relative_name,
                "suffix": path.suffix.lower(),
                "size_bytes": size_bytes,
                "mime_guess": mime_guess,
                "magic_format": detect_magic_format(header),
                "header_hex": header_hex_preview(header),
            }
        )
    return forensics


def preflight_validate_local_inputs(
    *,
    job_id: str,
    input_storage_prefix: str,
    local_temp_dir: Path,
    local_input_dir: Path,
    downloaded_file_count: int,
    downloaded_image_count: int,
) -> None:
    if not local_input_dir.exists():
        raise FileNotFoundError(
            "Pipeline preflight failed: local input directory does not exist. "
            f"jobId={job_id}, inputStoragePrefix={input_storage_prefix!r}, "
            f"localTempDir={local_temp_dir}, localInputDir={local_input_dir}, "
            f"downloadedFileCount={downloaded_file_count}, downloadedImageCount={downloaded_image_count}"
        )
    if not local_input_dir.is_dir():
        raise NotADirectoryError(
            "Pipeline preflight failed: local input path is not a directory. "
            f"jobId={job_id}, inputStoragePrefix={input_storage_prefix!r}, "
            f"localTempDir={local_temp_dir}, localInputDir={local_input_dir}, "
            f"downloadedFileCount={downloaded_file_count}, downloadedImageCount={downloaded_image_count}"
        )
    if downloaded_image_count <= 0:
        raise ValueError(
            "Pipeline preflight failed: zero image files were downloaded. "
            f"jobId={job_id}, inputStoragePrefix={input_storage_prefix!r}, "
            f"localTempDir={local_temp_dir}, localInputDir={local_input_dir}, "
            f"downloadedFileCount={downloaded_file_count}, downloadedImageCount={downloaded_image_count}"
        )


def compact_debug_metadata_for_firestore(
    debug_metadata: dict[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(debug_metadata, dict):
        return None
    compacted = dict(debug_metadata)
    names = compacted.get("downloaded_file_names")
    if isinstance(names, list) and len(names) > MAX_DEBUG_FILE_NAMES_IN_FIRESTORE:
        compacted["downloaded_file_names"] = names[:MAX_DEBUG_FILE_NAMES_IN_FIRESTORE]
        compacted["downloaded_file_names_total"] = len(names)
        compacted["downloaded_file_names_truncated"] = True
    forensics = compacted.get("downloaded_file_forensics")
    if isinstance(forensics, list) and len(forensics) > MAX_DEBUG_FILE_FORENSICS_IN_FIRESTORE:
        compacted["downloaded_file_forensics"] = forensics[:MAX_DEBUG_FILE_FORENSICS_IN_FIRESTORE]
        compacted["downloaded_file_forensics_total"] = len(forensics)
        compacted["downloaded_file_forensics_truncated"] = True
    return compacted


def _import_firebase():
    try:
        import firebase_admin
        from firebase_admin import credentials
        from firebase_admin import firestore, storage
    except ImportError as exc:  # pragma: no cover - import availability depends on worker env
        raise RuntimeError(
            "firebase-admin is not installed in this environment. "
            "Install requirements-firebase-worker.txt before running Firebase worker mode."
        ) from exc
    return firebase_admin, credentials, firestore, storage


def initialize_firebase(bucket_name: str | None):
    firebase_admin, credentials, firestore, storage = _import_firebase()
    app = firebase_admin.get_app() if firebase_admin._apps else None  # type: ignore[attr-defined]
    if app is None:
        options: dict[str, str] = {}
        if bucket_name:
            options["storageBucket"] = bucket_name

        project_id = (
            os.environ.get("FIREBASE_PROJECT_ID")
            or os.environ.get("GCLOUD_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or (bucket_name.removesuffix(".appspot.com") if bucket_name else None)
        )
        if project_id:
            options["projectId"] = project_id

        use_emulator_credentials = bool(
            os.environ.get("FIRESTORE_EMULATOR_HOST")
            or os.environ.get("FIREBASE_STORAGE_EMULATOR_HOST")
        )
        if use_emulator_credentials:
            class EmulatorCredentials(credentials.Base):  # pragma: no cover - emulator-only path
                def get_credential(self):
                    from google.auth.credentials import AnonymousCredentials

                    return AnonymousCredentials()

            firebase_admin.initialize_app(EmulatorCredentials(), options=options or None)
        else:
            firebase_admin.initialize_app(credentials.ApplicationDefault(), options=options or None)
    db = firestore.client()
    bucket = storage.bucket(bucket_name) if bucket_name else storage.bucket()
    return firestore, db, bucket


def claim_next_job(
    *,
    firestore_module,
    db,
    job_collection: str,
    job_id: str | None,
) -> tuple[Any, dict[str, object]] | None:
    collection = db.collection(job_collection)
    if job_id:
        snapshot = collection.document(job_id).get()
        if not snapshot.exists:
            raise FileNotFoundError(f"Job {job_id} was not found in {job_collection}.")
        candidate_refs = [snapshot.reference]
    else:
        candidate_refs = [snapshot.reference for snapshot in collection.where("status", "==", "queued").limit(1).stream()]

    if not candidate_refs:
        return None

    @firestore_module.transactional
    def _claim(transaction, job_ref):
        snapshot = job_ref.get(transaction=transaction)
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        if data.get("status") != "queued":
            return None
        transaction.update(
            job_ref,
            {
                "status": "running",
                "updatedAt": firestore_module.SERVER_TIMESTAMP,
                "startedAt": firestore_module.SERVER_TIMESTAMP,
            },
        )
        return data

    transaction = db.transaction()
    for ref in candidate_refs:
        data = _claim(transaction, ref)
        if data is not None:
            return ref, data
    return None


def download_prefix_to_directory(bucket, prefix: str, destination_dir: Path) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    normalized_prefix = normalize_prefix(prefix)
    downloaded: list[Path] = []
    for blob in bucket.list_blobs(prefix=normalized_prefix):
        if blob.name.endswith("/"):
            continue
        relative_path = blob.name[len(normalized_prefix) :].lstrip("/")
        target_path = destination_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target_path))
        downloaded.append(target_path)
    return downloaded


def download_blob_to_path(bucket, blob_path: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    blob = bucket.blob(normalize_prefix(blob_path))
    blob.download_to_filename(str(target_path))
    return target_path


def upload_output_files(
    bucket,
    *,
    local_output_dir: Path,
    output_storage_prefix: str,
    owner_uid: str | None = None,
    job_id: str | None = None,
) -> dict[str, str]:
    uploaded_paths: dict[str, str] = {}
    output_storage_prefix = normalize_prefix(output_storage_prefix)
    for file_name in [
        "app_results.json",
        "top_k_summary.json",
        "review_sheet.csv",
        "ranked_results.jsonl",
        "ranked_results.csv",
        "score_failures.json",
    ]:
        local_path = local_output_dir / file_name
        if not local_path.exists():
            continue
        remote_path = f"{output_storage_prefix}/{file_name}"
        blob = bucket.blob(remote_path)
        if owner_uid is not None or job_id is not None:
            blob.metadata = {
                key: value
                for key, value in {
                    "ownerUid": owner_uid,
                    "jobId": job_id,
                }.items()
                if value is not None
            }
        blob.upload_from_filename(str(local_path), content_type=guess_content_type(local_path))
        uploaded_paths[file_name] = remote_path
    return uploaded_paths


def maybe_run_vila_scoring(
    *,
    input_dir: Path,
    work_dir: Path,
    vila_python: str | None,
    vila_model_name: str | None,
    vila_prompt_preset: str,
    vila_local_files_only: bool,
    should_cancel: Callable[[], bool] | None = None,
) -> Path | None:
    if not vila_python or not vila_model_name:
        return None

    output_dir = work_dir / "vila_scores"
    command = [
        vila_python,
        "-m",
        "src.infer.score_image_folder_vila",
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--model_name",
        vila_model_name,
        "--prompt_preset",
        vila_prompt_preset,
    ]
    if vila_local_files_only:
        command.append("--local_files_only")

    run_command(command, cwd=repo_root(), should_cancel=should_cancel)
    output_csv = output_dir / "vila_scores.csv"
    return output_csv if output_csv.exists() else None


def run_pipeline_for_input_dir(
    *,
    input_dir: Path,
    output_dir: Path,
    pipeline_python: str,
    top_k: int,
    enable_diversity: bool,
    disable_aesthetic_scoring: bool,
    aesthetic_config: str | None,
    enable_pairwise_rerank: bool = False,
    rerank_pool_size: int | None = None,
    vila_scores_csv: Path | None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, object]:
    command = [
        pipeline_python,
        "-m",
        "src.infer.run_acut_pipeline",
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--top_k",
        str(top_k),
    ]
    if vila_scores_csv is not None:
        command.extend(["--vila_scores_csv", str(vila_scores_csv)])
    if disable_aesthetic_scoring:
        command.append("--disable_aesthetic_scoring")
    if aesthetic_config:
        command.extend(["--aesthetic_config", str(aesthetic_config)])
    if enable_pairwise_rerank:
        command.append("--enable_pairwise_rerank")
    if rerank_pool_size is not None:
        command.extend(["--rerank_pool_size", str(rerank_pool_size)])
    if enable_diversity:
        command.append("--enable_diversity")

    stdout = run_command(command, cwd=repo_root(), should_cancel=should_cancel)
    return json.loads(stdout) if stdout else {}


def local_mode_summary(
    *,
    pipeline_summary: dict[str, object],
    output_dir: Path,
    json_indent: int,
) -> None:
    app_results = parse_json_file(output_dir / "app_results.json")
    top_k_summary = parse_json_file(output_dir / "top_k_summary.json")
    summary = {
        "mode": "local",
        "pipeline_summary": pipeline_summary,
        "app_results_count": len(app_results) if isinstance(app_results, list) else None,
        "app_result_item_keys": sorted(app_results[0].keys()) if isinstance(app_results, list) and app_results else [],
        "top_k_summary_keys": sorted(top_k_summary.keys()) if isinstance(top_k_summary, dict) else [],
        "metadata": {
            "schema_version": top_k_summary.get("schema_version") if isinstance(top_k_summary, dict) else None,
            "ranking_stage": top_k_summary.get("ranking_stage") if isinstance(top_k_summary, dict) else None,
            "score_semantics": top_k_summary.get("score_semantics") if isinstance(top_k_summary, dict) else None,
            "diversity_enabled": top_k_summary.get("diversity_enabled") if isinstance(top_k_summary, dict) else None,
        },
    }
    print(json.dumps(summary, indent=json_indent, ensure_ascii=False))


def run_local_mode(args: argparse.Namespace) -> None:
    if not args.local_input_dir or not args.local_output_dir:
        raise ValueError("Local mode requires both --local_input_dir and --local_output_dir.")

    input_dir = Path(args.local_input_dir)
    output_dir = Path(args.local_output_dir)
    work_dir = output_dir / "_worker_local"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    vila_scores_csv = maybe_run_vila_scoring(
        input_dir=input_dir,
        work_dir=work_dir,
        vila_python=args.vila_python,
        vila_model_name=args.vila_model_name,
        vila_prompt_preset=args.vila_prompt_preset,
        vila_local_files_only=args.vila_local_files_only,
    )
    pipeline_summary = run_pipeline_for_input_dir(
        input_dir=input_dir,
        output_dir=output_dir,
        pipeline_python=args.pipeline_python,
        top_k=args.local_top_k,
        enable_diversity=args.local_enable_diversity,
        disable_aesthetic_scoring=args.disable_aesthetic_scoring,
        aesthetic_config=args.aesthetic_config,
        enable_pairwise_rerank=False,
        rerank_pool_size=None,
        vila_scores_csv=vila_scores_csv,
    )
    local_mode_summary(
        pipeline_summary=pipeline_summary,
        output_dir=output_dir,
        json_indent=args.json_indent,
    )


def summarize_firestore_result(
    *,
    top_k_summary: dict[str, object],
    uploaded_outputs: dict[str, str],
    pipeline_summary: dict[str, object],
    requested_top_k: int,
) -> dict[str, object]:
    return {
        "status": "done",
        "updatedAt": None,
        "completedAt": None,
        "schemaVersion": top_k_summary.get("schema_version"),
        "rankingStage": top_k_summary.get("ranking_stage"),
        "scoreSemantics": top_k_summary.get("score_semantics"),
        "diversityEnabled": top_k_summary.get("diversity_enabled"),
        "finalOrderingUsesDiversity": top_k_summary.get("final_ordering_uses_diversity"),
        "finalScoreMatchesFinalRanking": top_k_summary.get("final_score_matches_final_ranking"),
        "summary": {
            "schemaVersion": top_k_summary.get("schema_version"),
            "rankingStage": top_k_summary.get("ranking_stage"),
            "scoreSemantics": top_k_summary.get("score_semantics"),
            "diversityEnabled": top_k_summary.get("diversity_enabled"),
            "finalOrderingUsesDiversity": top_k_summary.get("final_ordering_uses_diversity"),
            "finalScoreMatchesFinalRanking": top_k_summary.get("final_score_matches_final_ranking"),
            "selectedCount": top_k_summary.get("selected_count"),
            "rejectedCount": top_k_summary.get("rejected_count"),
            "topKRequested": requested_top_k,
            "topKItems": top_k_summary.get("top_k"),
        },
        "topK": requested_top_k,
        "topKItems": top_k_summary.get("top_k"),
        "outputs": {
            "appResultsJsonPath": uploaded_outputs.get("app_results.json"),
            "topKSummaryJsonPath": uploaded_outputs.get("top_k_summary.json"),
            "reviewSheetCsvPath": uploaded_outputs.get("review_sheet.csv"),
            "rankedResultsJsonlPath": uploaded_outputs.get("ranked_results.jsonl"),
            "rankedResultsCsvPath": uploaded_outputs.get("ranked_results.csv"),
            "scoreFailuresJsonPath": uploaded_outputs.get("score_failures.json"),
        },
        "pipelineSummary": pipeline_summary,
        "errorMessage": None,
        "errorCode": None,
        "error": None,
    }


def build_structured_error(
    *,
    stage: str,
    error: Exception,
    debug_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    error_code = f"{FIRESTORE_ERROR_CODE_PREFIX}/{stage}/{error.__class__.__name__}".lower()
    structured_error: dict[str, object] = {
        "code": error_code,
        "message": str(error) or error.__class__.__name__,
    }
    details = {
        "stage": stage,
        "type": error.__class__.__name__,
        "traceback": traceback.format_exc(limit=8),
    }
    compacted_debug_metadata = compact_debug_metadata_for_firestore(debug_metadata)
    if compacted_debug_metadata:
        details["debug"] = compacted_debug_metadata
    if any(value for value in details.values()):
        structured_error["details"] = details
    return structured_error


def current_job_status(job_ref) -> str | None:
    snapshot = job_ref.get()
    if not snapshot.exists:
        return None
    data = snapshot.to_dict() or {}
    status = data.get("status")
    return str(status) if status is not None else None


def is_cancellation_requested(job_ref) -> bool:
    return current_job_status(job_ref) in {"cancelling", "cancelled"}


def raise_if_cancelled(job_ref) -> None:
    if is_cancellation_requested(job_ref):
        raise CancellationRequested("Analysis was cancelled by the user.")


def mark_job_cancelled(job_ref, firestore_module) -> None:
    job_ref.set(
        {
            "status": "cancelled",
            "updatedAt": firestore_module.SERVER_TIMESTAMP,
            "completedAt": firestore_module.SERVER_TIMESTAMP,
            "cancelledAt": firestore_module.SERVER_TIMESTAMP,
            "errorMessage": None,
            "errorCode": None,
            "error": None,
        },
        merge=True,
    )


def mark_job_error(
    job_ref,
    firestore_module,
    *,
    stage: str,
    error: Exception,
    debug_metadata: dict[str, object] | None = None,
) -> None:
    structured_error = build_structured_error(stage=stage, error=error, debug_metadata=debug_metadata)
    job_ref.set(
        {
            "status": "error",
            "updatedAt": firestore_module.SERVER_TIMESTAMP,
            "completedAt": firestore_module.SERVER_TIMESTAMP,
            "errorMessage": structured_error["message"],
            "errorCode": structured_error["code"],
            "error": structured_error,
            "debugMetadata": compact_debug_metadata_for_firestore(debug_metadata),
        },
        merge=True,
    )


def process_job(
    *,
    args: argparse.Namespace,
    firestore_module,
    bucket,
    job_ref,
    job_data: dict[str, object],
) -> dict[str, object]:
    job_id = job_ref.id
    input_storage_prefix = str(job_data.get("inputStoragePrefix") or "")
    output_storage_prefix = str(job_data.get("outputStoragePrefix") or f"acut_jobs/{job_id}/outputs")
    pipeline_config = job_data.get("pipelineConfig") if isinstance(job_data.get("pipelineConfig"), dict) else {}
    top_k = int(job_data.get("topK") or pipeline_config.get("topK") or DEFAULT_LOCAL_TOP_K)
    enable_diversity = parse_bool(job_data.get("enableDiversity") or pipeline_config.get("enableDiversity"), default=False)
    disable_aesthetic_scoring = args.disable_aesthetic_scoring or parse_bool(
        job_data.get("disableAestheticScoring") or pipeline_config.get("disableAestheticScoring"),
        default=False,
    )
    aesthetic_config = (
        pipeline_config.get("aestheticConfig")
        or pipeline_config.get("aesthetic_config")
        or args.aesthetic_config
    )
    aesthetic_config = str(aesthetic_config).strip() if aesthetic_config is not None else None
    enable_pairwise_rerank = parse_bool(
        job_data.get("enablePairwiseRerank") or pipeline_config.get("enablePairwiseRerank"),
        default=False,
    )
    rerank_pool_size = parse_optional_int(pipeline_config.get("rerankPoolSize") or pipeline_config.get("rerank_pool_size"))
    owner_uid = str(job_data.get("userId") or "").strip() or None
    temp_root = Path(tempfile.mkdtemp(prefix=f"acut_job_{job_id}_"))
    input_dir = temp_root / "inputs"
    output_dir = temp_root / "outputs"
    should_cleanup = True
    debug_metadata: dict[str, object] = {
        "job_id": job_id,
        "input_storage_prefix": input_storage_prefix,
        "local_temp_dir": str(temp_root),
        "local_input_dir": str(input_dir),
        "local_output_dir": str(output_dir),
        "downloaded_file_count": 0,
        "downloaded_file_names": [],
        "downloaded_file_forensics": [],
        "downloaded_total_bytes": 0,
        "downloaded_image_count": 0,
        "keep_failed_job_dirs": bool(args.keep_failed_job_dirs),
        "cleanup_skipped": False,
    }

    try:
        raise_if_cancelled(job_ref)
        download_prefix_to_directory(bucket, input_storage_prefix, input_dir)
        downloaded_inputs = sorted(path for path in input_dir.rglob("*") if path.is_file())
        downloaded_file_names = relative_file_names(downloaded_inputs, base_dir=input_dir)
        downloaded_file_forensics = collect_downloaded_file_forensics(downloaded_inputs, base_dir=input_dir)
        downloaded_file_count = len(downloaded_inputs)
        downloaded_total_bytes = compute_total_bytes(downloaded_inputs)
        downloaded_image_count = count_pipeline_images(downloaded_inputs)
        debug_metadata.update(
            {
                "downloaded_file_count": int(downloaded_file_count),
                "downloaded_file_names": downloaded_file_names,
                "downloaded_file_forensics": downloaded_file_forensics,
                "downloaded_total_bytes": int(downloaded_total_bytes),
                "downloaded_image_count": int(downloaded_image_count),
            }
        )
        emit_worker_event(
            "acut_input_download_summary",
            {
                "jobId": job_id,
                "inputStoragePrefix": input_storage_prefix,
                "localTempDir": str(temp_root),
                "localInputDir": str(input_dir),
                "localOutputDir": str(output_dir),
                "inputDirExists": input_dir.exists(),
                "downloadedFileCount": int(downloaded_file_count),
                "downloadedImageCount": int(downloaded_image_count),
                "downloadedTotalBytes": int(downloaded_total_bytes),
                "downloadedFileNames": downloaded_file_names,
                "downloadedFileForensics": downloaded_file_forensics,
            },
            json_indent=args.json_indent,
        )
        preflight_validate_local_inputs(
            job_id=job_id,
            input_storage_prefix=input_storage_prefix,
            local_temp_dir=temp_root,
            local_input_dir=input_dir,
            downloaded_file_count=downloaded_file_count,
            downloaded_image_count=downloaded_image_count,
        )
        raise_if_cancelled(job_ref)

        vila_scores_csv: Path | None = None
        vila_scores_storage_path = pipeline_config.get("vilaScoresStoragePath")
        if isinstance(vila_scores_storage_path, str) and vila_scores_storage_path.strip():
            vila_scores_csv = download_blob_to_path(
                bucket,
                vila_scores_storage_path,
                temp_root / "precomputed_vila_scores.csv",
            )
        else:
            vila_scores_csv = maybe_run_vila_scoring(
                input_dir=input_dir,
                work_dir=temp_root,
                vila_python=args.vila_python,
                vila_model_name=args.vila_model_name,
                vila_prompt_preset=args.vila_prompt_preset,
                vila_local_files_only=args.vila_local_files_only,
                should_cancel=lambda: is_cancellation_requested(job_ref),
            )

        pipeline_summary = run_pipeline_for_input_dir(
            input_dir=input_dir,
            output_dir=output_dir,
            pipeline_python=args.pipeline_python,
            top_k=top_k,
            enable_diversity=enable_diversity,
            disable_aesthetic_scoring=disable_aesthetic_scoring,
            aesthetic_config=aesthetic_config,
            enable_pairwise_rerank=enable_pairwise_rerank,
            rerank_pool_size=rerank_pool_size,
            vila_scores_csv=vila_scores_csv,
            should_cancel=lambda: is_cancellation_requested(job_ref),
        )
        raise_if_cancelled(job_ref)
        uploaded_outputs = upload_output_files(
            bucket,
            local_output_dir=output_dir,
            output_storage_prefix=output_storage_prefix,
            owner_uid=owner_uid,
            job_id=job_id,
        )
        top_k_summary = parse_json_file(output_dir / "top_k_summary.json")
        result_payload = summarize_firestore_result(
            top_k_summary=top_k_summary,
            uploaded_outputs=uploaded_outputs,
            pipeline_summary=pipeline_summary,
            requested_top_k=top_k,
        )
        raise_if_cancelled(job_ref)
        job_ref.set(
            {
                **result_payload,
                "updatedAt": firestore_module.SERVER_TIMESTAMP,
                "completedAt": firestore_module.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return {
            "jobId": job_id,
            "status": "done",
            "topKSummary": top_k_summary,
            "outputs": uploaded_outputs,
        }
    except Exception as exc:
        should_cleanup = not bool(args.keep_failed_job_dirs)
        debug_metadata["cleanup_skipped"] = not should_cleanup
        debug_metadata["first_failure_reason"] = f"{exc.__class__.__name__}: {exc}"
        if not should_cleanup:
            debug_metadata["preserved_temp_dir"] = str(temp_root)
        emit_worker_event(
            "acut_job_failure_forensics",
            {
                "jobId": job_id,
                "status": "error",
                "firstFailureReason": debug_metadata.get("first_failure_reason"),
                "localInputDir": debug_metadata.get("local_input_dir"),
                "localOutputDir": debug_metadata.get("local_output_dir"),
                "downloadedFileCount": debug_metadata.get("downloaded_file_count"),
                "downloadedFileNames": debug_metadata.get("downloaded_file_names"),
                "downloadedFileForensics": debug_metadata.get("downloaded_file_forensics"),
                "cleanupSkipped": debug_metadata.get("cleanup_skipped"),
                "preservedTempDir": debug_metadata.get("preserved_temp_dir"),
            },
            json_indent=args.json_indent,
        )
        setattr(exc, "_acut_debug_metadata", debug_metadata)
        raise
    finally:
        if should_cleanup:
            shutil.rmtree(temp_root, ignore_errors=True)
        else:
            emit_worker_event(
                "acut_failed_job_dir_preserved",
                {
                    "jobId": job_id,
                    "localTempDir": str(temp_root),
                    "localInputDir": str(input_dir),
                    "localOutputDir": str(output_dir),
                },
                json_indent=args.json_indent,
            )


def worker_loop(args: argparse.Namespace) -> None:
    firestore_module, db, bucket = initialize_firebase(args.bucket)

    while True:
        claimed = claim_next_job(
            firestore_module=firestore_module,
            db=db,
            job_collection=args.job_collection,
            job_id=args.job_id,
        )
        if claimed is None:
            if args.once or args.job_id:
                print(
                    json.dumps(
                        {
                            "status": "idle",
                            "message": "No queued A-cut jobs were available.",
                        },
                        indent=args.json_indent,
                        ensure_ascii=False,
                    )
                )
                return
            time.sleep(max(1.0, args.poll_interval_seconds))
            continue

        job_ref, job_data = claimed
        try:
            result = process_job(
                args=args,
                firestore_module=firestore_module,
                bucket=bucket,
                job_ref=job_ref,
                job_data=job_data,
            )
            print(json.dumps(result, indent=args.json_indent, ensure_ascii=False))
        except CancellationRequested as exc:
            mark_job_cancelled(job_ref, firestore_module)
            print(
                json.dumps(
                    {
                        "jobId": job_ref.id,
                        "status": "cancelled",
                        "message": str(exc),
                    },
                    indent=args.json_indent,
                    ensure_ascii=False,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on external Firebase environment
            mark_job_error(
                job_ref,
                firestore_module,
                stage="process_job",
                error=exc,
                debug_metadata=getattr(exc, "_acut_debug_metadata", None),
            )
            raise

        if args.once or args.job_id:
            return


def main() -> None:
    args = build_parser().parse_args()
    if args.local_input_dir or args.local_output_dir:
        run_local_mode(args)
        return
    worker_loop(args)


if __name__ == "__main__":
    main()
