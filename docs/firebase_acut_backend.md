# Firebase A-cut Backend

## Architecture

Use Firebase as the job queue and result transport, not as the heavy inference runtime.

Recommended flow:

1. Flutter uploads candidate images to Firebase Storage under `acut_jobs/{jobId}/inputs/`.
2. Flutter calls the Firebase callable function `enqueueAcutAnalysis`.
3. The callable validates the request, validates ownership metadata on uploaded inputs, and writes `jobs/{jobId}` with `status: queued`.
4. A Python worker watches Firestore for queued jobs.
5. The worker downloads inputs from Storage, runs the existing Python pipeline, uploads outputs, and updates Firestore.
6. Flutter can request cancellation through `cancelAcutAnalysis`.
7. Flutter listens to `jobs/{jobId}` and renders progress and final results.

This repo now includes both layers:

- `functions/src/*`: thin Firebase queue API
- `src/firebase/acut_job_worker.py`: Python worker for the real analysis work

## Why Not Run Python In Firebase Functions

The current A-cut pipeline depends on TensorFlow bundle scoring and optional VILA-lite prompt scoring. That is too heavy and too environment-specific for a practical Firebase Functions deployment.

For production:

- Firebase Functions should stay as the queue/control plane.
- The Python worker should run on Cloud Run, a VM, or another long-lived worker environment.

Cloud Run is the recommended production target.

## Firestore Job Shape

Collection:

- `jobs/{jobId}`

Queued job example:

```json
{
  "jobSchemaVersion": "acut_firestore_job.v1",
  "status": "queued",
  "createdAt": "serverTimestamp",
  "updatedAt": "serverTimestamp",
  "startedAt": null,
  "completedAt": null,
  "cancelRequestedAt": null,
  "cancelledAt": null,
  "cancelRequestedBy": null,
  "userId": "uid_123",
  "imageCount": 6,
  "inputStoragePrefix": "acut_jobs/job_123/inputs",
  "outputStoragePrefix": "acut_jobs/job_123/outputs",
  "topK": 5,
  "enableDiversity": false,
  "inputFiles": [
    {
      "uploadFileName": "000_IMG_1234.jpg",
      "displayName": "IMG_1234.jpg",
      "storagePath": "acut_jobs/job_123/inputs/000_IMG_1234.jpg",
      "selectedIndex": 0
    }
  ],
  "pipelineConfig": {
    "topK": 5,
    "enableDiversity": false
  },
  "errorCode": null,
  "error": null,
  "outputs": {
    "appResultsJsonPath": "acut_jobs/job_123/outputs/app_results.json",
    "topKSummaryJsonPath": "acut_jobs/job_123/outputs/top_k_summary.json",
    "reviewSheetCsvPath": "acut_jobs/job_123/outputs/review_sheet.csv"
  }
}
```

`inputFiles[].selectedIndex` is zero-based and should match the client-side selection order used by the Flutter app.

Completed job example:

```json
{
  "status": "done",
  "schemaVersion": "acut_product_app.v1",
  "rankingStage": "post_rerank",
  "scoreSemantics": "rank, selected, status, and final_score_after_rerank all reflect the same final post-rerank order because diversity reranking is disabled.",
  "diversityEnabled": false,
  "finalOrderingUsesDiversity": false,
  "finalScoreMatchesFinalRanking": true,
  "summary": {
    "selectedCount": 5,
    "rejectedCount": 10,
    "topKRequested": 5,
    "topKItems": [
      {
        "rank": 1,
        "image_path": "000_IMG_1234.jpg",
        "image_file_name": "000_IMG_1234.jpg",
        "final_score_after_rerank": 0.597029,
        "acut_short_reason": "Composition carried this frame into the A-cut, helped by technical quality."
      }
    ]
  },
  "topK": 5,
  "topKItems": [
    {
      "rank": 1,
      "image_path": "000_IMG_1234.jpg",
      "final_score_after_rerank": 0.597029,
      "acut_short_reason": "Composition carried this frame into the A-cut, helped by technical quality."
    }
  ],
  "outputs": {
    "appResultsJsonPath": "acut_jobs/job_123/outputs/app_results.json",
    "topKSummaryJsonPath": "acut_jobs/job_123/outputs/top_k_summary.json",
    "reviewSheetCsvPath": "acut_jobs/job_123/outputs/review_sheet.csv"
  }
}
```

## Storage Layout

- `acut_jobs/{jobId}/inputs/...`
- `acut_jobs/{jobId}/outputs/app_results.json`
- `acut_jobs/{jobId}/outputs/top_k_summary.json`
- `acut_jobs/{jobId}/outputs/review_sheet.csv`

## Rules Files

This repo now includes production-safe baseline rules:

- `firestore.rules`
- `storage.rules`

They assume:

- the mobile client signs in with Firebase Auth
- Firebase Anonymous Auth or another sign-in provider is enabled in the Firebase project
- `jobs/{jobId}.userId == request.auth.uid`
- Storage input uploads include `ownerUid` custom metadata matching `request.auth.uid`
- Storage input uploads include `jobId` custom metadata matching the path job id

Those assumptions are not wired in the current Flutter app yet, so real project
rollout still requires adding Auth or intentionally relaxing rules in a
non-production Firebase project for temporary testing only.

## Functions Layer

Entry point:

- `functions/src/index.ts`

Callable:

- `enqueueAcutAnalysis`
- `cancelAcutAnalysis`

Responsibilities:

- validate `jobId`, `inputStoragePrefix`, `imageCount`, `topK`
- validate `inputFiles` shape when provided
- verify uploaded inputs exist in Storage
- verify uploaded inputs are owned by the authenticated caller
- create or update `jobs/{jobId}` with `status: queued`
- return the job path and output paths

Cancel responsibilities:

- require authenticated caller
- verify `jobs/{jobId}.userId == auth.uid`
- move `queued -> cancelled`
- move `running -> cancelling`
- leave terminal jobs idempotent

## Firebase Setup

1. Create a Firebase project and enable Firestore and Storage.
2. Set the default Storage bucket that both the Flutter app and worker will use.
3. In this repo root, install the Functions dependencies:

```bash
cd functions
npm install
npm run build
```

4. Authenticate Firebase CLI and deploy the callable queue function:

```bash
firebase deploy --only functions
```

5. Install the Python worker dependencies:

```bash
cd <repo_root>/photo_score_project
PYTHONPATH=. ./.venv_gpu/bin/python -m pip install -r requirements-firebase-worker.txt
```

6. Run the worker locally for development, or containerize `src/firebase/acut_job_worker.py` for Cloud Run in production.

7. If you plan to use the Firebase emulator suite, ensure Java is available and the repo root contains:

- `firebase.json`
- `firestore.rules`
- `storage.rules`

## Flutter App Setup

The Flutter app expects:

- `firebase_core`
- `firebase_auth`
- `cloud_firestore`
- `firebase_storage`
- `cloud_functions`

Required platform setup:

1. Add the Android app and/or iOS app in Firebase.
2. Place `google-services.json` under `android/app/`.
3. Place `GoogleService-Info.plist` under `ios/Runner/`.
4. Apply the Google Services Gradle plugin on Android so `google-services.json` is consumed during build.
5. Run `flutterfire configure` so the placeholder `lib/firebase_options.dart` in
   the Flutter repo is replaced with real generated options.
6. Enable Firebase Anonymous Auth unless you are immediately replacing it with another auth method.
7. Ensure the app initializes Firebase and signs in before attempting A-cut analysis.
8. Point the app at the same Firebase project, region `asia-northeast3`, and Storage bucket used by the worker.
9. Normalize upload bytes to server-decodable formats (JPEG/PNG) before writing to Storage.
10. Keep file extension and content type aligned with actual bytes (for example, avoid uploading HEIC bytes as `.jpg`).

## Python Worker

Entry point:

- `src/firebase/acut_job_worker.py`

Worker responsibilities:

- claim a queued Firestore job
- mark it `running`
- download Storage inputs to a temp folder
- optionally run VILA scoring if worker config is available
- run `src/infer/run_acut_pipeline.py`
- upload `app_results.json`, `top_k_summary.json`, `review_sheet.csv`, and ranked exports
- update Firestore to `done` or `error`
- write structured Firestore errors shaped as `{ code, message, details? }`
- honor cooperative cancellation by treating `cancelling` / `cancelled` as stop signals

Worker setup:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m pip install -r requirements-firebase-worker.txt
```

`requirements-firebase-worker.txt` includes `pillow-heif` so the worker can decode HEIC/HEIF uploads.

Environment/config notes:

- `--bucket` or Firebase default bucket must be configured.
- `--pipeline_python` defaults to the current Python interpreter.
- `--vila_python` and `--vila_model_name` are optional. If omitted, the worker skips VILA scoring and still runs the A-cut pipeline.
- For VILA-lite scoring, a separate PyTorch-capable env is still recommended.

## Local Smoke Tests

### 1. Pipeline-only local worker smoke

This bypasses Firebase and validates the worker boundary against a local image folder:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/firebase/acut_job_worker.py \
  --local_input_dir test_samples \
  --local_output_dir outputs/firebase_worker_local_smoke \
  --local_top_k 5
```

Optional diversity:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/firebase/acut_job_worker.py \
  --local_input_dir test_samples \
  --local_output_dir outputs/firebase_worker_local_smoke_diversity \
  --local_top_k 5 \
  --local_enable_diversity
```

### 2. Firebase emulator path

From the repo root:

```bash
cd functions
npm install
npm run build
cd ..
firebase emulators:start --project demo-acut --only functions,firestore,storage
```

Run the Python worker in another terminal:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/firebase/acut_job_worker.py --once
```

Then upload sample files to Storage and call `enqueueAcutAnalysis`.

### 3. Emulator enqueue smoke

This repo also includes a small emulator integration smoke that:

- writes a dummy input object to Storage
- calls `enqueueAcutAnalysis`
- confirms that `jobs/{jobId}` is written in Firestore

```bash
cd <repo_root>/photo_score_project
firebase emulators:exec --project demo-acut --only functions,firestore,storage \
  "cd functions && node scripts/enqueue_emulator_smoke.mjs"
```

### 4. Full emulator flow smoke

This extends the smoke to cover the full async flow:

- uploads a real sample image to emulated Storage
- calls `enqueueAcutAnalysis`
- runs the Python worker once against emulated Firestore and Storage
- verifies the job reaches `status: done`

```bash
cd <repo_root>/photo_score_project
PYTHONPATH=. ./.venv_gpu/bin/python -m pip install -r requirements-firebase-worker.txt
firebase emulators:exec --project demo-acut --only functions,firestore,storage \
  "cd functions && node scripts/full_flow_emulator_smoke.mjs"
```

The worker now supports emulator-only anonymous credentials automatically when
`FIRESTORE_EMULATOR_HOST` or `FIREBASE_STORAGE_EMULATOR_HOST` is present.

## Cloud Run Worker Invocation

Production should run the Python worker outside Firebase Functions.

Recommended worker command:

```bash
PYTHONPATH=. python src/firebase/acut_job_worker.py \
  --bucket <your-storage-bucket> \
  --job_collection jobs
```

Optional VILA-lite worker configuration:

```bash
PYTHONPATH=. python src/firebase/acut_job_worker.py \
  --bucket <your-storage-bucket> \
  --job_collection jobs \
  --vila_python /opt/venvs/vila/bin/python \
  --vila_model_name <your-vila-model> \
  --vila_prompt_preset a_cut_basic
```

For Cloud Run, package this repo with the required Python environment, then run one worker process per container instance. The callable Function remains the queue entrypoint; Cloud Run pulls queued jobs from Firestore and writes outputs back to Firestore and Storage.

## Production Recommendation

Use this split in production:

- Firebase Functions: create/validate queued jobs
- Firestore + Storage: job state and artifacts
- Cloud Run: Python worker container that runs TensorFlow and optional VILA-lite scoring

That keeps the current WSL pipeline intact while moving only the orchestration boundary to Firebase.
