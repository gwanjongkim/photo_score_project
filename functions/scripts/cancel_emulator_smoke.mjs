import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const functionsDir = path.resolve(__dirname, "..");

const projectId = process.env.GCLOUD_PROJECT || "demo-acut";
const storageBucket = `${projectId}.appspot.com`;
const authUid = "emulator-anon-user";
const queuedJobId = "job_emulator_cancel_queued";
const runningJobId = "job_emulator_cancel_running";

initializeApp({
  projectId,
  storageBucket,
});

async function loadRunnerModule() {
  const moduleUrl = pathToFileURL(
    path.join(functionsDir, "lib", "acutRunner.js"),
  ).href;
  const moduleNamespace = await import(moduleUrl);
  return moduleNamespace.default ?? moduleNamespace;
}

async function seedJob(jobId, status) {
  const firestore = getFirestore();
  await firestore.collection("jobs").doc(jobId).set({
    jobSchemaVersion: "acut_firestore_job.v1",
    status,
    userId: authUid,
    imageCount: 1,
    inputStoragePrefix: `acut_jobs/${jobId}/inputs`,
    outputStoragePrefix: `acut_jobs/${jobId}/outputs`,
    topK: 1,
    enableDiversity: false,
    diversityEnabled: false,
    inputFiles: [],
    pipelineConfig: {
      topK: 1,
      enableDiversity: false,
    },
    clientContext: {
      photoTypeMode: "auto",
    },
    outputs: {
      appResultsJsonPath: `acut_jobs/${jobId}/outputs/app_results.json`,
      topKSummaryJsonPath: `acut_jobs/${jobId}/outputs/top_k_summary.json`,
      reviewSheetCsvPath: `acut_jobs/${jobId}/outputs/review_sheet.csv`,
    },
  }, { merge: true });
}

async function main() {
  const acutRunner = await loadRunnerModule();
  const { cancelAcutAnalysisJob } = acutRunner;

  await seedJob(queuedJobId, "queued");
  const queuedResult = await cancelAcutAnalysisJob(
    { jobId: queuedJobId },
    authUid,
  );

  await seedJob(runningJobId, "running");
  const runningResult = await cancelAcutAnalysisJob(
    { jobId: runningJobId },
    authUid,
  );

  const firestore = getFirestore();
  const queuedSnapshot = await firestore.collection("jobs").doc(queuedJobId).get();
  const runningSnapshot = await firestore.collection("jobs").doc(runningJobId).get();

  console.log(JSON.stringify({
    queuedResult,
    queuedJob: queuedSnapshot.data(),
    runningResult,
    runningJob: runningSnapshot.data(),
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
