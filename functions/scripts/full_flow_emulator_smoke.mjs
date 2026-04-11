import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawn } from "node:child_process";

import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import { getStorage } from "firebase-admin/storage";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const functionsDir = path.resolve(__dirname, "..");
const repoRoot = path.resolve(functionsDir, "..");

const projectId = process.env.GCLOUD_PROJECT || "demo-acut";
const storageBucket = `${projectId}.appspot.com`;
const authUid = "emulator-anon-user";
const jobId = "job_emulator_full_flow";
const inputStoragePrefix = `acut_jobs/${jobId}/inputs`;
const outputStoragePrefix = `acut_jobs/${jobId}/outputs`;
const sampleImagePath = path.join(
  repoRoot,
  "test_samples",
  "KakaoTalk_20260330_180646779.jpg",
);

initializeApp({
  projectId,
  storageBucket,
});

async function uploadSample() {
  const imageBytes = await readFile(sampleImagePath);
  const bucket = getStorage().bucket();
  const file = bucket.file(`${inputStoragePrefix}/000_test.jpg`);
  await file.save(imageBytes, {
    contentType: "image/jpeg",
    resumable: false,
    metadata: {
      metadata: {
        ownerUid: authUid,
        jobId,
      },
    },
  });
}

async function enqueueJob() {
  const moduleUrl = pathToFileURL(
    path.join(functionsDir, "lib", "acutRunner.js"),
  ).href;
  const moduleNamespace = await import(moduleUrl);
  const acutRunner = moduleNamespace.default ?? moduleNamespace;
  return acutRunner.enqueueAcutAnalysisJob({
    jobId,
    imageCount: 1,
    inputStoragePrefix,
    outputStoragePrefix,
    topK: 1,
    enableDiversity: false,
    inputFiles: [
      {
        uploadFileName: "000_test.jpg",
        displayName: "test.jpg",
        storagePath: `${inputStoragePrefix}/000_test.jpg`,
        selectedIndex: 0,
      },
    ],
    clientContext: {
      photoTypeMode: "auto",
    },
    pipelineConfig: {
      topK: 1,
      enableDiversity: false,
    },
  }, authUid);
}

async function runWorkerOnce() {
  const pythonPath = path.join(repoRoot, ".venv_gpu", "bin", "python");
  const workerPath = path.join(repoRoot, "src", "firebase", "acut_job_worker.py");
  const env = {
    ...process.env,
    PYTHONPATH: repoRoot,
  };

  await new Promise((resolve, reject) => {
    const child = spawn(
      pythonPath,
      [workerPath, "--once", "--bucket", storageBucket],
      {
        cwd: repoRoot,
        env,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("exit", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      reject(new Error(`Worker exited with code ${code}\nSTDOUT:\n${stdout}\nSTDERR:\n${stderr}`));
    });
    child.on("error", reject);
  });
}

async function waitForDone() {
  const firestore = getFirestore();
  for (let attempt = 0; attempt < 60; attempt++) {
    const snapshot = await firestore.collection("jobs").doc(jobId).get();
    const data = snapshot.data();
    if (data?.status === "done") {
      return data;
    }
    if (data?.status === "error") {
      throw new Error(`Job failed: ${JSON.stringify(data)}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error("Timed out waiting for worker to finish the queued job.");
}

async function main() {
  await uploadSample();
  const functionPayload = await enqueueJob();
  await runWorkerOnce();
  const job = await waitForDone();

  console.log(JSON.stringify({
    functionPayload,
    finalJob: {
      status: job.status,
      schemaVersion: job.schemaVersion,
      rankingStage: job.rankingStage,
      diversityEnabled: job.diversityEnabled,
      outputs: job.outputs,
      summary: job.summary,
    },
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
