import { initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import { getStorage } from "firebase-admin/storage";

const projectId = process.env.GCLOUD_PROJECT || "demo-acut";
const region = "asia-northeast3";
const functionUrl =
  `http://127.0.0.1:5001/${projectId}/${region}/enqueueAcutAnalysis`;
const jobId = "job_emulator_smoke";
const inputStoragePrefix = `acut_jobs/${jobId}/inputs`;
const outputStoragePrefix = `acut_jobs/${jobId}/outputs`;
const storageBucket = `${projectId}.appspot.com`;

initializeApp({
  projectId,
  storageBucket,
});

async function seedInputObject() {
  const bucket = getStorage().bucket();
  const file = bucket.file(`${inputStoragePrefix}/000_test.jpg`);
  await file.save(Buffer.from("emulator-smoke"), {
    contentType: "image/jpeg",
    resumable: false,
  });
}

async function callFunction() {
  const response = await fetch(functionUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      data: {
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
      },
    }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`Function call failed: ${response.status} ${JSON.stringify(payload)}`);
  }
  return payload;
}

async function readJobDocument() {
  const snapshot = await getFirestore().collection("jobs").doc(jobId).get();
  return snapshot.data() || null;
}

async function main() {
  await seedInputObject();
  const functionPayload = await callFunction();
  const job = await readJobDocument();
  console.log(JSON.stringify({
    functionPayload,
    jobSummary: job
      ? {
          status: job.status,
          imageCount: job.imageCount,
          topK: job.topK,
          inputStoragePrefix: job.inputStoragePrefix,
          outputStoragePrefix: job.outputStoragePrefix,
          outputs: job.outputs,
          worker: job.worker,
        }
      : null,
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
