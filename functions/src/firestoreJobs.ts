import { FieldValue, getFirestore } from "firebase-admin/firestore";

export type AcutJobStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "done"
  | "error";
export const ACUT_JOB_SCHEMA_VERSION = "acut_firestore_job.v1";

export class JobMutationError extends Error {
  constructor(
    readonly code: "not-found" | "permission-denied" | "failed-precondition",
    message: string,
  ) {
    super(message);
  }
}

export interface AcutInputFile {
  uploadFileName: string;
  displayName: string;
  storagePath: string;
  selectedIndex: number;
}

export interface QueueJobPayload {
  jobId: string;
  imageCount: number;
  inputStoragePrefix: string;
  outputStoragePrefix: string;
  topK: number;
  enableDiversity: boolean;
  userId?: string | null;
  inputFiles: AcutInputFile[];
  pipelineConfig: Record<string, unknown>;
  clientContext: Record<string, unknown>;
}

export interface QueueJobResult {
  jobId: string;
  status: AcutJobStatus;
  jobPath: string;
  outputStoragePrefix: string;
}

export interface CancelJobResult {
  jobId: string;
  status: AcutJobStatus;
  previousStatus: AcutJobStatus;
  jobPath: string;
}

export function jobsCollection(collectionName = "jobs") {
  return getFirestore().collection(collectionName);
}

export function buildQueuedJobDocument(payload: QueueJobPayload) {
  return {
    jobSchemaVersion: ACUT_JOB_SCHEMA_VERSION,
    status: "queued" as AcutJobStatus,
    createdAt: FieldValue.serverTimestamp(),
    updatedAt: FieldValue.serverTimestamp(),
    startedAt: null,
    completedAt: null,
    cancelRequestedAt: null,
    cancelledAt: null,
    cancelRequestedBy: null,
    userId: payload.userId ?? null,
    imageCount: payload.imageCount,
    inputStoragePrefix: payload.inputStoragePrefix,
    outputStoragePrefix: payload.outputStoragePrefix,
    topK: payload.topK,
    enableDiversity: payload.enableDiversity,
    diversityEnabled: payload.enableDiversity,
    inputFiles: payload.inputFiles,
    pipelineConfig: payload.pipelineConfig,
    clientContext: payload.clientContext,
    errorMessage: null,
    errorCode: null,
    finalOrderingUsesDiversity: null,
    finalScoreMatchesFinalRanking: null,
    summary: null,
    error: null,
    outputs: {
      appResultsJsonPath: `${payload.outputStoragePrefix}/app_results.json`,
      topKSummaryJsonPath: `${payload.outputStoragePrefix}/top_k_summary.json`,
      reviewSheetCsvPath: `${payload.outputStoragePrefix}/review_sheet.csv`,
    },
    worker: {
      kind: "python",
      queueMode: "firestore",
      recommendedRuntime: "cloud-run",
    },
  };
}

export async function createOrUpdateQueuedJob(
  payload: QueueJobPayload,
  collectionName = "jobs",
): Promise<QueueJobResult> {
  const ref = jobsCollection(collectionName).doc(payload.jobId);
  const snapshot = await ref.get();
  if (snapshot.exists) {
    const currentStatus = snapshot.get("status");
    if (currentStatus === "running" || currentStatus === "done") {
      throw new Error(
        `Job ${payload.jobId} already exists with terminal-or-active status ${String(currentStatus)}.`,
      );
    }
  }

  await ref.set(buildQueuedJobDocument(payload), { merge: true });
  return {
    jobId: payload.jobId,
    status: "queued",
    jobPath: ref.path,
    outputStoragePrefix: payload.outputStoragePrefix,
  };
}

function normalizeJobStatus(value: unknown): AcutJobStatus {
  switch (value) {
    case "queued":
    case "running":
    case "cancelling":
    case "cancelled":
    case "done":
    case "error":
      return value;
    default:
      throw new JobMutationError(
        "failed-precondition",
        `Unsupported job status ${String(value)}.`,
      );
  }
}

export async function requestJobCancellation(
  jobId: string,
  authUid: string,
  collectionName = "jobs",
): Promise<CancelJobResult> {
  const ref = jobsCollection(collectionName).doc(jobId);

  return getFirestore().runTransaction(async (transaction) => {
    const snapshot = await transaction.get(ref);
    if (!snapshot.exists) {
      throw new JobMutationError(
        "not-found",
        `Job ${jobId} was not found in ${collectionName}.`,
      );
    }

    const data = snapshot.data() ?? {};
    const userId = typeof data.userId === "string" ? data.userId : null;
    if (userId == null || userId !== authUid) {
      throw new JobMutationError(
        "permission-denied",
        `You do not have access to cancel job ${jobId}.`,
      );
    }

    const previousStatus = normalizeJobStatus(data.status);
    let nextStatus = previousStatus;
    if (previousStatus === "queued") {
      nextStatus = "cancelled";
      transaction.update(ref, {
        status: nextStatus,
        updatedAt: FieldValue.serverTimestamp(),
        completedAt: FieldValue.serverTimestamp(),
        cancelRequestedAt: FieldValue.serverTimestamp(),
        cancelledAt: FieldValue.serverTimestamp(),
        cancelRequestedBy: authUid,
        errorMessage: null,
        errorCode: null,
        error: null,
      });
    } else if (previousStatus === "running") {
      nextStatus = "cancelling";
      transaction.update(ref, {
        status: nextStatus,
        updatedAt: FieldValue.serverTimestamp(),
        cancelRequestedAt: FieldValue.serverTimestamp(),
        cancelRequestedBy: authUid,
      });
    } else if (previousStatus === "cancelling" || previousStatus === "cancelled" || previousStatus === "done" || previousStatus === "error") {
      nextStatus = previousStatus;
    }

    return {
      jobId,
      status: nextStatus,
      previousStatus,
      jobPath: ref.path,
    };
  });
}
