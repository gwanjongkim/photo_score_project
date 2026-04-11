"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.JobMutationError = exports.ACUT_JOB_SCHEMA_VERSION = void 0;
exports.jobsCollection = jobsCollection;
exports.buildQueuedJobDocument = buildQueuedJobDocument;
exports.createOrUpdateQueuedJob = createOrUpdateQueuedJob;
exports.requestJobCancellation = requestJobCancellation;
const firestore_1 = require("firebase-admin/firestore");
exports.ACUT_JOB_SCHEMA_VERSION = "acut_firestore_job.v1";
class JobMutationError extends Error {
    code;
    constructor(code, message) {
        super(message);
        this.code = code;
    }
}
exports.JobMutationError = JobMutationError;
function jobsCollection(collectionName = "jobs") {
    return (0, firestore_1.getFirestore)().collection(collectionName);
}
function buildQueuedJobDocument(payload) {
    return {
        jobSchemaVersion: exports.ACUT_JOB_SCHEMA_VERSION,
        status: "queued",
        createdAt: firestore_1.FieldValue.serverTimestamp(),
        updatedAt: firestore_1.FieldValue.serverTimestamp(),
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
async function createOrUpdateQueuedJob(payload, collectionName = "jobs") {
    const ref = jobsCollection(collectionName).doc(payload.jobId);
    const snapshot = await ref.get();
    if (snapshot.exists) {
        const currentStatus = snapshot.get("status");
        if (currentStatus === "running" || currentStatus === "done") {
            throw new Error(`Job ${payload.jobId} already exists with terminal-or-active status ${String(currentStatus)}.`);
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
function normalizeJobStatus(value) {
    switch (value) {
        case "queued":
        case "running":
        case "cancelling":
        case "cancelled":
        case "done":
        case "error":
            return value;
        default:
            throw new JobMutationError("failed-precondition", `Unsupported job status ${String(value)}.`);
    }
}
async function requestJobCancellation(jobId, authUid, collectionName = "jobs") {
    const ref = jobsCollection(collectionName).doc(jobId);
    return (0, firestore_1.getFirestore)().runTransaction(async (transaction) => {
        const snapshot = await transaction.get(ref);
        if (!snapshot.exists) {
            throw new JobMutationError("not-found", `Job ${jobId} was not found in ${collectionName}.`);
        }
        const data = snapshot.data() ?? {};
        const userId = typeof data.userId === "string" ? data.userId : null;
        if (userId == null || userId !== authUid) {
            throw new JobMutationError("permission-denied", `You do not have access to cancel job ${jobId}.`);
        }
        const previousStatus = normalizeJobStatus(data.status);
        let nextStatus = previousStatus;
        if (previousStatus === "queued") {
            nextStatus = "cancelled";
            transaction.update(ref, {
                status: nextStatus,
                updatedAt: firestore_1.FieldValue.serverTimestamp(),
                completedAt: firestore_1.FieldValue.serverTimestamp(),
                cancelRequestedAt: firestore_1.FieldValue.serverTimestamp(),
                cancelledAt: firestore_1.FieldValue.serverTimestamp(),
                cancelRequestedBy: authUid,
                errorMessage: null,
                errorCode: null,
                error: null,
            });
        }
        else if (previousStatus === "running") {
            nextStatus = "cancelling";
            transaction.update(ref, {
                status: nextStatus,
                updatedAt: firestore_1.FieldValue.serverTimestamp(),
                cancelRequestedAt: firestore_1.FieldValue.serverTimestamp(),
                cancelRequestedBy: authUid,
            });
        }
        else if (previousStatus === "cancelling" || previousStatus === "cancelled" || previousStatus === "done" || previousStatus === "error") {
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
