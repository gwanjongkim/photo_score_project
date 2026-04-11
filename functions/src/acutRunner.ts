import * as logger from "firebase-functions/logger";
import { HttpsError } from "firebase-functions/v2/https";

import {
  type AcutInputFile,
  createOrUpdateQueuedJob,
  JobMutationError,
  type QueueJobPayload,
  requestJobCancellation,
} from "./firestoreJobs.js";
import {
  assertObjectsExistUnderPrefix,
  assertObjectsOwnedByUser,
  buildOutputStoragePrefix,
  normalizeStoragePrefix,
} from "./storageUtils.js";

class ValidationError extends Error {}

function asAuthenticatedUid(authUid: string | null): string {
  if (authUid == null || authUid.trim().length == 0) {
    logger.warn("A-cut callable reached runner without request.auth.uid", {
      authUidPresent: authUid != null,
      authUidTrimmedLength: typeof authUid === "string" ? authUid.trim().length : null,
    });
    throw new HttpsError(
      "unauthenticated",
      "Firebase authentication is required before starting or cancelling A-cut analysis.",
    );
  }
  return authUid.trim();
}

function asNonEmptyString(value: unknown, fieldName: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ValidationError(`${fieldName} must be a non-empty string.`);
  }
  return value.trim();
}

function asPositiveInt(value: unknown, fieldName: string, fallback?: number): number {
  const numeric = value == null ? fallback : Number(value);
  if (!Number.isInteger(numeric) || (numeric ?? 0) <= 0) {
    throw new ValidationError(`${fieldName} must be a positive integer.`);
  }
  return numeric as number;
}

function asNonNegativeInt(value: unknown, fieldName: string, fallback?: number): number {
  const numeric = value == null ? fallback : Number(value);
  if (!Number.isInteger(numeric) || (numeric ?? 0) < 0) {
    throw new ValidationError(`${fieldName} must be a non-negative integer.`);
  }
  return numeric as number;
}

function asBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function asStringMap(value: unknown): Record<string, unknown> {
  if (value == null) {
    return {};
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new ValidationError("Expected an object payload.");
  }
  return value as Record<string, unknown>;
}

function parseInputFiles(value: unknown): AcutInputFile[] {
  if (value == null) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new ValidationError("inputFiles must be an array when provided.");
  }

  return value.map((entry, index) => {
    const map = asStringMap(entry);
    return {
      uploadFileName: asNonEmptyString(
        map.uploadFileName,
        `inputFiles[${index}].uploadFileName`,
      ),
      displayName: asNonEmptyString(
        map.displayName,
        `inputFiles[${index}].displayName`,
      ),
      storagePath: asNonEmptyString(
        map.storagePath,
        `inputFiles[${index}].storagePath`,
      ),
      selectedIndex: asNonNegativeInt(
        map.selectedIndex,
        `inputFiles[${index}].selectedIndex`,
        0,
      ),
    };
  });
}

function assertInputFilesMatchRequest(
  inputFiles: AcutInputFile[],
  {
    imageCount,
    inputStoragePrefix,
  }: {
    imageCount: number;
    inputStoragePrefix: string;
  },
) {
  if (inputFiles.length > 0 && inputFiles.length !== imageCount) {
    throw new ValidationError(
      `inputFiles length ${inputFiles.length} must match imageCount ${imageCount}.`,
    );
  }

  for (const file of inputFiles) {
    if (!file.storagePath.startsWith(`${inputStoragePrefix}/`)) {
      throw new ValidationError(
        `inputFiles storagePath ${file.storagePath} must stay under ${inputStoragePrefix}.`,
      );
    }
    const fileName = file.storagePath.split("/").pop();
    if (fileName !== file.uploadFileName) {
      throw new ValidationError(
        `inputFiles uploadFileName ${file.uploadFileName} must match the trailing file name in storagePath ${file.storagePath}.`,
      );
    }
  }
}

export async function enqueueAcutAnalysisJob(
  rawData: unknown,
  authUid: string | null,
  collectionName = "jobs",
) {
  try {
    const authenticatedUid = asAuthenticatedUid(authUid);
    const data = asStringMap(rawData);
    const jobId = asNonEmptyString(data.jobId, "jobId");
    const imageCount = asPositiveInt(data.imageCount, "imageCount");
    const inputStoragePrefix = normalizeStoragePrefix(
      asNonEmptyString(data.inputStoragePrefix, "inputStoragePrefix"),
    );
    const outputStoragePrefix = normalizeStoragePrefix(
      typeof data.outputStoragePrefix === "string" && data.outputStoragePrefix.trim().length > 0
        ? data.outputStoragePrefix
        : buildOutputStoragePrefix(jobId),
    );
    const topK = asPositiveInt(data.topK, "topK", 5);
    const enableDiversity = asBoolean(data.enableDiversity, false);
    const inputFiles = parseInputFiles(data.inputFiles);
    const pipelineConfig = asStringMap(data.pipelineConfig);
    const clientContext = asStringMap(data.clientContext);

    assertInputFilesMatchRequest(inputFiles, {
      imageCount,
      inputStoragePrefix,
    });

    await assertObjectsExistUnderPrefix(inputStoragePrefix, imageCount);
    await assertObjectsOwnedByUser(inputStoragePrefix, authenticatedUid);

    const payload: QueueJobPayload = {
      jobId,
      imageCount,
      inputStoragePrefix,
      outputStoragePrefix,
      topK,
      enableDiversity,
      userId: authenticatedUid,
      inputFiles,
      pipelineConfig: {
        ...pipelineConfig,
        topK,
        enableDiversity,
      },
      clientContext,
    };

    const result = await createOrUpdateQueuedJob(payload, collectionName);
    return {
      ...result,
      workerMode: "external_python_worker",
      inputStoragePrefix,
      outputs: {
        appResultsJsonPath: `${outputStoragePrefix}/app_results.json`,
        topKSummaryJsonPath: `${outputStoragePrefix}/top_k_summary.json`,
        reviewSheetCsvPath: `${outputStoragePrefix}/review_sheet.csv`,
      },
    };
  } catch (error) {
    if (error instanceof ValidationError) {
      throw new HttpsError("invalid-argument", error.message);
    }
    if (error instanceof JobMutationError) {
      throw new HttpsError(error.code, error.message);
    }
    if (error instanceof HttpsError) {
      throw error;
    }
    throw new HttpsError(
      "internal",
      error instanceof Error ? error.message : "Failed to enqueue A-cut analysis job.",
    );
  }
}

export async function cancelAcutAnalysisJob(
  rawData: unknown,
  authUid: string | null,
  collectionName = "jobs",
) {
  try {
    const authenticatedUid = asAuthenticatedUid(authUid);
    const data = asStringMap(rawData);
    const jobId = asNonEmptyString(data.jobId, "jobId");
    return await requestJobCancellation(jobId, authenticatedUid, collectionName);
  } catch (error) {
    if (error instanceof ValidationError) {
      throw new HttpsError("invalid-argument", error.message);
    }
    if (error instanceof JobMutationError) {
      throw new HttpsError(error.code, error.message);
    }
    if (error instanceof HttpsError) {
      throw error;
    }
    throw new HttpsError(
      "internal",
      error instanceof Error ? error.message : "Failed to cancel A-cut analysis job.",
    );
  }
}
