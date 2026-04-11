"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.enqueueAcutAnalysisJob = enqueueAcutAnalysisJob;
exports.cancelAcutAnalysisJob = cancelAcutAnalysisJob;
const logger = __importStar(require("firebase-functions/logger"));
const https_1 = require("firebase-functions/v2/https");
const firestoreJobs_js_1 = require("./firestoreJobs.js");
const storageUtils_js_1 = require("./storageUtils.js");
class ValidationError extends Error {
}
function asAuthenticatedUid(authUid) {
    if (authUid == null || authUid.trim().length == 0) {
        logger.warn("A-cut callable reached runner without request.auth.uid", {
            authUidPresent: authUid != null,
            authUidTrimmedLength: typeof authUid === "string" ? authUid.trim().length : null,
        });
        throw new https_1.HttpsError("unauthenticated", "Firebase authentication is required before starting or cancelling A-cut analysis.");
    }
    return authUid.trim();
}
function asNonEmptyString(value, fieldName) {
    if (typeof value !== "string" || value.trim().length === 0) {
        throw new ValidationError(`${fieldName} must be a non-empty string.`);
    }
    return value.trim();
}
function asPositiveInt(value, fieldName, fallback) {
    const numeric = value == null ? fallback : Number(value);
    if (!Number.isInteger(numeric) || (numeric ?? 0) <= 0) {
        throw new ValidationError(`${fieldName} must be a positive integer.`);
    }
    return numeric;
}
function asNonNegativeInt(value, fieldName, fallback) {
    const numeric = value == null ? fallback : Number(value);
    if (!Number.isInteger(numeric) || (numeric ?? 0) < 0) {
        throw new ValidationError(`${fieldName} must be a non-negative integer.`);
    }
    return numeric;
}
function asBoolean(value, fallback = false) {
    return typeof value === "boolean" ? value : fallback;
}
function asStringMap(value) {
    if (value == null) {
        return {};
    }
    if (typeof value !== "object" || Array.isArray(value)) {
        throw new ValidationError("Expected an object payload.");
    }
    return value;
}
function parseInputFiles(value) {
    if (value == null) {
        return [];
    }
    if (!Array.isArray(value)) {
        throw new ValidationError("inputFiles must be an array when provided.");
    }
    return value.map((entry, index) => {
        const map = asStringMap(entry);
        return {
            uploadFileName: asNonEmptyString(map.uploadFileName, `inputFiles[${index}].uploadFileName`),
            displayName: asNonEmptyString(map.displayName, `inputFiles[${index}].displayName`),
            storagePath: asNonEmptyString(map.storagePath, `inputFiles[${index}].storagePath`),
            selectedIndex: asNonNegativeInt(map.selectedIndex, `inputFiles[${index}].selectedIndex`, 0),
        };
    });
}
function assertInputFilesMatchRequest(inputFiles, { imageCount, inputStoragePrefix, }) {
    if (inputFiles.length > 0 && inputFiles.length !== imageCount) {
        throw new ValidationError(`inputFiles length ${inputFiles.length} must match imageCount ${imageCount}.`);
    }
    for (const file of inputFiles) {
        if (!file.storagePath.startsWith(`${inputStoragePrefix}/`)) {
            throw new ValidationError(`inputFiles storagePath ${file.storagePath} must stay under ${inputStoragePrefix}.`);
        }
        const fileName = file.storagePath.split("/").pop();
        if (fileName !== file.uploadFileName) {
            throw new ValidationError(`inputFiles uploadFileName ${file.uploadFileName} must match the trailing file name in storagePath ${file.storagePath}.`);
        }
    }
}
async function enqueueAcutAnalysisJob(rawData, authUid, collectionName = "jobs") {
    try {
        const authenticatedUid = asAuthenticatedUid(authUid);
        const data = asStringMap(rawData);
        const jobId = asNonEmptyString(data.jobId, "jobId");
        const imageCount = asPositiveInt(data.imageCount, "imageCount");
        const inputStoragePrefix = (0, storageUtils_js_1.normalizeStoragePrefix)(asNonEmptyString(data.inputStoragePrefix, "inputStoragePrefix"));
        const outputStoragePrefix = (0, storageUtils_js_1.normalizeStoragePrefix)(typeof data.outputStoragePrefix === "string" && data.outputStoragePrefix.trim().length > 0
            ? data.outputStoragePrefix
            : (0, storageUtils_js_1.buildOutputStoragePrefix)(jobId));
        const topK = asPositiveInt(data.topK, "topK", 5);
        const enableDiversity = asBoolean(data.enableDiversity, false);
        const inputFiles = parseInputFiles(data.inputFiles);
        const pipelineConfig = asStringMap(data.pipelineConfig);
        const clientContext = asStringMap(data.clientContext);
        assertInputFilesMatchRequest(inputFiles, {
            imageCount,
            inputStoragePrefix,
        });
        await (0, storageUtils_js_1.assertObjectsExistUnderPrefix)(inputStoragePrefix, imageCount);
        await (0, storageUtils_js_1.assertObjectsOwnedByUser)(inputStoragePrefix, authenticatedUid);
        const payload = {
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
        const result = await (0, firestoreJobs_js_1.createOrUpdateQueuedJob)(payload, collectionName);
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
    }
    catch (error) {
        if (error instanceof ValidationError) {
            throw new https_1.HttpsError("invalid-argument", error.message);
        }
        if (error instanceof firestoreJobs_js_1.JobMutationError) {
            throw new https_1.HttpsError(error.code, error.message);
        }
        if (error instanceof https_1.HttpsError) {
            throw error;
        }
        throw new https_1.HttpsError("internal", error instanceof Error ? error.message : "Failed to enqueue A-cut analysis job.");
    }
}
async function cancelAcutAnalysisJob(rawData, authUid, collectionName = "jobs") {
    try {
        const authenticatedUid = asAuthenticatedUid(authUid);
        const data = asStringMap(rawData);
        const jobId = asNonEmptyString(data.jobId, "jobId");
        return await (0, firestoreJobs_js_1.requestJobCancellation)(jobId, authenticatedUid, collectionName);
    }
    catch (error) {
        if (error instanceof ValidationError) {
            throw new https_1.HttpsError("invalid-argument", error.message);
        }
        if (error instanceof firestoreJobs_js_1.JobMutationError) {
            throw new https_1.HttpsError(error.code, error.message);
        }
        if (error instanceof https_1.HttpsError) {
            throw error;
        }
        throw new https_1.HttpsError("internal", error instanceof Error ? error.message : "Failed to cancel A-cut analysis job.");
    }
}
