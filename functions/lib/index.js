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
exports.cancelAcutAnalysis = exports.enqueueAcutAnalysis = void 0;
const app_1 = require("firebase-admin/app");
const logger = __importStar(require("firebase-functions/logger"));
const https_1 = require("firebase-functions/v2/https");
const options_1 = require("firebase-functions/v2/options");
const acutRunner_js_1 = require("./acutRunner.js");
const CALLABLE_AUTH_HEADER = "x-callable-context-auth";
const ORIGINAL_AUTH_HEADER = "x-original-auth";
(0, app_1.initializeApp)();
(0, options_1.setGlobalOptions)({
    region: "asia-northeast3",
    maxInstances: 10,
});
function asObject(value) {
    if (value == null || typeof value !== "object" || Array.isArray(value)) {
        return null;
    }
    return value;
}
function sortedKeys(value) {
    const object = asObject(value);
    return object ? Object.keys(object).sort() : [];
}
function hasNonEmptyString(value) {
    return typeof value === "string" && value.trim().length > 0;
}
function headerScheme(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    if (trimmed.length === 0) {
        return null;
    }
    const [scheme] = trimmed.split(/\s+/, 1);
    return scheme ?? null;
}
function summarizeCallableRequest(request) {
    const data = asObject(request.data);
    const authHeader = request.rawRequest.header("Authorization") ?? undefined;
    return {
        auth: {
            present: request.auth != null,
            uid: request.auth?.uid ?? null,
            tokenAud: typeof request.auth?.token.aud === "string" ? request.auth.token.aud : null,
            tokenIss: typeof request.auth?.token.iss === "string" ? request.auth.token.iss : null,
            signInProvider: typeof request.auth?.token.firebase?.sign_in_provider === "string"
                ? request.auth.token.firebase.sign_in_provider
                : null,
        },
        app: {
            present: request.app != null,
            appId: request.app?.appId ?? null,
            alreadyConsumed: request.app?.alreadyConsumed ?? null,
        },
        instanceIdTokenPresent: request.instanceIdToken != null,
        acceptsStreaming: request.acceptsStreaming,
        http: {
            method: request.rawRequest.method,
            originalUrl: request.rawRequest.originalUrl ?? request.rawRequest.path ?? null,
            authHeaderPresent: authHeader != null,
            authHeaderScheme: headerScheme(authHeader),
            callableAuthHeaderPresent: request.rawRequest.header(CALLABLE_AUTH_HEADER) != null,
            originalAuthHeaderPresent: request.rawRequest.header(ORIGINAL_AUTH_HEADER) != null,
            appCheckHeaderPresent: request.rawRequest.header("X-Firebase-AppCheck") != null,
            instanceIdHeaderPresent: request.rawRequest.header("Firebase-Instance-ID-Token") != null,
            userAgent: request.rawRequest.header("User-Agent") ?? null,
            origin: request.rawRequest.header("Origin") ?? null,
            referer: request.rawRequest.header("Referer") ?? null,
        },
        data: {
            present: data != null,
            keys: data ? Object.keys(data).sort() : [],
            jobIdPresent: hasNonEmptyString(data?.jobId),
            imageCount: typeof data?.imageCount === "number" || typeof data?.imageCount === "string"
                ? data.imageCount
                : null,
            inputStoragePrefixPresent: hasNonEmptyString(data?.inputStoragePrefix),
            outputStoragePrefixPresent: hasNonEmptyString(data?.outputStoragePrefix),
            inputFilesCount: Array.isArray(data?.inputFiles) ? data.inputFiles.length : null,
            pipelineConfigKeys: sortedKeys(data?.pipelineConfig),
            clientContextKeys: sortedKeys(data?.clientContext),
        },
    };
}
function summarizeError(error) {
    if (error instanceof Error) {
        return {
            name: error.name,
            message: error.message,
            code: "code" in error && typeof error.code === "string"
                ? error.code
                : null,
        };
    }
    return {
        name: typeof error,
        message: String(error),
        code: null,
    };
}
async function runCallable(callableName, request, handler) {
    const diagnostics = summarizeCallableRequest(request);
    logger.info(`${callableName} request diagnostics`, diagnostics);
    try {
        return await handler();
    }
    catch (error) {
        logger.warn(`${callableName} failed`, {
            error: summarizeError(error),
            diagnostics,
        });
        throw error;
    }
}
exports.enqueueAcutAnalysis = (0, https_1.onCall)({
    memory: "256MiB",
    timeoutSeconds: 60,
    invoker: "public",
}, async (request) => {
    return runCallable("enqueueAcutAnalysis", request, async () => (0, acutRunner_js_1.enqueueAcutAnalysisJob)(request.data, request.auth?.uid ?? null));
});
exports.cancelAcutAnalysis = (0, https_1.onCall)({
    memory: "256MiB",
    timeoutSeconds: 60,
    invoker: "public",
}, async (request) => {
    return runCallable("cancelAcutAnalysis", request, async () => (0, acutRunner_js_1.cancelAcutAnalysisJob)(request.data, request.auth?.uid ?? null));
});
