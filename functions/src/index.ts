import { initializeApp } from "firebase-admin/app";
import * as logger from "firebase-functions/logger";
import { type CallableRequest, onCall } from "firebase-functions/v2/https";
import { setGlobalOptions } from "firebase-functions/v2/options";

import { cancelAcutAnalysisJob, enqueueAcutAnalysisJob } from "./acutRunner.js";

const CALLABLE_AUTH_HEADER = "x-callable-context-auth";
const ORIGINAL_AUTH_HEADER = "x-original-auth";

initializeApp();

setGlobalOptions({
  region: "asia-northeast3",
  maxInstances: 10,
});

function asObject(value: unknown): Record<string, unknown> | null {
  if (value == null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function sortedKeys(value: unknown): string[] {
  const object = asObject(value);
  return object ? Object.keys(object).sort() : [];
}

function hasNonEmptyString(value: unknown): boolean {
  return typeof value === "string" && value.trim().length > 0;
}

function headerScheme(value: string | undefined): string | null {
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

function summarizeCallableRequest(request: CallableRequest<unknown>) {
  const data = asObject(request.data);
  const authHeader = request.rawRequest.header("Authorization") ?? undefined;

  return {
    auth: {
      present: request.auth != null,
      uid: request.auth?.uid ?? null,
      tokenAud: typeof request.auth?.token.aud === "string" ? request.auth.token.aud : null,
      tokenIss: typeof request.auth?.token.iss === "string" ? request.auth.token.iss : null,
      signInProvider:
        typeof request.auth?.token.firebase?.sign_in_provider === "string"
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
      instanceIdHeaderPresent:
        request.rawRequest.header("Firebase-Instance-ID-Token") != null,
      userAgent: request.rawRequest.header("User-Agent") ?? null,
      origin: request.rawRequest.header("Origin") ?? null,
      referer: request.rawRequest.header("Referer") ?? null,
    },
    data: {
      present: data != null,
      keys: data ? Object.keys(data).sort() : [],
      jobIdPresent: hasNonEmptyString(data?.jobId),
      imageCount:
        typeof data?.imageCount === "number" || typeof data?.imageCount === "string"
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

function summarizeError(error: unknown) {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      code:
        "code" in error && typeof (error as { code?: unknown }).code === "string"
          ? (error as { code: string }).code
          : null,
    };
  }
  return {
    name: typeof error,
    message: String(error),
    code: null,
  };
}

async function runCallable(
  callableName: string,
  request: CallableRequest<unknown>,
  handler: () => Promise<unknown>,
) {
  const diagnostics = summarizeCallableRequest(request);
  logger.info(`${callableName} request diagnostics`, diagnostics);

  try {
    return await handler();
  } catch (error) {
    logger.warn(`${callableName} failed`, {
      error: summarizeError(error),
      diagnostics,
    });
    throw error;
  }
}

export const enqueueAcutAnalysis = onCall(
  {
    memory: "256MiB",
    timeoutSeconds: 60,
    invoker: "public",
  },
  async (request) => {
    return runCallable("enqueueAcutAnalysis", request, async () =>
      enqueueAcutAnalysisJob(request.data, request.auth?.uid ?? null),
    );
  },
);

export const cancelAcutAnalysis = onCall(
  {
    memory: "256MiB",
    timeoutSeconds: 60,
    invoker: "public",
  },
  async (request) => {
    return runCallable("cancelAcutAnalysis", request, async () =>
      cancelAcutAnalysisJob(request.data, request.auth?.uid ?? null),
    );
  },
);
