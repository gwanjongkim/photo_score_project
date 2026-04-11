import { getStorage } from "firebase-admin/storage";

export interface StorageObjectInfo {
  path: string;
  name: string;
  size: number;
  ownerUid: string | null;
}

export function normalizeStoragePrefix(prefix: string): string {
  return prefix.replace(/^\/+|\/+$/g, "");
}

export function buildInputStoragePrefix(jobId: string): string {
  return `acut_jobs/${jobId}/inputs`;
}

export function buildOutputStoragePrefix(jobId: string): string {
  return `acut_jobs/${jobId}/outputs`;
}

function bucketForName(bucketName?: string) {
  return bucketName ? getStorage().bucket(bucketName) : getStorage().bucket();
}

export async function listObjectsUnderPrefix(
  prefix: string,
  bucketName?: string,
): Promise<StorageObjectInfo[]> {
  const normalizedPrefix = normalizeStoragePrefix(prefix);
  const [files] = await bucketForName(bucketName).getFiles({
    prefix: normalizedPrefix,
  });

  return files
    .filter((file) => !file.name.endsWith("/"))
    .map((file) => ({
      path: file.name,
      name: file.name.split("/").pop() ?? file.name,
      size: Number(file.metadata.size ?? 0),
      ownerUid:
        typeof file.metadata.metadata?.ownerUid === "string"
          ? file.metadata.metadata.ownerUid
          : null,
    }));
}

export async function assertObjectsExistUnderPrefix(
  prefix: string,
  minCount: number,
  bucketName?: string,
): Promise<StorageObjectInfo[]> {
  const objects = await listObjectsUnderPrefix(prefix, bucketName);
  if (objects.length < minCount) {
    throw new Error(
      `Expected at least ${minCount} uploaded files under ${normalizeStoragePrefix(prefix)}, found ${objects.length}.`,
    );
  }
  return objects;
}

export async function assertObjectsOwnedByUser(
  prefix: string,
  authUid: string,
  bucketName?: string,
): Promise<StorageObjectInfo[]> {
  const objects = await listObjectsUnderPrefix(prefix, bucketName);
  const nonOwnedObjects = objects.filter((object) => object.ownerUid !== authUid);
  if (nonOwnedObjects.length > 0) {
    throw new Error(
      `Uploaded files under ${normalizeStoragePrefix(prefix)} were not all owned by ${authUid}.`,
    );
  }
  return objects;
}
