"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.normalizeStoragePrefix = normalizeStoragePrefix;
exports.buildInputStoragePrefix = buildInputStoragePrefix;
exports.buildOutputStoragePrefix = buildOutputStoragePrefix;
exports.listObjectsUnderPrefix = listObjectsUnderPrefix;
exports.assertObjectsExistUnderPrefix = assertObjectsExistUnderPrefix;
exports.assertObjectsOwnedByUser = assertObjectsOwnedByUser;
const storage_1 = require("firebase-admin/storage");
function normalizeStoragePrefix(prefix) {
    return prefix.replace(/^\/+|\/+$/g, "");
}
function buildInputStoragePrefix(jobId) {
    return `acut_jobs/${jobId}/inputs`;
}
function buildOutputStoragePrefix(jobId) {
    return `acut_jobs/${jobId}/outputs`;
}
function bucketForName(bucketName) {
    return bucketName ? (0, storage_1.getStorage)().bucket(bucketName) : (0, storage_1.getStorage)().bucket();
}
async function listObjectsUnderPrefix(prefix, bucketName) {
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
        ownerUid: typeof file.metadata.metadata?.ownerUid === "string"
            ? file.metadata.metadata.ownerUid
            : null,
    }));
}
async function assertObjectsExistUnderPrefix(prefix, minCount, bucketName) {
    const objects = await listObjectsUnderPrefix(prefix, bucketName);
    if (objects.length < minCount) {
        throw new Error(`Expected at least ${minCount} uploaded files under ${normalizeStoragePrefix(prefix)}, found ${objects.length}.`);
    }
    return objects;
}
async function assertObjectsOwnedByUser(prefix, authUid, bucketName) {
    const objects = await listObjectsUnderPrefix(prefix, bucketName);
    const nonOwnedObjects = objects.filter((object) => object.ownerUid !== authUid);
    if (nonOwnedObjects.length > 0) {
        throw new Error(`Uploaded files under ${normalizeStoragePrefix(prefix)} were not all owned by ${authUid}.`);
    }
    return objects;
}
