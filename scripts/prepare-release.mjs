import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

/** Validate SemVer before interpolating a version into Cargo's TOML files. */
function validVersion(version) {
  const match = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/.exec(version);
  return match !== null && (!match[4] || match[4].split(".").every(
    (identifier) => !/^\d+$/.test(identifier) || !/^0\d/.test(identifier),
  ));
}

/** Replace only ocracle's checked-in sentinel, leaving dependency versions alone. */
function replaceVersion(contents, header, version, filename) {
  const blocks = contents.split(/(?=^\[)/m);
  const index = blocks.findIndex((block) =>
    block.startsWith(`${header}\n`) && /^name\s*=\s*"ocracle"\s*$/m.test(block));
  if (index === -1 || !/^version\s*=\s*"0\.0\.0"\s*$/m.test(blocks[index])) {
    throw new Error(`${filename} must contain ocracle's version = "0.0.0" sentinel`);
  }
  blocks[index] = blocks[index].replace(/^version\s*=\s*"0\.0\.0"/m, `version = "${version}"`);
  return blocks.join("");
}

/** Update the manifest and lockfile together after validating both inputs. */
export async function prepareRelease(version, directory = process.cwd()) {
  if (typeof version !== "string" || !validVersion(version) || version === "0.0.0") {
    throw new Error("Expected a non-sentinel SemVer release or build version");
  }
  const manifestPath = path.join(directory, "Cargo.toml");
  const lockPath = path.join(directory, "Cargo.lock");
  const [manifest, lock] = await Promise.all([
    readFile(manifestPath, "utf8"), readFile(lockPath, "utf8"),
  ]);
  const nextManifest = replaceVersion(manifest, "[package]", version, "Cargo.toml");
  const nextLock = replaceVersion(lock, "[[package]]", version, "Cargo.lock");
  await writeFile(manifestPath, nextManifest);
  await writeFile(lockPath, nextLock);
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  try {
    await prepareRelease(process.argv[2]);
    console.log(`Prepared ocracle ${process.argv[2]}`);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
