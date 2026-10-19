import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { prepareRelease } from "./prepare-release.mjs";

const manifest = '[package]\nname = "ocracle"\nversion = "0.0.0"\nedition = "2024"\n';
const lock = 'version = 4\n\n[[package]]\nname = "dependency"\nversion = "0.0.0"\n\n[[package]]\nname = "ocracle"\nversion = "0.0.0"\n';

async function fixture(callback) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "ocracle-release-"));
  try {
    await writeFile(path.join(directory, "Cargo.toml"), manifest);
    await writeFile(path.join(directory, "Cargo.lock"), lock);
    await callback(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

test("release preparation synchronizes ocracle without changing dependency sentinels", async () => {
  await fixture(async (directory) => {
    await prepareRelease("1.2.3", directory);
    assert.equal(await readFile(path.join(directory, "Cargo.toml"), "utf8"),
      manifest.replace('version = "0.0.0"', 'version = "1.2.3"'));
    assert.equal(await readFile(path.join(directory, "Cargo.lock"), "utf8"),
      lock.replace('name = "ocracle"\nversion = "0.0.0"', 'name = "ocracle"\nversion = "1.2.3"'));
  });
});

test("missing lockfile sentinel fails before modifying the manifest", async () => {
  await fixture(async (directory) => {
    await writeFile(path.join(directory, "Cargo.lock"), lock.replace('name = "ocracle"', 'name = "other"'));
    await assert.rejects(prepareRelease("1.2.3", directory), /Cargo\.lock.*sentinel/);
    assert.equal(await readFile(path.join(directory, "Cargo.toml"), "utf8"), manifest);
  });
});

test("malformed, shell-sensitive, and sentinel versions are rejected without writes", async () => {
  await fixture(async (directory) => {
    for (const version of [undefined, "0.0.0", "01.2.3", "1.2", "1.2.3-01", '1.2.3"', "$(false)"]) {
      await assert.rejects(prepareRelease(version, directory), /SemVer/);
    }
    assert.equal(await readFile(path.join(directory, "Cargo.toml"), "utf8"), manifest);
    assert.equal(await readFile(path.join(directory, "Cargo.lock"), "utf8"), lock);
  });
});

test("prerelease and run-number build versions can be prepared", async () => {
  for (const version of ["1.2.3-rc.1+build.42", "0.0.123456789"]) {
    await fixture(async (directory) => {
      await prepareRelease(version, directory);
      assert.ok((await readFile(path.join(directory, "Cargo.toml"), "utf8")).includes(`version = "${version}"`));
    });
  }
});
