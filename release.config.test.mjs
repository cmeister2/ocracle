import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import test from "node:test";
import configuration from "./release.config.js";

// Resolve the module from semantic-release's dependency graph, as releases do.
const require = createRequire(import.meta.url);
const releaseRequire = createRequire(require.resolve("semantic-release"));
const generatorPath = releaseRequire.resolve("@semantic-release/release-notes-generator");
const { generateNotes } = await import(pathToFileURL(generatorPath).href);
const [, options] = configuration.plugins.find(
  (plugin) => Array.isArray(plugin) && plugin[0] === "@semantic-release/release-notes-generator",
);

test("the real release graph renders Conventional Commits 10 notes and links", async () => {
  const notes = await generateNotes(options, {
    cwd: process.cwd(),
    options: { repositoryUrl: "https://github.com/cmeister2/ocracle.git" },
    branch: { name: "main" },
    lastRelease: { version: "1.1.0", gitTag: "1.1.0" },
    nextRelease: { version: "2.0.0", gitTag: "2.0.0" },
    commits: [
      { hash: "a".repeat(40), message: "feat: recognise italic reminder text\n\nCloses #42" },
      { hash: "b".repeat(40), message: "fix: recover motion-blurred mana symbols" },
      {
        hash: "c".repeat(40),
        message:
          "feat: return calibrated confidence\n\nBREAKING CHANGE: confidence now uses a probability from zero to one.",
      },
    ],
    logger: { log() {} },
  });

  assert.match(notes, /### Features/);
  assert.match(notes, /### Bug Fixes/);
  assert.match(notes, /^### .*BREAKING CHANGES$/m);
  assert.match(notes, /recognise italic reminder text/);
  assert.match(notes, /recover motion-blurred mana symbols/);
  assert.match(notes, /return calibrated confidence/);
  assert.match(notes, /confidence now uses a probability from zero to one\./);
  assert.match(notes, /https:\/\/github\.com\/cmeister2\/ocracle\/compare\/1\.1\.0\.\.\.2\.0\.0/);
  assert.match(notes, /\[#42\]\(https:\/\/github\.com\/cmeister2\/ocracle\/issues\/42\)/);
  assert.match(notes, /https:\/\/github\.com\/cmeister2\/ocracle\/commit\/a{40}/);
  assert.match(notes, /https:\/\/github\.com\/cmeister2\/ocracle\/commit\/b{40}/);
  assert.match(notes, /https:\/\/github\.com\/cmeister2\/ocracle\/commit\/c{40}/);
});

test("release tags are plain SemVer and npm publishing is excluded", () => {
  assert.equal(configuration.tagFormat, "${version}");
  assert.deepEqual(configuration.branches, ["main", "master"]);
  assert.ok(!configuration.plugins.some((plugin) =>
    (Array.isArray(plugin) ? plugin[0] : plugin) === "@semantic-release/npm"));
});
