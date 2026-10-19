/** Release Rust packages and GitHub notes from Conventional Commit history. */
export default {
  branches: ["main", "master"],
  tagFormat: "${version}",
  plugins: [
    "@semantic-release/commit-analyzer",
    ["@semantic-release/release-notes-generator", { preset: "conventionalcommits" }],
    [
      "@semantic-release/exec",
      {
        verifyConditionsCmd:
          'node -e "if (!process.env.CARGO_REGISTRY_TOKEN) { console.error(\'Set the CARGO_REGISTRY_TOKEN repository secret before publishing to crates.io.\'); process.exit(1); }"',
        prepareCmd: "node scripts/prepare-release.mjs ${nextRelease.version}",
        publishCmd: "cargo publish --locked --allow-dirty",
      },
    ],
    [
      "@semantic-release/github",
      { successComment: false, failComment: false, releasedLabels: false },
    ],
  ],
};
