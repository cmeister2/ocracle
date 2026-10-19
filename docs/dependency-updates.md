# Dependency updates

Renovate extends `github>cmeister2/renovate`. Most dependencies keep the shared
automation policy; the repository adds these compatibility boundaries:

- Python stays within 3.12 because the native environment locks identify CPython
  3.12 wheel bytes and the current ONNX Runtime package lacks Python 3.15 support.
  Validate distributions and regenerate a separate environment before moving
  to another Python minor; patch updates remain allowed.
- Ocrs 0.13 exposes RTen 0.26 models and geometry types. Keep the application's
  direct RTen family compatible until a coordinated Ocrs/runtime upgrade passes
  the Rust, MSRV and real-image checks.
- `training/experimental_paddle/runner/` is historical export evidence. Its
  manifest, lockfile and source hashes identify the executable used for the
  recorded native/ONNX/RTen comparison. Exclude this directory from discovery
  and lock maintenance; assess new probe dependencies in a separate experiment
  instead of rewriting the old report's identities.

Renovate's `ignorePaths` array replaces inherited values, so its explicit list
retains all eight standard exclusions from the shared preset and adds only the
historical runner. Root Cargo and npm dependency updates remain enabled.
Changes still require the complete `Success` CI check.
