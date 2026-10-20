//! Host-owned ONNX Runtime environment lifetime regression.
//!
//! Run separately from `native_backend`: ort permits only one environment per
//! process, and this test must create the host environment before ocracle loads.
//! Set `OCRACLE_NATIVE_TESTS=1` and `ORT_DYLIB_PATH` after downloading the models.

#![cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, ensure};
use ocracle::{CardOcr, InferenceBackend, Layout, ModelConfig, Options};
use ort::{logging::LoggerFunction, session::Session, value::TensorRef};

/// Tiny ONNX identity graph with one float32 `[1,1,2,2]` input and output.
const IDENTITY_ONNX: &[u8] = b"\x08\x08\x3a\x68\x0a\x19\x0a\x05\x69\x6e\x70\x75\x74\x12\x06\x6f\x75\x74\x70\x75\x74\x22\x08\x49\x64\x65\x6e\x74\x69\x74\x79\x12\x08\x69\x64\x65\x6e\x74\x69\x74\x79\x5a\x1f\x0a\x05\x69\x6e\x70\x75\x74\x12\x16\x0a\x14\x08\x01\x12\x10\x0a\x02\x08\x01\x0a\x02\x08\x01\x0a\x02\x08\x02\x0a\x02\x08\x02\x62\x20\x0a\x06\x6f\x75\x74\x70\x75\x74\x12\x16\x0a\x14\x08\x01\x12\x10\x0a\x02\x08\x01\x0a\x02\x08\x01\x0a\x02\x08\x02\x0a\x02\x08\x02\x42\x04\x0a\x00\x10\x0d";

/// Resolve a fixture or registered artifact relative to this checkout.
fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

/// Resolve the required library only when native regressions are selected.
///
/// # Errors
///
/// Returns an error if an explicitly selected run has no installed native library.
fn native_library() -> Result<Option<PathBuf>> {
    if std::env::var("OCRACLE_NATIVE_TESTS").as_deref() != Ok("1") {
        eprintln!("Native environment test skipped; set OCRACLE_NATIVE_TESTS=1 to require it");
        return Ok(None);
    }
    let library = std::env::var_os("ORT_DYLIB_PATH")
        .context("native integration tests require explicit ORT_DYLIB_PATH")?;
    let library = PathBuf::from(library);
    ensure!(
        library.is_file(),
        "ORT_DYLIB_PATH must identify an installed native library"
    );
    Ok(Some(library))
}

/// Exercise an independent host session after the OCR engines have been dropped.
///
/// # Errors
///
/// Returns an error if inference fails or the identity graph changes its input.
fn check_host_session(session: &mut Session) -> Result<()> {
    let input = [0.25_f32, -1., 0.5, 2.];
    let tensor = TensorRef::from_array_view(([1, 1, 2, 2], &input[..]))?;
    let outputs = session.run(ort::inputs![tensor])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    ensure!(
        shape.as_ref() == [1, 1, 2, 2],
        "host session output shape changed"
    );
    ensure!(data == input, "host session lost its environment");
    Ok(())
}

#[test]
fn host_environment_is_shared_and_released_with_its_last_session() -> Result<()> {
    let Some(library) = native_library()? else {
        return Ok(());
    };
    // The environment owns this logger. Its weak reference observes teardown
    // without retaining an environment handle or relying on native pointers.
    let logger: LoggerFunction = Arc::new(|_, _, _, _, _| {});
    let logger_lifetime = Arc::downgrade(&logger);
    let environment = ort::init_from(&library)?.with_logger(logger).build()?;
    let mut host_session = Session::builder(&environment)?
        .with_intra_threads(1)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .commit_from_memory(IDENTITY_ONNX)?;

    let model = fixture("models/ppocr-v6-medium.onnx");
    let detector = fixture("models/text-detection.onnx");
    let config: ModelConfig =
        serde_json::from_slice(&std::fs::read(fixture("models/paddle-v6-medium/en.json"))?)?;

    let missing_directory = tempfile::tempdir()?;
    let missing_model = CardOcr::load_with_onnxruntime_environment(
        missing_directory.path().join("missing.onnx"),
        None,
        config.clone(),
        &environment,
        Some(1),
    )
    .err()
    .context("missing recognition model unexpectedly loaded")?;
    ensure!(
        missing_model.to_string().contains("read recognition model"),
        "missing-model diagnostic lost its context"
    );
    let invalid_threads = CardOcr::load_profile_with_onnxruntime_environment(
        "en",
        env!("CARGO_MANIFEST_DIR"),
        None,
        &environment,
        Some(0),
    )
    .err()
    .context("zero inference threads unexpectedly accepted")?;
    ensure!(
        invalid_threads.to_string().contains("thread count"),
        "invalid-thread diagnostic lost its context"
    );
    let default_environment = CardOcr::load_with_backend(
        &model,
        None,
        config.clone(),
        InferenceBackend::OnnxRuntime,
        Some(1),
    )
    .err()
    .context("default loader unexpectedly replaced the host environment")?;
    ensure!(
        format!("{default_environment:#}").contains("load_with_onnxruntime_environment"),
        "default loader did not explain how to share the host environment"
    );

    let direct = CardOcr::load_with_onnxruntime_environment(
        &model,
        Some(&detector),
        config,
        &environment,
        Some(2),
    )?;
    let profile = CardOcr::load_profile_with_onnxruntime_environment(
        "en",
        env!("CARGO_MANIFEST_DIR"),
        None,
        &environment,
        Some(2),
    )?;
    drop(environment);
    ensure!(
        logger_lifetime.upgrade().is_some(),
        "dropping the original handle released a live environment"
    );

    let title = image::open(fixture("docs/gallery/assets/beleren-title.png"))?;
    let options = Options {
        single_line: true,
        ..Options::default()
    };
    for engine in [&direct, &profile] {
        let result = engine.recognize(&title, options)?;
        ensure!(
            result.text == "Divination",
            "shared environment changed OCR"
        );
        ensure!(
            result.model.backend == InferenceBackend::OnnxRuntime,
            "host environment did not select ONNX Runtime"
        );
    }
    let body = image::open(fixture("docs/performance/fixtures/divination-body.png"))?;
    let result = direct.recognize(
        &body,
        Options {
            layout: Layout::Text,
            ..Options::default()
        },
    )?;
    ensure!(
        result.text.contains("Draw two cards.") && result.lines.len() >= 4,
        "detector did not retain the shared environment"
    );

    drop(direct);
    drop(profile);
    check_host_session(&mut host_session)?;
    ensure!(
        logger_lifetime.upgrade().is_some(),
        "dropping OCR engines released the host's environment"
    );
    drop(host_session);
    ensure!(
        logger_lifetime.upgrade().is_none(),
        "ocracle retained the host environment after its last session was dropped"
    );
    Ok(())
}
