//! Native CPU backend regressions on real, held-out printed card pixels.
//!
//! Set `OCRACLE_NATIVE_TESTS=1` and `ORT_DYLIB_PATH` after downloading the models:
//! `cargo test --release --features onnxruntime --test native_backend`.
//! Other all-feature/coverage jobs intentionally do not require native artifacts.

#![cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use ocracle::{CardOcr, InferenceBackend, Layout, Options, Recognition};

/// Exact medium recognition weights registered for the English/French profiles.
const MEDIUM_SHA256: &str = "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba";

/// Exact detector weights used by both runtime paths.
const DETECTOR_SHA256: &str = "a917b23dbd9524b465df7e922641b3ff2981623df4ded5a0234004ef2fee7cfe";

/// Resolve a fixture or registered artifact relative to this checkout.
fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

/// Require native artifacts only when this integration suite is explicitly selected.
///
/// # Errors
///
/// Returns an error if a selected run lacks its explicitly supplied native library.
fn native_tests_enabled() -> Result<bool> {
    if std::env::var("OCRACLE_NATIVE_TESTS").as_deref() != Ok("1") {
        eprintln!("Native integration suite skipped; set OCRACLE_NATIVE_TESTS=1 to require it");
        return Ok(false);
    }
    let library = std::env::var_os("ORT_DYLIB_PATH")
        .context("native integration tests require explicit ORT_DYLIB_PATH")?;
    ensure!(
        Path::new(&library).is_file(),
        "ORT_DYLIB_PATH must identify an installed native library"
    );
    Ok(true)
}

/// Load the same verified profile and optional detector for each backend.
///
/// # Errors
///
/// Returns an error for missing or mismatched model artifacts or runtime setup.
fn engine(language: &str, backend: InferenceBackend, detection: bool) -> Result<CardOcr> {
    let detector = fixture("models/text-detection.onnx");
    CardOcr::load_profile_with_backend(
        language,
        env!("CARGO_MANIFEST_DIR"),
        detection.then_some(detector.as_path()),
        backend,
        (backend == InferenceBackend::OnnxRuntime).then_some(2),
    )
}

/// Check that each complete result retains its runtime and actual weight identity.
///
/// # Errors
///
/// Returns errors for provenance changes, invalid scores or inconsistent candidates.
fn check_provenance(
    result: &Recognition,
    backend: InferenceBackend,
    language: &str,
    detection: bool,
) -> Result<()> {
    ensure!(result.model.backend == backend, "wrong inference backend");
    ensure!(result.model.language == language, "wrong printed language");
    ensure!(
        result.model.sha256 == MEDIUM_SHA256,
        "recognizer weights changed"
    );
    ensure!(
        result.model.detector_sha256.as_deref() == detection.then_some(DETECTOR_SHA256),
        "detector provenance changed"
    );
    ensure!(
        result.confidence.is_finite() && (0. ..=1.).contains(&result.confidence),
        "invalid aggregate confidence"
    );
    ensure!(
        result.candidates.len() == 1,
        "original pass should retain one candidate"
    );
    ensure!(
        result.candidates[0].text == result.text,
        "selected candidate text differs"
    );
    let serialized = serde_json::to_value(result)?;
    let expected_backend = match backend {
        InferenceBackend::Rten => "rten",
        InferenceBackend::OnnxRuntime => "onnxruntime",
    };
    ensure!(
        serialized["model"]["backend"] == expected_backend,
        "serialized runtime provenance differs"
    );
    Ok(())
}

/// Compare complete predictions and original-coordinate boxes without hiding errors.
///
/// # Errors
///
/// Returns an error for text, line ordering/count or geometry regressions.
fn compare_predictions(reference: &Recognition, observed: &Recognition) -> Result<()> {
    ensure!(
        reference.image_size == observed.image_size,
        "input dimensions changed"
    );
    ensure!(
        reference.text == observed.text,
        "backend text differs:\nRTen: {:?}\nORT: {:?}",
        reference.text,
        observed.text
    );
    ensure!(
        reference.lines.len() == observed.lines.len(),
        "backend line count differs"
    );
    let [image_width, image_height] = reference.image_size;
    for (before, after) in reference.lines.iter().zip(&observed.lines) {
        ensure!(
            before.text == after.text,
            "line text or reading order changed"
        );
        for line in [before, after] {
            let [x, y, width, height] = line.bounds;
            ensure!(
                width > 0
                    && height > 0
                    && u64::from(x) + u64::from(width) <= u64::from(image_width)
                    && u64::from(y) + u64::from(height) <= u64::from(image_height),
                "line escaped the original image coordinate system"
            );
            ensure!(
                line.confidence.is_finite() && (0. ..=1.).contains(&line.confidence),
                "invalid line confidence"
            );
        }
        let [x0, y0, w0, h0] = before.bounds.map(u64::from);
        let [x1, y1, w1, h1] = after.bounds.map(u64::from);
        for (expected, actual) in
            [x0, y0, x0 + w0, y0 + h0]
                .into_iter()
                .zip([x1, y1, x1 + w1, y1 + h1])
        {
            // Float convolution differences can move thresholded mask edges
            // slightly. Text and line ordering remain exact requirements.
            ensure!(
                expected.abs_diff(actual) <= 2,
                "line edge moved by more than two pixels"
            );
        }
    }
    Ok(())
}

#[test]
fn native_profiles_preserve_english_and_french_printed_lines() -> Result<()> {
    if !native_tests_enabled()? {
        return Ok(());
    }
    for (language, path, expected) in [
        ("en", "docs/gallery/assets/beleren-title.png", "Divination"),
        (
            "fr",
            "training/latin-eval/images/fr-m21-47-clean.png",
            "Sphère de capture",
        ),
    ] {
        let image = image::open(fixture(path))?;
        let options = Options {
            single_line: true,
            ..Options::default()
        };
        let reference =
            engine(language, InferenceBackend::Rten, false)?.recognize(&image, options)?;
        let native = engine(language, InferenceBackend::OnnxRuntime, false)?;
        let observed = native.recognize(&image, options)?;
        ensure!(reference.text == expected, "RTen printed-line gold changed");
        ensure!(
            observed.text == expected,
            "native printed-line gold changed"
        );
        compare_predictions(&reference, &observed)?;
        for (result, backend) in [
            (&reference, InferenceBackend::Rten),
            (&observed, InferenceBackend::OnnxRuntime),
        ] {
            check_provenance(result, backend, language, false)?;
            ensure!(result.lines.len() == 1, "single line fragmented");
            ensure!(
                result.lines[0].bounds == [0, 0, image.width(), image.height()],
                "single-line bounds should cover the original crop"
            );
        }
        // Reuse one engine to catch stale input/output buffers across calls.
        compare_predictions(&observed, &native.recognize(&image, options)?)?;
    }
    Ok(())
}

#[test]
fn native_detection_preserves_multiline_card_text_and_original_coordinates() -> Result<()> {
    if !native_tests_enabled()? {
        return Ok(());
    }
    let image = image::open(fixture("docs/performance/fixtures/divination-body.png"))?;
    let options = Options {
        single_line: false,
        layout: Layout::Text,
        ..Options::default()
    };
    let reference = engine("en", InferenceBackend::Rten, true)?.recognize(&image, options)?;
    let native = engine("en", InferenceBackend::OnnxRuntime, true)?;
    let observed = native.recognize(&image, options)?;
    ensure!(
        reference.lines.len() >= 4,
        "body text collapsed or was lost"
    );
    ensure!(
        reference.text.contains("Draw two cards.")
            && reference.text.contains("mentor of Mistral Isle"),
        "body fixture lost its printed rules or flavor attribution"
    );
    compare_predictions(&reference, &observed)?;
    // Explicit RTen workers use the portable detector too. Its pixel/geometry
    // contract and per-engine pool must preserve the default global-pool path.
    let detector = fixture("models/text-detection.onnx");
    let explicit_rten = CardOcr::load_profile_with_backend(
        "en",
        env!("CARGO_MANIFEST_DIR"),
        Some(&detector),
        InferenceBackend::Rten,
        Some(2),
    )?;
    let explicit_result = explicit_rten.recognize(&image, options)?;
    compare_predictions(&reference, &explicit_result)?;
    check_provenance(&explicit_result, InferenceBackend::Rten, "en", true)?;
    check_provenance(&reference, InferenceBackend::Rten, "en", true)?;
    check_provenance(&observed, InferenceBackend::OnnxRuntime, "en", true)?;

    // A detector-loaded engine must still bypass detection and omit its digest
    // when the next input explicitly requests single-line recognition.
    let title = image::open(fixture("docs/gallery/assets/beleren-title.png"))?;
    let line = native.recognize(
        &title,
        Options {
            single_line: true,
            ..Options::default()
        },
    )?;
    ensure!(
        line.text == "Divination",
        "engine reuse after detection changed text"
    );
    check_provenance(&line, InferenceBackend::OnnxRuntime, "en", false)?;
    Ok(())
}
