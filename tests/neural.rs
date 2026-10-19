//! End-to-end inference regressions using separately downloaded neural weights.
//!
//! Run with `cargo test --release --test neural -- --ignored` after downloading
//! the models. The small scan crops are held out from specialized training.

use std::path::Path;

use image::{DynamicImage, RgbImage, imageops};
use ocracle::{CardOcr, ModelConfig, Options, Passes};

/// Resolve a repository fixture or downloaded model.
fn fixture(path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

/// Load the official base recognizer and optional detector.
fn engine(detection: bool) -> CardOcr {
    CardOcr::load(
        fixture("models/text-recognition-base.rten"),
        detection
            .then(|| fixture("models/text-detection.onnx"))
            .as_deref(),
        ModelConfig::default(),
    )
    .unwrap()
}

#[test]
#[ignore = "requires downloaded models; run training/download_models.py first"]
fn reads_real_beleren_title_and_preserves_candidates_and_coordinates() {
    let image = image::open(fixture("docs/gallery/assets/beleren-title.png")).unwrap();
    let result = engine(false)
        .recognize(
            &image,
            Options {
                single_line: true,
                passes: Passes::All,
                ..Options::default()
            },
        )
        .unwrap();
    assert_eq!(result.text.trim(), "Divination");
    assert!(result.confidence.is_finite() && result.confidence > 0.5 && result.confidence <= 1.);
    assert_eq!(result.candidates.len(), 2);
    assert_eq!(
        result.lines[0].bounds,
        [0, 0, image.width(), image.height()]
    );
    assert_eq!(
        result.model.sha256,
        "606d9a0414c6b73c99df75b707c11c70d1c8b12e1d4f900922e185fc37bfca65"
    );
    let serialized = serde_json::to_string(&result).unwrap();
    assert!(serialized.contains("uncalibrated"));
}

#[test]
#[ignore = "requires downloaded models; run training/download_models.py first"]
fn detects_and_reads_multiple_printed_lines() {
    let title = image::open(fixture("docs/gallery/assets/beleren-title.png"))
        .unwrap()
        .into_rgb8();
    let rules = image::open(fixture("docs/gallery/assets/modern-rules.png"))
        .unwrap()
        .into_rgb8();
    let mut canvas = RgbImage::from_pixel(380, 180, image::Rgb([245, 240, 220]));
    imageops::overlay(&mut canvas, &title, 20, 20);
    imageops::overlay(&mut canvas, &rules, 20, 100);
    let image = DynamicImage::ImageRgb8(canvas);
    let ocr = CardOcr::open(env!("CARGO_MANIFEST_DIR")).unwrap();
    let result = ocr.read(&image).unwrap();
    assert!(result.text.contains("Divination"), "{}", result.text);
    assert!(result.text.contains("Draw two cards."), "{}", result.text);
    assert!(result.lines.len() >= 2);
    for line in &result.lines {
        let [x, y, width, height] = line.bounds;
        assert!(x + width <= image.width() && y + height <= image.height());
    }
    assert!(result.model.detector_sha256.is_some());
    let line_only = ocr
        .recognize(
            &DynamicImage::ImageRgb8(title),
            Options {
                single_line: true,
                passes: Passes::Original,
                ..Options::default()
            },
        )
        .unwrap();
    assert!(line_only.model.detector_sha256.is_none());
}

#[test]
#[ignore = "requires downloaded models; run training/download_models.py first"]
fn invalid_inputs_return_errors_without_inference() {
    let ocr = engine(false);
    let empty = DynamicImage::new_rgb8(0, 0);
    assert!(
        ocr.recognize(&empty, Options::default())
            .unwrap_err()
            .to_string()
            .contains("no pixels")
    );
    let image = DynamicImage::new_rgb8(32, 32);
    assert!(
        ocr.recognize(&image, Options::default())
            .unwrap_err()
            .to_string()
            .contains("requires --detection-model")
    );
}

#[test]
#[ignore = "requires the pinned default recognizer; run training/download_models.py"]
fn language_profile_selects_the_trained_model_for_the_whole_card() {
    for (language, path, expected) in [
        ("en", "docs/gallery/assets/beleren-title.png", "Divination"),
        (
            "fr",
            "training/latin-eval/images/fr-m21-47-clean.png",
            "Sphère de capture",
        ),
    ] {
        let ocr = if language == "en" {
            CardOcr::open(env!("CARGO_MANIFEST_DIR"))
        } else {
            CardOcr::open_language(language, env!("CARGO_MANIFEST_DIR"))
        }
        .unwrap();
        let image = image::open(fixture(path)).unwrap();
        let result = ocr
            .recognize(
                &image,
                Options {
                    single_line: true,
                    ..Options::default()
                },
            )
            .unwrap();
        assert_eq!(result.text.trim(), expected);
        assert_eq!(result.model.language, language);
        assert_eq!(
            result.model.sha256,
            "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba"
        );
        assert_eq!(result.candidates.len(), 1);
        assert!(result.method.starts_with("original"));
    }
    assert!(
        CardOcr::load_profile("unsupported-test-profile", env!("CARGO_MANIFEST_DIR"), None)
            .err()
            .unwrap()
            .to_string()
            .contains("no trained language profile")
    );
}

#[test]
#[ignore = "requires the pinned colour recognizer; run training/download_models.py"]
fn pinned_small_colour_export_preserves_printed_french_accents() {
    // Keep the original small-model gold assertion bound to its actual weights.
    // Medium's known fongoïde regression remains in its complete gallery report.
    let config =
        serde_json::from_slice(&std::fs::read(fixture("models/paddle-v6/fr.json")).unwrap())
            .unwrap();
    let ocr = CardOcr::load(fixture("models/ppocr-v6-small.onnx"), None, config).unwrap();
    let image = image::open(fixture("training/latin-eval/images/fr-m21-182-clean.png")).unwrap();
    let result = ocr
        .recognize(
            &image,
            Options {
                single_line: true,
                passes: Passes::Original,
                ..Options::default()
            },
        )
        .unwrap();
    assert_eq!(result.text.trim(), "Renaissance fongoïde");
    assert_eq!(result.model.language, "fr");
    assert!(result.method.contains("LCNetV4 + LightSVTR"));
    assert!(result.confidence.is_finite() && result.confidence > 0.5);
    assert_eq!(
        result.lines[0].bounds,
        [0, 0, image.width(), image.height()]
    );
}
