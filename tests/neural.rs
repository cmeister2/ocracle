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
    let ocr = engine(true);
    let result = ocr
        .recognize(
            &image,
            Options {
                single_line: false,
                passes: Passes::Original,
                ..Options::default()
            },
        )
        .unwrap();
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
