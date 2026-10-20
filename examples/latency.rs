//! Reproducible warm OCR latency benchmark; run with `cargo run --release --example latency`.
use anyhow::{Context, Result, ensure};
use ocracle::{CardOcr, InferenceBackend, Layout, ModelConfig, Options};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

/// Load models once, warm inference, then report individual frame latencies.
///
/// # Errors
/// Returns errors for invalid arguments, missing artifacts or inference failures.
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 3,
        "usage: latency MODEL CONFIG IMAGE [iterations=20] [line|full]"
    );
    let iterations: usize = args.get(3).map_or(Ok(20), |s| s.parse())?;
    ensure!(iterations > 0, "iterations must be positive");
    let single_line = args.get(4).is_some_and(|s| s == "line");
    let config: ModelConfig = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let image = image::open(&args[2])?;
    let detector = std::env::var_os("OCRACLE_DETECTOR_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/text-detection.onnx"));
    let layout = match std::env::var("OCRACLE_LAYOUT").as_deref() {
        Ok("standard_card") => Layout::StandardCard,
        Ok("text") | Err(_) => Layout::Text,
        _ => anyhow::bail!("OCRACLE_LAYOUT must be text or standard_card"),
    };
    let backend = match std::env::var("OCRACLE_BACKEND").as_deref() {
        Ok("onnxruntime") => InferenceBackend::OnnxRuntime,
        Ok("rten") | Err(_) => InferenceBackend::Rten,
        _ => anyhow::bail!("OCRACLE_BACKEND must be rten or onnxruntime"),
    };
    let threads = if backend == InferenceBackend::OnnxRuntime {
        Some(
            std::env::var("RTEN_NUM_THREADS")
                .context("set RTEN_NUM_THREADS for benchmark parity")?
                .parse()?,
        )
    } else {
        None
    };
    let start = Instant::now();
    let engine = CardOcr::load_with_backend(
        &args[0],
        (!single_line).then_some(detector.as_path()),
        config,
        backend,
        threads,
    )?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let options = Options {
        single_line,
        layout,
        ..Options::default()
    };
    let start = Instant::now();
    let reference = engine.recognize(&image, options)?;
    let cold_ms = start.elapsed().as_secs_f64() * 1000.;
    for _ in 0..3 {
        engine.recognize(&image, options)?;
    }
    let mut samples = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let result = engine.recognize(&image, options)?;
        samples.push(start.elapsed().as_secs_f64() * 1000.);
        ensure!(
            result.text == reference.text,
            "nondeterministic text output"
        );
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model": reference.model, "image": args[2], "image_size": reference.image_size,
            "single_line": single_line, "layout": layout, "backend": backend, "threads": std::env::var("RTEN_NUM_THREADS").unwrap_or_else(|_| "default".into()),
            "arch": std::env::consts::ARCH, "os": std::env::consts::OS,
            "load_ms": load_ms, "cold_ms": cold_ms, "samples_ms": samples,
            "median_ms": sorted[iterations / 2], "p95_ms": sorted[(iterations * 95).div_ceil(100) - 1],
            "text": reference.text, "lines": reference.lines
        }))?
    );
    Ok(())
}
