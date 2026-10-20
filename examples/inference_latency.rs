//! Diagnostic inference-only latency on deterministic synthetic tensors.
//!
//! This deliberately excludes image preprocessing, detection postprocessing and
//! CTC decoding. Its measurements are not application or Android frame rates.

use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rten::Model;
use rten_tensor::{Tensor, prelude::*};
use serde_json::json;
use sha2::{Digest, Sha256};

/// Parameters for one synthetic, single-input/single-output model diagnostic.
#[derive(Parser)]
struct Args {
    /// ONNX or RTen model artifact to load once.
    model: PathBuf,
    /// Input dimensions, for example 1,1,800,600 or 1,3,48,320.
    #[arg(long, value_delimiter = ',', required = true)]
    shape: Vec<usize>,
    /// Number of measured warm calls after cold inference and three warmups.
    #[arg(long, default_value_t = 20)]
    iterations: usize,
    /// Optional destination for complete output as little-endian float32.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Validate a categorical-probability final axis of this length.
    #[arg(long)]
    classes: Option<usize>,
}

/// Serialize floats in a runtime-independent format for input and output hashes.
fn float_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Hash exact bytes without depending on the host byte order.
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Run model inference without including tensor creation or output inspection.
///
/// # Errors
///
/// Returns an error if model inference fails or the output is not float32.
fn inference(model: &Model, input: &Tensor<f32>) -> Result<(Tensor<f32>, f64)> {
    let start = Instant::now();
    let output = model.run_one(input.view().into(), None)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.;
    let output: Tensor<f32> = output
        .try_into()
        .context("diagnostic requires float32 output")?;
    Ok((output, elapsed))
}

/// Measure one model and retain enough data to compare another runtime.
///
/// # Errors
///
/// Returns errors for invalid arguments, missing models, invalid/non-deterministic
/// outputs, or output file/JSON failures.
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.iterations > 0 && !args.shape.is_empty() && args.shape.iter().all(|&dim| dim > 0),
        "shape dimensions and iterations must be positive"
    );
    let elements = args
        .shape
        .iter()
        .try_fold(1_usize, |size, &dim| size.checked_mul(dim))
        .context("input shape overflows")?;
    ensure!(elements <= 100_000_000, "diagnostic input is too large");
    let threads: usize = std::env::var("RTEN_NUM_THREADS")
        .context("set RTEN_NUM_THREADS=4 (or another explicit comparison count)")?
        .parse()?;
    ensure!(threads > 0, "RTEN_NUM_THREADS must be positive");
    // Division by a power of two makes all values exactly representable in f32,
    // so NumPy and Rust generate byte-identical inputs without RNG differences.
    let input_data: Vec<f32> = (0..elements)
        .map(|index| (index % 251) as f32 / 128. - 1.)
        .collect();
    let input_sha256 = digest(&float_bytes(&input_data));
    let input = Tensor::from_data(args.shape.as_slice(), input_data);
    let model_bytes = std::fs::read(&args.model)?;
    let model_sha256 = digest(&model_bytes);
    let start = Instant::now();
    let model = Model::load(model_bytes)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    ensure!(
        model.input_ids().len() == 1 && model.output_ids().len() == 1,
        "diagnostic requires one input and one output"
    );
    let (reference, cold_ms) = inference(&model, &input)?;
    let output_shape = reference.shape().to_vec();
    let reference = reference.into_data();
    ensure!(
        !reference.is_empty() && reference.iter().all(|value| value.is_finite()),
        "output must contain finite values"
    );
    for _ in 0..3 {
        inference(&model, &input)?;
    }
    let mut samples = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let (output, elapsed) = inference(&model, &input)?;
        samples.push(elapsed);
        ensure!(output.shape() == output_shape, "output shape changed");
        ensure!(
            output.into_data() == reference,
            "output values changed between identical inputs"
        );
    }
    let categorical = if let Some(classes) = args.classes {
        ensure!(
            classes > 1 && output_shape.last() == Some(&classes),
            "output class axis differs from --classes"
        );
        ensure!(
            reference
                .iter()
                .all(|value| (0. ..=1.000_001).contains(value)),
            "output values are not probabilities"
        );
        let row_sum_max_error = reference
            .chunks_exact(classes)
            .map(|row| (row.iter().map(|&value| f64::from(value)).sum::<f64>() - 1.).abs())
            .fold(0., f64::max);
        ensure!(
            row_sum_max_error <= 0.001,
            "categorical probabilities are not normalized"
        );
        Some(json!({"classes": classes, "row_sum_max_error": row_sum_max_error}))
    } else {
        None
    };
    let raw_output = float_bytes(&reference);
    if let Some(path) = &args.output {
        std::fs::write(path, &raw_output)?;
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "scope": "synthetic tensor inference only; not OCR/application or Android FPS",
            "runtime": "rten", "runtime_version": "0.26.0",
            "model": args.model, "model_sha256": model_sha256,
            "input_shape": args.shape, "input_sha256": input_sha256,
            "input_formula": "float32((index % 251) / 128 - 1), contiguous row-major",
            "threads": threads, "arch": std::env::consts::ARCH, "os": std::env::consts::OS,
            "warmup_runs_after_cold": 3, "load_ms": load_ms, "cold_ms": cold_ms,
            "load_scope": "model parsing/optimization; file read excluded",
            "samples_ms": samples, "median_ms": sorted[args.iterations / 2],
            "p95_ms": sorted[(args.iterations * 95).div_ceil(100) - 1],
            "output": {
                "shape": output_shape, "elements": reference.len(),
                "min": reference.iter().copied().fold(f32::INFINITY, f32::min),
                "max": reference.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                "mean": reference.iter().map(|&value| f64::from(value)).sum::<f64>() / reference.len() as f64,
                "first_values": &reference[..reference.len().min(8)],
                "sha256": digest(&raw_output), "categorical_probabilities": categorical,
                "identical_between_calls": true, "raw_f32le": args.output
            }
        }))?
    );
    Ok(())
}
