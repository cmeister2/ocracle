//! Reproduce CTC decoding latency with `cargo run --release --example ctc_latency`.
//!
//! The baseline is the original copy, probability validation, per-class natural
//! logarithm and greedy log-probability decoder. Both paths use identical input.

use std::{hint::black_box, time::Instant};

use anyhow::{Result, ensure};
use ocracle::ctc;
use serde_json::json;

/// Compare the old and optimized paths on deterministic normalized CTC rows.
///
/// # Errors
///
/// Returns errors for malformed arguments, inconsistent decoder results, or JSON
/// serialization failures.
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let classes = args.first().map_or(Ok(97_usize), |value| value.parse())?;
    let steps = args.get(1).map_or(Ok(128_usize), |value| value.parse())?;
    let iterations = args.get(2).map_or(Ok(1000_usize), |value| value.parse())?;
    ensure!(
        (2..=100_000).contains(&classes) && steps > 0 && iterations > 0,
        "usage: ctc_latency [classes=97, 2..100000] [steps=128, >0] [iterations=1000, >0]"
    );
    let alphabet: Vec<_> = (1..classes)
        .filter_map(|index| char::from_u32(0x10000 + index as u32))
        .collect();
    let mut probabilities = Vec::with_capacity(steps * classes);
    for step in 0..steps {
        let label = if step % 11 == 0 {
            0
        } else {
            1 + (step / 3) % (classes - 1)
        };
        let peak = 0.7 + (step % 7) as f32 * 0.04;
        probabilities.extend((0..classes).map(|class| {
            if class == label {
                peak
            } else {
                (1. - peak) / (classes - 1) as f32
            }
        }));
    }
    let baseline = || baseline_decode(black_box(&probabilities), steps, classes, &alphabet);
    let optimized =
        || ctc::decode_probabilities(black_box(&probabilities), steps, classes, &alphabet);
    let reference = baseline()?;
    ensure!(optimized()? == reference, "decoder output changed");
    let mut baseline_samples = Vec::new();
    let mut optimized_samples = Vec::new();
    for _ in 0..10 {
        black_box(baseline()?);
        black_box(optimized()?);
    }
    for sample in 0..15 {
        // Alternate measurement order to reduce warmup and thermal order bias.
        if sample % 2 == 0 {
            baseline_samples.push(measure(iterations, baseline)?);
            optimized_samples.push(measure(iterations, optimized)?);
        } else {
            optimized_samples.push(measure(iterations, optimized)?);
            baseline_samples.push(measure(iterations, baseline)?);
        }
    }
    let mut baseline_sorted = baseline_samples.clone();
    let mut optimized_sorted = optimized_samples.clone();
    baseline_sorted.sort_by(f64::total_cmp);
    optimized_sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "classes": classes, "steps": steps, "iterations_per_sample": iterations,
            "arch": std::env::consts::ARCH, "os": std::env::consts::OS,
            "baseline_samples_us": baseline_samples,
            "optimized_samples_us": optimized_samples,
            "baseline_median_us": baseline_sorted[7],
            "optimized_median_us": optimized_sorted[7],
            "speedup": baseline_sorted[7] / optimized_sorted[7],
            "output_identical": true,
            "confidence": reference.1,
            "text_characters": reference.0.chars().count(),
        }))?
    );
    Ok(())
}

/// Run one interleaved measurement sample and report microseconds per decode.
///
/// # Errors
///
/// Returns a decoder error.
fn measure(iterations: usize, decode: impl Fn() -> Result<(String, f32)>) -> Result<f64> {
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(decode()?);
    }
    Ok(start.elapsed().as_secs_f64() * 1e6 / iterations as f64)
}

/// Reproduce the original production conversion and decoding cost.
///
/// # Errors
///
/// Returns an error for invalid probabilities or malformed decoder input.
fn baseline_decode(
    probabilities: &[f32],
    steps: usize,
    classes: usize,
    alphabet: &[char],
) -> Result<(String, f32)> {
    let mut data = probabilities.to_vec();
    ensure!(
        classes > 0 && data.len().is_multiple_of(classes),
        "invalid CTC class axis"
    );
    for row in data.chunks_exact_mut(classes) {
        ensure!(
            row.iter()
                .all(|value| value.is_finite() && (0. ..=1.000_001).contains(value))
                && (row.iter().sum::<f32>() - 1.).abs() <= 0.001,
            "model must output normalized class probabilities"
        );
        for value in row {
            *value = value.clamp(1e-30, 1.).ln();
        }
    }
    ctc::decode(&data, steps, classes, alphabet)
}
