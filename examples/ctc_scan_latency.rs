//! Compare safe CTC row scans with `cargo run --release --example ctc_scan_latency`.
//!
//! These experiments preserve left-to-right summation. The four-lane variants
//! expose independent min/max work to compiler SIMD without unsafe code.
//! They are not used by the production decoder.

use std::{hint::black_box, time::Instant};

use anyhow::{Result, ensure};
use serde_json::json;

/// A row reduction, including the last class tied after clamping and logarithms.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Scan {
    /// Sequential sum, whose exact rounding controls distribution validation.
    sum: f32,
    /// Maximum raw probability.
    maximum: f32,
    /// Last class tied after the probability-to-log conversion.
    label: usize,
    /// Whether all values and the sequential sum meet the decoder contract.
    valid: bool,
}

/// Function pointer shared by independently compiled scan variants.
type Scanner = fn(&[f32]) -> Scan;

/// Original fused scalar validation, summation and argmax loop.
#[inline(never)]
fn fused(row: &[f32]) -> Scan {
    let mut maximum = 0_f32;
    let mut label = 0;
    let mut sum = 0_f32;
    let mut valid = true;
    for (index, &value) in row.iter().enumerate() {
        valid &= (0. ..=1.000_001).contains(&value);
        sum += value;
        if value >= maximum {
            maximum = value;
            label = index;
        }
    }
    complete(row, sum, maximum, valid, Some(label))
}

/// Four independent min/max lanes plus the original ordered scalar sum.
#[inline(never)]
fn lanes4(row: &[f32]) -> Scan {
    let mut minima = [0_f32; 4];
    let mut maxima = [0_f32; 4];
    let mut sum = 0_f32;
    let (chunks, tail) = row.as_chunks::<4>();
    for chunk in chunks {
        for lane in 0..4 {
            minima[lane] = minima[lane].min(chunk[lane]);
            maxima[lane] = maxima[lane].max(chunk[lane]);
        }
        for &value in chunk {
            sum += value;
        }
    }
    let mut minimum = minima.into_iter().fold(0_f32, f32::min);
    let mut maximum = maxima.into_iter().fold(0_f32, f32::max);
    for &value in tail {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        sum += value;
    }
    // NaNs can be ignored by min/max but always make the ordered sum NaN.
    complete(
        row,
        sum,
        maximum,
        minimum >= 0. && maximum <= 1.000_001,
        None,
    )
}

/// Four min/max lanes in one pass, followed by a separate ordered sum pass.
#[inline(never)]
fn separate4(row: &[f32]) -> Scan {
    let mut minima = [0_f32; 4];
    let mut maxima = [0_f32; 4];
    let (chunks, tail) = row.as_chunks::<4>();
    for chunk in chunks {
        for lane in 0..4 {
            minima[lane] = minima[lane].min(chunk[lane]);
            maxima[lane] = maxima[lane].max(chunk[lane]);
        }
    }
    let mut minimum = minima.into_iter().fold(0_f32, f32::min);
    let mut maximum = maxima.into_iter().fold(0_f32, f32::max);
    for &value in tail {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    let sum = row.iter().sum::<f32>();
    complete(
        row,
        sum,
        maximum,
        minimum >= 0. && maximum <= 1.000_001,
        None,
    )
}

/// Finish validation and preserve rounded-logarithm ties for every variant.
fn complete(
    row: &[f32],
    sum: f32,
    maximum: f32,
    values_valid: bool,
    raw_label: Option<usize>,
) -> Scan {
    let valid = values_valid && (sum - 1.).abs() <= 0.001;
    let mut label = raw_label.unwrap_or(0);
    if valid {
        let mut tie_floor = maximum.clamp(1e-30, 1.);
        let score = tie_floor.ln();
        while tie_floor > 1e-30 {
            let lower = tie_floor.next_down().max(1e-30);
            if lower.ln() != score {
                break;
            }
            tie_floor = lower;
        }
        let offset = raw_label.map_or(0, |label| label + 1);
        if let Some(index) = row[offset..].iter().rposition(|&value| value >= tie_floor) {
            label = offset + index;
        }
    }
    Scan {
        sum,
        maximum,
        label,
        valid,
    }
}

/// Benchmark deterministic normalized rows, rotating variant order each sample.
///
/// # Errors
///
/// Returns errors for invalid arguments, changed reduction results or JSON output.
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let classes = args
        .first()
        .map_or(Ok(18_710_usize), |value| value.parse())?;
    let steps = args.get(1).map_or(Ok(128_usize), |value| value.parse())?;
    let iterations = args.get(2).map_or(Ok(20_usize), |value| value.parse())?;
    ensure!(
        (2..=100_000).contains(&classes) && steps > 0 && iterations > 0,
        "usage: ctc_scan_latency [classes=18710, 2..100000] [steps=128, >0] [iterations=20, >0]"
    );
    let mut data = Vec::with_capacity(steps * classes);
    let mut state = 0x4f43_5241_u32;
    for step in 0..steps {
        // Distribute maximum labels across the alphabet to avoid measuring only
        // an almost-full reverse scan or only an almost-empty reverse scan.
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let label = state as usize % classes;
        let peak = 0.7 + (step % 7) as f32 * 0.04;
        data.extend((0..classes).map(|class| {
            if class == label {
                peak
            } else {
                (1. - peak) / (classes - 1) as f32
            }
        }));
    }
    let variants: [(&str, Scanner); 3] = [
        ("fused", fused),
        ("lanes4", lanes4),
        ("separate4", separate4),
    ];
    for row in data.chunks_exact(classes) {
        let reference = fused(row);
        ensure!(reference.valid, "invalid generated row");
        for (name, scan) in variants {
            ensure!(scan(row) == reference, "{name} scan differs");
        }
    }
    let mut samples = [Vec::new(), Vec::new(), Vec::new()];
    for sample in 0..18 {
        for offset in 0..variants.len() {
            let index = (sample + offset) % variants.len();
            let scan = variants[index].1;
            let start = Instant::now();
            for _ in 0..iterations {
                for row in black_box(&data).chunks_exact(classes) {
                    black_box(scan(black_box(row)));
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1e3 / iterations as f64;
            if sample >= 3 {
                samples[index].push(elapsed);
            }
        }
    }
    let results: Vec<_> = variants
        .iter()
        .enumerate()
        .map(|(index, (name, _))| {
            let mut sorted = samples[index].clone();
            sorted.sort_by(f64::total_cmp);
            json!({"variant": name, "median_ms": sorted[7], "samples_ms": samples[index]})
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "classes": classes, "steps": steps, "iterations_per_sample": iterations,
            "arch": std::env::consts::ARCH, "os": std::env::consts::OS,
            "output_identical": true, "results": results,
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Float order, tails and validation contracts constrain the SIMD experiment.

    use super::{Scan, fused, lanes4, separate4};

    #[test]
    fn scans_preserve_ordered_sums_argmax_ties_and_tails() {
        for len in [2, 3, 4, 5, 7, 97, 128, 18_710] {
            let mut row = vec![0.25 / (len - 1) as f32; len];
            for label in [0, len / 2, len - 1] {
                row[label] = 0.75;
                for scan in [lanes4, separate4] {
                    assert_eq!(scan(&row), fused(&row));
                }
                row[label] = 0.25 / (len - 1) as f32;
            }
        }
        for row in [
            vec![0.5, 0.5],
            vec![1., 0., -0., 1e-30],
            vec![0.2, 0.2, 0.2, 0.2, 0.2],
            vec![1.000_000_5, 0., 0., 0.],
        ] {
            for scan in [lanes4, separate4] {
                assert_eq!(scan(&row), fused(&row));
            }
        }
        let mut tied = vec![0.009_999; 100];
        tied[1] = 0.01_f32;
        tied[99] = 0.01_f32.next_down();
        assert_eq!(fused(&tied).label, 99);
        assert_eq!(lanes4(&tied), fused(&tied));
        assert_eq!(separate4(&tied), fused(&tied));
    }

    #[test]
    fn scans_reject_nonfinite_out_of_range_and_unnormalized_values() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
            for position in 0..9 {
                let mut row = vec![0.125; 9];
                row[position] = value;
                for scan in [fused, lanes4, separate4] {
                    assert!(!scan(&row).valid);
                }
            }
        }
        for row in [vec![0.; 4], vec![0.2, 0.2], vec![0.5, 0.5, 0.5]] {
            for scan in [fused, lanes4, separate4] {
                assert!(!scan(&row).valid);
            }
        }
    }

    #[test]
    fn scans_preserve_validation_at_float_sum_boundary() {
        let mut row = vec![0.001; 1000];
        let mut saw_valid = false;
        let mut saw_invalid = false;
        for offset in -256..=256 {
            row[0] = 0.002 + offset as f32 * f32::EPSILON;
            let expected: Scan = fused(&row);
            saw_valid |= expected.valid;
            saw_invalid |= !expected.valid;
            for scan in [lanes4, separate4] {
                let actual = scan(&row);
                assert_eq!(actual.valid, expected.valid);
                assert_eq!(actual.sum, expected.sum);
                if expected.valid {
                    assert_eq!(actual, expected);
                }
            }
        }
        assert!(saw_valid && saw_invalid);
    }
}
