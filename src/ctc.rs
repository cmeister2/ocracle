//! Greedy CTC decoding and an explicitly uncalibrated confidence score.
//!
//! Each decoder accepts one sequence as a row-major `[steps, classes]` slice.
//! Class zero is the CTC blank; class `i > 0` maps to `alphabet[i - 1]`.
//! Consecutive repeated labels are merged, with blanks separating repeated
//! characters. The result contains the decoded text and the geometric mean of
//! emitted-character posterior peaks. This score is not a calibrated
//! probability that the text is correct.
//!
//! ```
//! use ocracle::ctc::decode_probabilities;
//!
//! # fn main() -> anyhow::Result<()> {
//! let probabilities = [
//!     0.1, 0.8, 0.1, // a
//!     1.0, 0.0, 0.0, // blank
//!     0.1, 0.1, 0.8, // b
//! ];
//! let (text, _confidence) = decode_probabilities(&probabilities, 3, 3, &['a', 'b'])?;
//! assert_eq!(text, "ab");
//! # Ok(())
//! # }
//! ```

use anyhow::{Result, ensure};

/// Decode timestep log probabilities, merging repeated labels before blanks.
///
/// Within each repeated label run the peak posterior is used for the emitted
/// character. The returned score is the geometric mean over emitted characters.
///
/// # Errors
///
/// Returns an error for inconsistent tensor size, nonfinite values, positive
/// log probabilities or an alphabet/class-count mismatch.
pub fn decode(
    log_probs: &[f32],
    steps: usize,
    classes: usize,
    alphabet: &[char],
) -> Result<(String, f32)> {
    validate_shape(log_probs.len(), steps, classes, alphabet)?;
    ensure!(
        log_probs.iter().all(|p| p.is_finite() && *p <= 0.001),
        "model must output finite log probabilities"
    );
    let mut text = String::new();
    let mut previous = 0;
    let mut peaks: Vec<f32> = Vec::new();
    for row in log_probs.chunks_exact(classes) {
        let (label, score) = row
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, 0.));
        if label != 0 {
            if label != previous {
                text.push(alphabet[label - 1]);
                peaks.push(score.min(0.));
            } else if let Some(peak) = peaks.last_mut() {
                *peak = peak.max(score.min(0.));
            }
        }
        previous = label;
    }
    let confidence = if peaks.is_empty() {
        0.
    } else {
        (peaks.iter().sum::<f32>() / peaks.len() as f32).exp()
    };
    Ok((text, confidence))
}

/// Decode normalized class probabilities without taking a logarithm per class.
///
/// This preserves the clamp, rounded-logarithm ties and confidence calculation
/// of converting every probability to a log probability before [`decode`].
///
/// # Errors
///
/// Returns an error for an inconsistent shape, an alphabet/class-count mismatch,
/// nonfinite or out-of-range values, or a row sum differing from one by >0.001.
pub fn decode_probabilities(
    probabilities: &[f32],
    steps: usize,
    classes: usize,
    alphabet: &[char],
) -> Result<(String, f32)> {
    validate_shape(probabilities.len(), steps, classes, alphabet)?;
    let mut text = String::new();
    let mut previous = 0;
    let mut peaks: Vec<f32> = Vec::new();
    for row in probabilities.chunks_exact(classes) {
        // Independent min/max lanes let LLVM emit SIMD without reassociating
        // the sum: its original left-to-right rounding controls validation.
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
        // Min/max may ignore NaNs, but NaNs and infinities always fail the sum
        // check. Together these checks preserve per-value range validation.
        ensure!(
            minimum >= 0. && maximum <= 1.000_001 && (sum - 1.).abs() <= 0.001,
            "model must output normalized class probabilities"
        );

        let maximum = maximum.clamp(1e-30, 1.);
        let score = maximum.ln();
        // The old decoder selects the last class tied after `clamp().ln()`.
        // Distinct adjacent probabilities can round to the same f32 logarithm.
        // Find that exact interval using only a few neighboring logarithms,
        // then compare the remaining probabilities without calling `ln`.
        let mut tie_floor = maximum;
        while tie_floor > 1e-30 {
            let lower = tie_floor.next_down().max(1e-30);
            if lower.ln() != score {
                break;
            }
            tie_floor = lower;
        }
        let label = row
            .iter()
            .rposition(|&value| value >= tie_floor)
            .unwrap_or(0);

        if label != 0 {
            if label != previous {
                text.push(alphabet[label - 1]);
                peaks.push(score);
            } else if let Some(peak) = peaks.last_mut() {
                *peak = peak.max(score);
            }
        }
        previous = label;
    }
    let confidence = if peaks.is_empty() {
        0.
    } else {
        (peaks.iter().sum::<f32>() / peaks.len() as f32).exp()
    };
    Ok((text, confidence))
}

/// Check tensor dimensions before traversing either CTC value representation.
///
/// # Errors
///
/// Returns an error for inconsistent tensor size or alphabet/class count.
fn validate_shape(len: usize, steps: usize, classes: usize, alphabet: &[char]) -> Result<()> {
    ensure!(
        classes == alphabet.len() + 1,
        "model class count does not match alphabet"
    );
    ensure!(
        steps.checked_mul(classes) == Some(len),
        "invalid CTC tensor size"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    //! CTC semantics that must survive changes to the neural runtime.

    use anyhow::{Result, ensure};

    use super::{decode, decode_probabilities};

    #[test]
    fn repeated_letters_require_a_blank_separator() {
        let rows: Vec<f32> = [1, 1, 0, 1, 2]
            .into_iter()
            .flat_map(|label| {
                (0..3).map(move |i| {
                    if i == label {
                        0.8_f32.ln()
                    } else {
                        0.1_f32.ln()
                    }
                })
            })
            .collect();
        let (text, score) = decode(&rows, 5, 3, &['a', 'b']).unwrap();
        assert_eq!(text, "aab");
        assert!((score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn blank_and_malformed_outputs_are_explicit() {
        assert_eq!(
            decode(&[0., -2.], 1, 2, &['a']).unwrap(),
            (String::new(), 0.)
        );
        assert!(decode(&[f32::NAN, 0.], 1, 2, &['a']).is_err());
        assert!(decode(&[0.], 1, 1, &['a']).is_err());
        assert!(decode(&[0., 1.], 1, 2, &['a']).is_err());
    }

    #[test]
    fn unicode_labels_use_character_indices_instead_of_utf8_byte_offsets() {
        let rows: Vec<f32> = [1, 2, 0, 3]
            .into_iter()
            .flat_map(|label| {
                (0..4).map(move |i| {
                    if i == label {
                        0.7_f32.ln()
                    } else {
                        0.1_f32.ln()
                    }
                })
            })
            .collect();
        let (text, score) = decode(&rows, 4, 4, &['é', 'Ж', '龍']).unwrap();
        assert_eq!(text, "éЖ龍");
        assert!((score - 0.7).abs() < 1e-6);
    }

    /// Evaluate the original probability-to-log conversion for equivalence.
    fn reference_probabilities(
        probabilities: &[f32],
        steps: usize,
        alphabet: &[char],
    ) -> (String, f32) {
        reference_probability_result(probabilities, steps, alphabet).unwrap()
    }

    /// Reproduce the original validation and per-class conversion, including
    /// sequential summation at the normalization tolerance boundary.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed probabilities or decoder input.
    fn reference_probability_result(
        probabilities: &[f32],
        steps: usize,
        alphabet: &[char],
    ) -> Result<(String, f32)> {
        let classes = alphabet.len() + 1;
        ensure!(
            probabilities.len().is_multiple_of(classes),
            "invalid CTC class axis"
        );
        let mut logs = probabilities.to_vec();
        for row in logs.chunks_exact_mut(classes) {
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
        decode(&logs, steps, classes, alphabet)
    }

    #[test]
    fn probability_decoding_matches_log_conversion_with_blanks_and_unicode() {
        let probabilities = [
            0.1, 0.8, 0.1, 0.1, 0.7, 0.2, 1., 0., 0., 0.1, 0.8, 0.1, 0.2, 0.1, 0.7,
        ];
        let alphabet = ['é', '龍'];
        assert_eq!(
            decode_probabilities(&probabilities, 5, 3, &alphabet).unwrap(),
            reference_probabilities(&probabilities, 5, &alphabet),
        );
        assert_eq!(
            decode_probabilities(&probabilities, 5, 3, &alphabet)
                .unwrap()
                .0,
            "éé龍"
        );
    }

    #[test]
    fn probability_decoding_preserves_ties_created_by_rounded_logarithms() {
        let maximum = 0.01_f32;
        let lower = maximum.next_down();
        assert_eq!(maximum.ln(), lower.ln());
        let alphabet: Vec<_> = (0..99)
            .map(|index| char::from_u32(0x100 + index).unwrap())
            .collect();
        let mut probabilities = vec![0.009_999; 100];
        probabilities[1] = maximum;
        probabilities[99] = lower;
        let decoded = decode_probabilities(&probabilities, 1, 100, &alphabet).unwrap();
        assert_eq!(decoded.0, alphabet[98].to_string());
        assert_eq!(
            decoded,
            reference_probabilities(&probabilities, 1, &alphabet),
        );
    }

    #[test]
    fn probability_decoding_preserves_clamps_exact_ties_and_empty_results() {
        let alphabet = ['a', 'b'];
        for probabilities in [
            [1.000_000_5, 0., -0.],
            [0., 1.000_000_5, 1e-35],
            [0., 0.5, 0.5],
            [0.5, 0.5, 0.],
            [1e-35, 0.4, 0.6],
        ] {
            assert_eq!(
                decode_probabilities(&probabilities, 1, 3, &alphabet).unwrap(),
                reference_probabilities(&probabilities, 1, &alphabet),
            );
        }
        assert_eq!(
            decode_probabilities(&[], 0, 3, &alphabet).unwrap(),
            (String::new(), 0.)
        );
    }

    #[test]
    fn probability_decoding_matches_conversion_across_alphabet_sizes() {
        // Fixed PRNG state makes the broad equivalence corpus reproducible.
        let mut state = 0x4f43_5241_u32;
        for classes in [2, 3, 97, 128, 7100, 18_710] {
            let alphabet: Vec<_> = (1..classes)
                .map(|index| char::from_u32(0x1000 + index as u32).unwrap())
                .collect();
            let mut probabilities = Vec::new();
            for step in 0..32 {
                let mut row = Vec::new();
                for class in 0..classes {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    row.push(if (step + class) % 7 == 0 {
                        0.
                    } else {
                        (state >> 8) as f32
                    });
                }
                let sum: f32 = row.iter().sum();
                probabilities.extend(row.into_iter().map(|value| value / sum));
            }
            assert_eq!(
                decode_probabilities(&probabilities, 32, classes, &alphabet).unwrap(),
                reference_probabilities(&probabilities, 32, &alphabet),
            );
        }
    }

    #[test]
    fn probability_simd_lanes_and_tails_preserve_labels_and_confidence() {
        for classes in 2..=9 {
            let alphabet: Vec<_> = (1..classes)
                .map(|index| char::from_u32(0x100 + index as u32).unwrap())
                .collect();
            for label in [0, classes / 2, classes - 1] {
                let mut row = vec![0.25 / (classes - 1) as f32; classes];
                row[label] = 0.75;
                assert_eq!(
                    decode_probabilities(&row, 1, classes, &alphabet).unwrap(),
                    reference_probabilities(&row, 1, &alphabet),
                );
            }
        }
    }

    #[test]
    fn probability_simd_lanes_and_tails_reject_malformed_values() {
        let alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.000_01] {
            for position in 0..9 {
                let mut row = vec![0.125; 9];
                row[position] = value;
                assert!(reference_probability_result(&row, 1, &alphabet).is_err());
                assert!(decode_probabilities(&row, 1, 9, &alphabet).is_err());
            }
        }
    }

    #[test]
    fn probability_simd_preserves_original_normalization_boundary() {
        let alphabet: Vec<_> = (1..1000)
            .map(|index| char::from_u32(0x100 + index).unwrap())
            .collect();
        let mut probabilities = vec![0.001; 1000];
        let mut saw_valid = false;
        let mut saw_invalid = false;
        for offset in -256..=256 {
            probabilities[0] = 0.002 + offset as f32 * f32::EPSILON;
            let expected = reference_probability_result(&probabilities, 1, &alphabet);
            let actual = decode_probabilities(&probabilities, 1, 1000, &alphabet);
            match expected {
                Ok(expected) => {
                    saw_valid = true;
                    assert_eq!(actual.unwrap(), expected);
                }
                Err(_) => {
                    saw_invalid = true;
                    assert!(actual.is_err());
                }
            }
        }
        assert!(saw_valid && saw_invalid);
    }

    #[test]
    fn probability_decoding_rejects_malformed_values_and_dimensions() {
        for probabilities in [
            [f32::NAN, 1.],
            [f32::INFINITY, 0.],
            [f32::NEG_INFINITY, 1.],
            [-0.01, 1.01],
            [0., 1.000_01],
            [0.2, 0.2],
            [0., 0.],
        ] {
            assert!(decode_probabilities(&probabilities, 1, 2, &['a']).is_err());
        }
        assert!(decode_probabilities(&[], 0, 0, &[]).is_err());
        assert!(decode_probabilities(&[1.], 1, 1, &['a']).is_err());
        assert!(decode_probabilities(&[1.], 1, 2, &['a']).is_err());
        assert!(decode_probabilities(&[1., 0.], usize::MAX, 2, &['a']).is_err());
    }
}
