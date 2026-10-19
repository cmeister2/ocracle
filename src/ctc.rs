//! Greedy CTC decoding and an explicitly uncalibrated confidence score.

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
pub(crate) fn decode(
    log_probs: &[f32],
    steps: usize,
    classes: usize,
    alphabet: &[char],
) -> Result<(String, f32)> {
    ensure!(
        classes == alphabet.len() + 1,
        "model class count does not match alphabet"
    );
    ensure!(
        steps.checked_mul(classes) == Some(log_probs.len()),
        "invalid CTC tensor size"
    );
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

#[cfg(test)]
mod tests {
    //! CTC semantics that must survive changes to the neural runtime.

    use super::decode;

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
}
