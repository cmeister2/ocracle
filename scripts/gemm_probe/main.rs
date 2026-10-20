//! Isolated RTen generic-versus-selected GEMM probe; prepared outside the workspace.
use std::hint::black_box;
use std::time::Instant;

use rten_gemm::{GemmExecutor, GemmInputA, GemmInputB, GemmOptions, WithKernel};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;

fn main() {
    let repetitions: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "20".into())
        .parse()
        .expect("positive repetition count");
    assert!(repetitions > 0);
    let generic = GemmExecutor::<f32>::with_generic_kernel();
    let selected = GemmExecutor::<f32>::new();
    // Synthetic shapes resembling a pointwise convolution and a recognition
    // classifier. This is not an end-to-end OCR or Android frame-rate test.
    for (name, m, n, k) in [
        ("pointwise", 128, 768, 128),
        ("recognition_head", 80, 18_710, 192),
    ] {
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i % 31) as f32 - 15.0) / 16.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i % 29) as f32 - 14.0) / 16.0)
            .collect();
        let a = NdTensor::from_data([m, k], a_data);
        let b = NdTensor::from_data([k, n], b_data);
        let mut reference = vec![0.0; m * n];
        generic
            .gemm(
                &mut reference,
                GemmInputA::Unpacked(a.view()),
                GemmInputB::Unpacked(b.view()),
                GemmOptions::default(),
            )
            .unwrap();
        // Check independent sampled dot products, outside all timing.
        for sample in 0..16 {
            let row = sample * 7 % m;
            let col = sample * 271 % n;
            let expected: f32 = (0..k).map(|i| a[[row, i]] * b[[i, col]]).sum();
            assert_eq!(reference[row * n + col], expected);
        }
        for executor in [&generic, &selected] {
            let mut output = vec![0.0; m * n];
            for _ in 0..3 {
                executor
                    .gemm(
                        &mut output,
                        GemmInputA::Unpacked(a.view()),
                        GemmInputB::Unpacked(b.view()),
                        GemmOptions::default(),
                    )
                    .unwrap();
            }
            assert_eq!(output, reference);
            let mut samples = Vec::new();
            for _ in 0..repetitions {
                let start = Instant::now();
                executor
                    .gemm(
                        black_box(&mut output),
                        GemmInputA::Unpacked(black_box(a.view())),
                        GemmInputB::Unpacked(black_box(b.view())),
                        GemmOptions::default(),
                    )
                    .unwrap();
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
                black_box(&output);
            }
            assert_eq!(output, reference);
            let mut ordered = samples.clone();
            ordered.sort_by(f64::total_cmp);
            println!(
                "{{\"shape_name\":\"{name}\",\"m\":{m},\"n\":{n},\"k\":{k},\
                 \"kernel\":\"{}\",\"median_ms\":{},\"samples_ms\":{:?},\
                 \"warmup_runs\":3,\"packing_timed\":true,\"outputs_equal\":true,\
                 \"reference_dot_products\":16}}",
                executor.kernel_name(),
                ordered[ordered.len() / 2],
                samples,
            );
        }
    }
}
