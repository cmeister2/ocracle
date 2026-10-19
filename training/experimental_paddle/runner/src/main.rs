//! Isolated raw-tensor compatibility probe; this is not a production OCR engine.

use std::{error::Error, fs, path::Path, time::Instant};

use rten::Model;
use rten_tensor::{AsView, Layout, NdTensor, Tensor};

/// Read little-endian f32 input, execute an unmodified model, and save its output.
///
/// # Errors
///
/// Returns errors for invalid arguments, inconsistent input sizes, unreadable
/// files, incompatible models/tensors or inference/output failures.
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: paddle-rten-probe MODEL INPUT_F32 WIDTH OUTPUT_F32".into());
    }
    let width: usize = args[3].parse()?;
    if !(1..=4096).contains(&width) {
        return Err("width must be in 1..=4096".into());
    }
    let bytes = fs::read(&args[2])?;
    let channels = bytes.len() / (48 * width * 4);
    if ![1, 3].contains(&channels) || bytes.len() != channels * 48 * width * 4 {
        return Err("input length disagrees with [1, 1 or 3, 48, width]".into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
        .collect();
    let model = Model::load(fs::read(Path::new(&args[1]))?)?;
    let input = NdTensor::from_data([1, channels, 48, width], values);
    let start = Instant::now();
    let output: Tensor<f32> = model.run_one(input.view().into(), None)?.try_into()?;
    let elapsed = start.elapsed();
    let mut output_bytes = Vec::with_capacity(output.len() * 4);
    for value in output.iter() {
        output_bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&args[4], output_bytes)?;
    println!(
        "shape={:?} elapsed_ms={:.3}",
        output.shape(),
        elapsed.as_secs_f64() * 1000.
    );
    Ok(())
}
