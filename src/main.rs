//! Command-line interface for reproducible MTG card OCR.

use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ocracle::{CardOcr, InferenceBackend, Layout, ModelConfig, Options, Passes};

/// Extract printed Magic: The Gathering text from card images.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Operation to perform.
    #[command(subcommand)]
    command: Command,
}

/// Available OCR operations.
#[derive(Subcommand)]
enum Command {
    /// Recognize a card or text crop and print JSON with confidence and provenance.
    Recognize {
        /// Image to read (PNG, JPEG or WebP).
        image: PathBuf,
        /// CTC recognition weights in RTen or ONNX format.
        #[arg(long, hide = true)]
        recognition_model: Option<PathBuf>,
        /// Printed language, such as en or fr; defaults to English.
        #[arg(long, alias = "profile", conflicts_with_all = ["recognition_model", "model_config"])]
        language: Option<String>,
        /// Root directory containing models/ and its downloaded weights.
        #[arg(long, default_value = ".")]
        model_root: PathBuf,
        /// Detector override; profiles otherwise use models/text-detection.onnx.
        #[arg(long, hide = true)]
        detection_model: Option<PathBuf>,
        /// JSON checkpoint alphabet, input height, architecture and model name.
        #[arg(long, hide = true, requires = "recognition_model")]
        model_config: Option<PathBuf>,
        /// CPU inference runtime; onnxruntime requires its Cargo feature and native library.
        #[arg(long, value_enum, default_value = "rten")]
        backend: InferenceBackend,
        /// Inference worker threads for either backend; omission keeps runtime defaults.
        #[arg(long)]
        threads: Option<usize>,
        /// Treat the entire image as one line and skip detection.
        #[arg(long)]
        line: bool,
        /// Image preprocessing attempts to evaluate.
        #[arg(long, hide = true, value_enum, default_value = "original")]
        passes: Passes,
        /// Illustration masking for standard upright cards; text scans the whole image.
        #[arg(long, hide = true, value_enum, default_value = "text")]
        layout: Layout,
    },
}

/// Execute the requested operation and report contextual errors.
///
/// # Errors
///
/// Returns errors for unreadable images/configuration/models, inference failures
/// or failure to write output. CLI argument failures are handled by Clap.
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Recognize {
            image,
            recognition_model,
            language,
            model_root,
            detection_model,
            model_config,
            backend,
            threads,
            line,
            passes,
            layout,
        } => {
            let language = language.or_else(|| recognition_model.is_none().then(|| "en".into()));
            let engine = if let Some(language) = language {
                let detector = detection_model
                    .or_else(|| (!line).then(|| model_root.join("models/text-detection.onnx")));
                CardOcr::load_profile_with_backend(
                    &language,
                    model_root,
                    detector.as_deref(),
                    backend,
                    threads,
                )?
            } else {
                let config = model_config
                    .map(|path| -> Result<ModelConfig> {
                        serde_json::from_slice(&fs::read(path).context("read model configuration")?)
                            .context("parse model configuration")
                    })
                    .transpose()?
                    .unwrap_or_default();
                CardOcr::load_with_backend(
                    recognition_model.context("supply --language or --recognition-model")?,
                    detection_model.as_deref(),
                    config,
                    backend,
                    threads,
                )?
            };
            let image = image::ImageReader::open(&image)
                .context("open image")?
                .with_guessed_format()?
                .decode()
                .context("decode image")?;
            let result = engine.recognize(
                &image,
                Options {
                    single_line: line,
                    passes,
                    layout,
                },
            )?;
            let mut stdout = io::stdout().lock();
            serde_json::to_writer_pretty(&mut stdout, &result)?;
            writeln!(stdout)?;
        }
    }
    Ok(())
}
