//! Command-line interface for reproducible MTG card OCR.

use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ocracle::{CardOcr, Layout, ModelConfig, Options, Passes};

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
        /// CRNN recognition weights in RTen or ONNX format.
        #[arg(long, required_unless_present = "language")]
        recognition_model: Option<PathBuf>,
        /// One trained language profile for the whole card (currently en).
        #[arg(long, visible_alias = "profile", conflicts_with_all = ["recognition_model", "model_config"])]
        language: Option<String>,
        /// Repository/model root containing models/profiles.json for --language.
        #[arg(long, default_value = ".")]
        model_root: PathBuf,
        /// Neural word detector; required unless --line is used.
        #[arg(long)]
        detection_model: Option<PathBuf>,
        /// JSON checkpoint alphabet, input_height and model_name configuration.
        #[arg(long)]
        model_config: Option<PathBuf>,
        /// Treat the entire image as one line and skip detection.
        #[arg(long)]
        line: bool,
        /// Image preprocessing attempts to evaluate.
        #[arg(long, value_enum, default_value = "all")]
        passes: Passes,
        /// Illustration masking for standard upright cards; text scans the whole image.
        #[arg(long, value_enum, default_value = "text")]
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
            line,
            passes,
            layout,
        } => {
            let engine = if let Some(language) = language {
                let detector = detection_model
                    .or_else(|| (!line).then(|| model_root.join("models/text-detection.onnx")));
                CardOcr::load_profile(&language, model_root, detector.as_deref())?
            } else {
                let config = model_config
                    .map(|path| -> Result<ModelConfig> {
                        serde_json::from_slice(&fs::read(path).context("read model configuration")?)
                            .context("parse model configuration")
                    })
                    .transpose()?
                    .unwrap_or_default();
                CardOcr::load(
                    recognition_model.context("supply --language or --recognition-model")?,
                    detection_model.as_deref(),
                    config,
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
