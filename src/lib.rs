//! Neural OCR dedicated to printed Magic: The Gathering card text.
//!
//! [`CardOcr`] reads images, detects text lines, and decodes a neural network's CTC
//! predictions. Small crops can bypass detection with [`Options::single_line`].
//! Results preserve competing preprocessing passes and report uncalibrated
//! character posterior scores. They never substitute text from a card database.
//!
//! Download the separately distributed models with the repository's
//! `training/download_models.py` script. See the README for model provenance,
//! historic font coverage, training and limitations.
//!
//! ```no_run
//! use ocracle::{CardOcr, ModelConfig, Options};
//!
//! # fn main() -> anyhow::Result<()> {
//! let ocr = CardOcr::load("models/text-recognition-base.rten", None, ModelConfig::default())?;
//! let image = image::open("text-line.png")?;
//! let result = ocr.recognize(&image, Options {
//!     single_line: true,
//!     ..Options::default()
//! })?;
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! ```

mod ctc;
mod engine;
mod types;

pub use engine::CardOcr;
pub use types::{
    Candidate, Layout, Line, ModelConfig, ModelIdentity, Options, Passes, Recognition,
};
