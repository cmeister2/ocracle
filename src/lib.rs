//! Neural OCR dedicated to printed Magic: The Gathering card text.
//!
//! [`CardOcr::open`] loads the default recognizer; [`CardOcr::read`] returns text,
//! coordinates, method and uncalibrated character posterior scores. Use
//! [`CardOcr::open_language`] when the printed language is known. Model choice
//! and preprocessing do not require caller configuration. Predictions never
//! substitute text from a card database.
//!
//! Download the separately distributed models with the repository's
//! `training/download_models.py` script. See the README for model provenance,
//! historic font coverage, training and limitations.
//!
//! ```no_run
//! use ocracle::CardOcr;
//!
//! # fn main() -> anyhow::Result<()> {
//! let ocr = CardOcr::open(".")?;
//! let result = ocr.read(&image::open("card.jpg")?)?;
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! ```

mod ctc;
mod engine;
mod types;

pub use engine::CardOcr;
pub use types::{
    Candidate, ColorMode, InputConfig, Layout, Line, ModelConfig, ModelIdentity, Options,
    OutputLayout, OutputValues, Passes, Recognition, WidthRounding,
};
