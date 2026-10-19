//! Serializable inference configuration, provenance and results.

use serde::{Deserialize, Serialize};

/// Configuration tied to a recognition checkpoint's training alphabet.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Characters indexed by CTC class minus one; class zero is blank.
    pub alphabet: String,
    /// Fixed height of the grayscale recognition tensor.
    pub input_height: u32,
    /// Human-readable checkpoint identifier, not an accuracy claim.
    pub model_name: String,
    /// Recognition architecture shown in method descriptions, such as `CRNN`.
    ///
    /// This metadata describes the exported model; it does not select a decoder.
    /// All supported exports must emit the same CTC tensor contract.
    #[serde(default = "crnn")]
    pub architecture: String,
    /// Printed card language handled by these weights (Scryfall language code).
    ///
    /// A card uses one language profile, including Latin collector/artist text.
    #[serde(default = "english")]
    pub language: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            alphabet: " 0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~€ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz".into(),
            input_height: 64,
            model_name: "ocrs CRNN base".into(),
            architecture: crnn(),
            language: english(),
        }
    }
}

/// Preprocessing passes to evaluate without consulting expected text.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Passes {
    /// Read the input pixels without image enhancement.
    Original,
    /// Enlarge, stretch contrast and apply a mild unsharp mask.
    Enhanced,
    /// Evaluate both and retain the one with the higher posterior score.
    #[default]
    All,
}

/// How to handle the illustration area of a Magic card.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// Mask the illustration for upright images with a standard card aspect ratio.
    Auto,
    /// Mask the conventional illustration area; unsuitable for alternate layouts.
    StandardCard,
    /// Detect text throughout the image, including any artwork.
    #[default]
    Text,
}

/// Options for an individual image.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Treat the image as a single text line, bypassing text detection.
    pub single_line: bool,
    /// Which image preprocessing candidates to evaluate.
    pub passes: Passes,
    /// Standard card illustration handling; line crops always bypass masking.
    pub layout: Layout,
}

/// Identity of the actual recognition weights used for this result.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelIdentity {
    /// Model name supplied with its alphabet and input shape.
    pub name: String,
    /// Printed card language supported by the loaded recognition profile.
    pub language: String,
    /// SHA-256 of the loaded recognition model file.
    pub sha256: String,
    /// SHA-256 of the detector, absent when detection was disabled.
    pub detector_sha256: Option<String>,
}

/// Default language of the original Ocrs recognition checkpoint.
fn english() -> String {
    "en".into()
}

/// Architecture of the original Ocrs recognition checkpoint.
fn crnn() -> String {
    "CRNN".into()
}

/// Recognized text line and its position in the original input image.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Line {
    /// Raw CTC output, with no dictionary or Oracle text substitution.
    pub text: String,
    /// Geometric mean of emitted character posterior peaks, in [0, 1].
    ///
    /// This is uncalibrated; it is not the probability the line is correct.
    pub confidence: f32,
    /// Axis-aligned `[x, y, width, height]` in original image pixels.
    pub bounds: [u32; 4],
}

/// Result from one image preprocessing pass.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Candidate {
    /// Image transformation and recognition algorithm actually used.
    pub method: String,
    /// Lines joined in reading order with newline separators.
    pub text: String,
    /// Character-count weighted geometric mean of line scores; zero if empty.
    pub confidence: f32,
    /// Text lines in reading order, including individual posterior scores.
    pub lines: Vec<Line>,
}

/// Complete result including provenance and competing interpretations.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Recognition {
    /// Selected candidate's text.
    pub text: String,
    /// Selected candidate's uncalibrated character posterior score.
    pub confidence: f32,
    /// Explicit interpretation of the numeric confidence field.
    pub confidence_kind: String,
    /// Selected preprocessing method plus neural decoding method.
    pub method: String,
    /// Original image dimensions `[width, height]`.
    pub image_size: [u32; 2],
    /// Selected candidate's recognized lines.
    pub lines: Vec<Line>,
    /// All evaluated passes, including their raw text and scores.
    pub candidates: Vec<Candidate>,
    /// Recognition model and optional detector provenance.
    pub model: ModelIdentity,
}
