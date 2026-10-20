//! Serializable inference configuration, provenance and results.

use serde::{Deserialize, Serialize};

/// CPU runtime used for neural inference.
///
/// The optional ONNX Runtime backend requires the `onnxruntime` Cargo feature
/// and a compatible native ONNX Runtime library deployed with the application.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackend {
    /// Portable Rust runtime, supporting RTen and ONNX model files.
    #[default]
    Rten,
    /// ONNX Runtime CPU execution; accepts ONNX model files only.
    #[serde(rename = "onnxruntime")]
    #[value(name = "onnxruntime")]
    OnnxRuntime,
}

/// Configuration tied to a recognition checkpoint's training alphabet.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Characters indexed by CTC class minus one; class zero is blank.
    pub alphabet: String,
    /// Fixed height of the recognition tensor.
    pub input_height: u32,
    /// Pixel channels, affine normalization, width rounding and right padding.
    #[serde(default)]
    pub input: InputConfig,
    /// Axis order of the model's three-dimensional CTC output.
    #[serde(default)]
    pub output_layout: OutputLayout,
    /// Whether output values are probabilities or their natural logarithms.
    #[serde(default)]
    pub output_values: OutputValues,
    /// Human-readable checkpoint identifier, not an accuracy claim.
    pub model_name: String,
    /// Recognition architecture shown in method descriptions, such as `CRNN`.
    ///
    /// This metadata describes the exported model; it does not select a decoder.
    /// Tensor interpretation comes from the input and output configuration.
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
            input: InputConfig::default(),
            output_layout: OutputLayout::default(),
            output_values: OutputValues::default(),
            model_name: "ocrs CRNN base".into(),
            architecture: crnn(),
            language: english(),
        }
    }
}

/// Pixel preparation tied to a recognition model's training pipeline.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct InputConfig {
    /// Channel order in the planar `[1,channels,height,width]` tensor.
    pub color_mode: ColorMode,
    /// Multiplier applied to each 8-bit channel before adding `offset`.
    pub scale: f32,
    /// Value added after multiplying each channel by `scale`.
    pub offset: f32,
    /// Minimum padded tensor width, using normalized zero on the right.
    ///
    /// Zero disables padding. Content retains its aspect ratio.
    pub padding_width: u32,
    /// Smallest resized content width before padding; 16 for the Ocrs CRNN.
    ///
    /// Colour profiles with independent right padding generally use 1.
    pub min_content_width: u32,
    /// Rounding used when calculating the aspect-preserving content width.
    pub width_rounding: WidthRounding,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            color_mode: ColorMode::default(),
            scale: 1. / 255.,
            offset: -0.5,
            padding_width: 0,
            min_content_width: 16,
            width_rounding: WidthRounding::default(),
        }
    }
}

/// Channels supplied to the recognition network.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorMode {
    /// PIL-compatible, rounded Rec. 601 luminance in one channel.
    #[default]
    Grayscale,
    /// Three channels in red, green, blue order.
    Rgb,
    /// Three channels in blue, green, red order.
    Bgr,
}

/// Rounding convention used by the model's recognition preprocessing.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WidthRounding {
    /// Round to nearest, resolving exact halves toward the even integer.
    #[default]
    NearestEven,
    /// Round upward, as in PaddleOCR's recognition processor.
    Ceil,
}

/// Axis order of a CTC recognition output tensor.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum OutputLayout {
    /// Time, batch, class; used by the Ocrs CRNN.
    #[default]
    Tnc,
    /// Batch, time, class; used by PaddleOCR inference exports.
    Ntc,
}

/// Numeric interpretation of the recognition output.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputValues {
    /// Natural logarithms of normalized class probabilities.
    #[default]
    LogProbabilities,
    /// Normalized class probabilities, including the blank at class zero.
    Probabilities,
}

/// Preprocessing passes to evaluate without consulting expected text.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Passes {
    /// Read the input pixels without image enhancement.
    #[default]
    Original,
    /// Enlarge, stretch contrast and apply a mild unsharp mask.
    Enhanced,
    /// Evaluate both and retain the one with the higher posterior score.
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
    /// Execution runtime; old saved results default to RTen.
    #[serde(default)]
    pub backend: InferenceBackend,
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
