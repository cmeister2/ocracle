//! Neural inference, image preprocessing and original-coordinate line layout.

use std::{fs, path::Path};

use anyhow::{Context, Result, ensure};
use image::{
    DynamicImage, GrayImage, RgbImage,
    imageops::{self, FilterType},
};
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;
use rten_imageproc::bounding_rect;
use rten_tensor::Layout as _;
use rten_tensor::{NdTensor, prelude::*};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{
    Candidate, Layout, Line, ModelConfig, ModelIdentity, Options, Passes, Recognition, ctc,
};

/// Reusable MTG text recognizer backed by a CTC neural network.
pub struct CardOcr {
    /// Optional word detector, absent for line-only deployments.
    detector: Option<OcrEngine>,
    /// Recognition network emitting timestep log probabilities.
    recognizer: Model,
    /// Checkpoint-specific alphabet and input height.
    config: ModelConfig,
    /// File digests identifying the actual weights in use.
    identity: ModelIdentity,
}

/// Registry of available, trained per-card language profiles.
#[derive(Deserialize)]
struct ProfileRegistry {
    /// Registry format understood by this crate.
    schema_version: u32,
    /// Trained profiles; coverage backlog entries belong in the training inventory.
    profiles: Vec<LanguageProfile>,
}

/// Checkpoint and character mapping for a single printed card language.
#[derive(Deserialize)]
struct LanguageProfile {
    /// Unique selection key, usually a Scryfall language code.
    id: String,
    /// Language identifier expected in the matching model configuration.
    language: String,
    /// Recognition model path relative to the supplied model root.
    model: String,
    /// Alphabet and preprocessing configuration relative to the model root.
    configuration: String,
    /// Digest required of the recognition weights actually loaded.
    sha256: String,
    /// Digest of the exact alphabet/preprocessing configuration file.
    configuration_sha256: String,
}

impl CardOcr {
    /// Select one trained language profile for a whole card.
    ///
    /// Reads `models/profiles.json` below `model_root`. Only entries backed by
    /// trained models belong in this registry. Unknown languages return an
    /// error, so an English model cannot silently stand in for another script.
    /// The current distributed profile is `en`.
    ///
    /// # Errors
    ///
    /// Returns errors for unreadable or invalid registry/configuration files,
    /// unsupported registry versions, missing/duplicate profile identifiers,
    /// mismatched languages, model loading failures or incorrect model/config digests.
    pub fn load_profile(
        profile_id: &str,
        model_root: impl AsRef<Path>,
        detection_model: Option<&Path>,
    ) -> Result<Self> {
        let root = model_root.as_ref();
        let registry: ProfileRegistry = serde_json::from_slice(
            &fs::read(root.join("models/profiles.json"))
                .context("read trained language profiles")?,
        )
        .context("parse trained language profiles")?;
        ensure!(
            registry.schema_version == 1,
            "unsupported language profile registry version"
        );
        let mut matches = registry
            .profiles
            .iter()
            .filter(|profile| profile.id == profile_id);
        let profile = matches.next().with_context(|| {
            format!(
                "no trained language profile {profile_id:?}; available: {}",
                registry
                    .profiles
                    .iter()
                    .map(|p| p.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        ensure!(
            matches.next().is_none(),
            "duplicate language profile {profile_id:?}"
        );
        let config_bytes = fs::read(root.join(&profile.configuration))
            .context("read language profile configuration")?;
        ensure!(
            digest(&config_bytes) == profile.configuration_sha256,
            "language profile configuration SHA-256 mismatch"
        );
        let config: ModelConfig = serde_json::from_slice(&config_bytes)
            .context("parse language profile configuration")?;
        ensure!(
            config.language == profile.language,
            "language profile and model configuration disagree"
        );
        let engine = Self::load(root.join(&profile.model), detection_model, config)?;
        ensure!(
            engine.identity.sha256 == profile.sha256,
            "language profile recognition model SHA-256 mismatch"
        );
        Ok(engine)
    }

    /// Load recognition weights and an optional word detector.
    ///
    /// The recognizer accepts `[1,1,height,width]` grayscale values in
    /// `[-0.5,0.5]` and returns `[time,1,alphabet_size+1]` log probabilities.
    ///
    /// # Errors
    ///
    /// Returns an error for unreadable or incompatible model files, or invalid
    /// alphabet/height configuration. Shape mismatches are reported at inference.
    pub fn load(
        recognition_model: impl AsRef<Path>,
        detection_model: Option<&Path>,
        config: ModelConfig,
    ) -> Result<Self> {
        ensure!(!config.alphabet.is_empty(), "recognition alphabet is empty");
        ensure!(
            !config.architecture.trim().is_empty(),
            "recognition architecture description is empty"
        );
        ensure!(
            (8..=256).contains(&config.input_height),
            "input height must be in 8..=256"
        );
        let recognition_bytes =
            fs::read(recognition_model.as_ref()).context("read recognition model")?;
        let recognizer =
            Model::load(recognition_bytes.clone()).context("load recognition model")?;
        let detector_bytes = detection_model
            .map(fs::read)
            .transpose()
            .context("read detection model")?;
        let detector = detector_bytes
            .as_ref()
            .map(|bytes| -> Result<_> {
                OcrEngine::new(OcrEngineParams {
                    detection_model: Some(
                        Model::load(bytes.clone()).context("load detection model")?,
                    ),
                    ..Default::default()
                })
            })
            .transpose()?;
        let identity = ModelIdentity {
            name: config.model_name.clone(),
            language: config.language.clone(),
            sha256: digest(&recognition_bytes),
            detector_sha256: detector_bytes.as_ref().map(|bytes| digest(bytes)),
        };
        Ok(Self {
            detector,
            recognizer,
            config,
            identity,
        })
    }

    /// Recognize a card, a text box, or an isolated line crop.
    ///
    /// The `all` setting selects the preprocessing pass with the highest mean
    /// emitted-character posterior. Such selection can favor short incorrect
    /// outputs; all alternatives remain in the result for inspection.
    ///
    /// # Errors
    ///
    /// Returns an error for empty or excessively large images, missing detector
    /// in multiline mode, inference failures or incompatible model outputs.
    pub fn recognize(&self, image: &DynamicImage, options: Options) -> Result<Recognition> {
        let (width, height) = (image.width(), image.height());
        ensure!(width > 0 && height > 0, "image has no pixels");
        ensure!(
            u64::from(width) * u64::from(height) <= 24_000_000,
            "image exceeds 24 megapixels"
        );
        let mut original = image.to_rgb8();
        let card_layout = !options.single_line
            && match options.layout {
                Layout::Auto => {
                    height >= 300 && (0.68..=0.75).contains(&(width as f32 / height as f32))
                }
                Layout::StandardCard => true,
                Layout::Text => false,
            };
        if card_layout {
            mask_artwork(&mut original);
        }
        let layout_method = if card_layout {
            "MTG standard-frame artwork mask + "
        } else {
            ""
        };
        let mut candidates = Vec::new();
        let architecture = &self.config.architecture;
        if matches!(options.passes, Passes::Original | Passes::All) {
            candidates.push(self.run_pass(
                &original,
                [width, height],
                options.single_line,
                card_layout,
                &format!("{layout_method}original + {architecture} / greedy CTC"),
            )?);
        }
        if matches!(options.passes, Passes::Enhanced | Passes::All) {
            let enhanced = enhance(&original);
            let scale = enhanced.width() / width;
            candidates.push(self.run_pass(
                &enhanced,
                [width, height],
                options.single_line,
                card_layout,
                &format!("{layout_method}{scale}x Lanczos + contrast stretch + unsharp mask + {architecture} / greedy CTC"),
            )?);
        }
        let best = candidates
            .iter()
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
            .context("no preprocessing candidates")?;
        let mut model = self.identity.clone();
        if options.single_line {
            model.detector_sha256 = None;
        }
        Ok(Recognition {
            text: best.text.clone(),
            confidence: best.confidence,
            confidence_kind: "uncalibrated geometric mean of emitted CTC character posterior peaks"
                .into(),
            method: best.method.clone(),
            image_size: [width, height],
            lines: best.lines.clone(),
            candidates,
            model,
        })
    }

    /// Run detection and recognition for a single preprocessing candidate.
    ///
    /// # Errors
    ///
    /// Returns an error if multiline detection is unavailable or neural inference fails.
    fn run_pass(
        &self,
        image: &RgbImage,
        original_size: [u32; 2],
        single_line: bool,
        card_layout: bool,
        method: &str,
    ) -> Result<Candidate> {
        let regions = if single_line {
            vec![[0, 0, image.width(), image.height()]]
        } else {
            self.detect_regions(image)?
        };
        let gray = grayscale(image);
        let mut lines = Vec::new();
        for [x, y, w, h] in regions {
            if w == 0 || h == 0 {
                continue;
            }
            // Ignore detections on the artificial mask or its horizontal edges.
            let center_y = (y as f32 + h as f32 / 2.) / image.height() as f32;
            if card_layout && (0.105..0.54).contains(&center_y) {
                continue;
            }
            let crop = imageops::crop_imm(&gray, x, y, w, h).to_image();
            let (text, confidence) = self.recognize_line(&crop)?;
            if text.trim().is_empty() {
                continue;
            }
            let left =
                ((u64::from(x) * u64::from(original_size[0])) / u64::from(image.width())) as u32;
            let top =
                ((u64::from(y) * u64::from(original_size[1])) / u64::from(image.height())) as u32;
            let right = (u64::from(x + w) * u64::from(original_size[0]))
                .div_ceil(u64::from(image.width())) as u32;
            let bottom = (u64::from(y + h) * u64::from(original_size[1]))
                .div_ceil(u64::from(image.height())) as u32;
            lines.push(Line {
                text,
                confidence,
                bounds: [left, top, right - left, bottom - top],
            });
        }
        let count: usize = lines.iter().map(|line| line.text.chars().count()).sum();
        let confidence = if count == 0 {
            0.
        } else {
            (lines
                .iter()
                .map(|line| {
                    line.confidence.max(f32::MIN_POSITIVE).ln() * line.text.chars().count() as f32
                })
                .sum::<f32>()
                / count as f32)
                .exp()
        };
        Ok(Candidate {
            method: method.into(),
            text: lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            confidence,
            lines,
        })
    }

    /// Detect words, group them into reading order and bound each line.
    ///
    /// # Errors
    ///
    /// Returns an error when there is no detector or detection fails.
    fn detect_regions(&self, image: &RgbImage) -> Result<Vec<[u32; 4]>> {
        let detector = self
            .detector
            .as_ref()
            .context("multiline OCR requires --detection-model; use --line for a line crop")?;
        let source = ImageSource::from_bytes(image.as_raw(), image.dimensions())?;
        let input = detector.prepare_input(source)?;
        let words = detector.detect_words(&input)?;
        let lines = detector.find_text_lines(&input, &words);
        Ok(lines
            .iter()
            .filter_map(|words| {
                let rect = bounding_rect(words.iter())?.integral_bounding_rect();
                let left = rect.left().max(0).min(image.width() as i32) as u32;
                let top = rect.top().max(0).min(image.height() as i32) as u32;
                let right = rect.right().max(0).min(image.width() as i32) as u32;
                let bottom = rect.bottom().max(0).min(image.height() as i32) as u32;
                Some([
                    left,
                    top,
                    right.saturating_sub(left),
                    bottom.saturating_sub(top),
                ])
            })
            .collect())
    }

    /// Resize a grayscale line and decode the network's CTC log probabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if the line is too wide or the recognition model fails
    /// or has an unexpected shape, datatype or alphabet size.
    fn recognize_line(&self, gray: &GrayImage) -> Result<(String, f32)> {
        let height = self.config.input_height;
        let width = (f64::from(gray.width()) * f64::from(height) / f64::from(gray.height()))
            .round_ties_even()
            .max(16.) as u64;
        ensure!(
            width <= 4096,
            "text line is too wide; split into smaller crops"
        );
        let resized = imageops::resize(gray, width as u32, height, FilterType::Triangle);
        let input = NdTensor::from_data(
            [1, 1, height as usize, width as usize],
            resized
                .as_raw()
                .iter()
                .map(|p| f32::from(*p) / 255. - 0.5)
                .collect::<Vec<_>>(),
        );
        let output = self
            .recognizer
            .run_one(input.view().into(), None)
            .context("recognition model inference")?;
        let output: NdTensor<f32, 3> = output
            .try_into()
            .context("recognition model must output a float [time, 1, class] tensor")?;
        let [steps, batch, classes] = output.shape();
        ensure!(batch == 1, "recognition output batch must be one");
        let data: Vec<f32> = output.iter().copied().collect();
        ctc::decode(
            &data,
            steps,
            classes,
            &self.config.alphabet.chars().collect::<Vec<_>>(),
        )
    }
}

/// Compute a lowercase SHA-256 file identity.
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Mask standard-frame illustration pixels to suppress false text in the art.
///
/// The rectangle is deliberately restricted to the shared interior of original
/// and modern standard illustrations. Showcase, full-art and split cards need
/// the explicit text layout instead.
fn mask_artwork(image: &mut RgbImage) {
    let (width, height) = image.dimensions();
    for y in height * 12 / 100..height * 54 / 100 {
        for x in width * 4 / 100..width * 96 / 100 {
            image.put_pixel(x, y, image::Rgb([220, 220, 220]));
        }
    }
}

/// Convert RGB to the same rounded Rec. 601 luminance used in PIL training.
fn grayscale(image: &RgbImage) -> GrayImage {
    GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let [r, g, b] = image.get_pixel(x, y).0.map(u32::from);
        image::Luma([((19595 * r + 38470 * g + 7471 * b + 32768) >> 16) as u8])
    })
}

/// Enlarge small type and modestly increase its local contrast.
fn enhance(image: &RgbImage) -> RgbImage {
    let (width, height) = image.dimensions();
    let scale = if u64::from(width) * u64::from(height) <= 6_000_000 {
        2
    } else {
        1
    };
    let mut gray = grayscale(&imageops::resize(
        image,
        width * scale,
        height * scale,
        FilterType::Lanczos3,
    ));
    let mut hist = [0_u64; 256];
    for pixel in gray.as_raw() {
        hist[usize::from(*pixel)] += 1;
    }
    let tail = u64::from(gray.width()) * u64::from(gray.height()) / 100;
    let quantile = |target| {
        let mut total = 0;
        hist.iter()
            .position(|count| {
                total += count;
                total > target
            })
            .unwrap_or(255) as u8
    };
    let low = quantile(tail);
    let high = quantile(u64::from(gray.width()) * u64::from(gray.height()) - tail - 1);
    if high > low {
        for pixel in gray.as_mut() {
            *pixel = ((u16::from(pixel.saturating_sub(low)) * 255 / u16::from(high - low)).min(255))
                as u8;
        }
    }
    DynamicImage::ImageLuma8(imageops::unsharpen(&gray, 1., 3)).into_rgb8()
}
