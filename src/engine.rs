//! Neural inference, image preprocessing and original-coordinate line layout.

use std::{borrow::Cow, fs, path::Path};

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
    Candidate, ColorMode, InferenceBackend, Layout, Line, ModelConfig, ModelIdentity, Options,
    OutputLayout, OutputValues, Passes, Recognition, WidthRounding, ctc,
    detector::PortableDetector,
    runtime::{RuntimeModel, RuntimeOptions},
};

/// Reusable MTG text recognizer backed by a CTC neural network.
///
/// Use [`Self::open`] and [`Self::read`] for the default image-to-text pipeline.
pub struct CardOcr {
    /// Optional word detector, absent for line-only deployments.
    detector: Option<OcrEngine>,
    /// Recognition network emitting timestep log probabilities.
    recognizer: RuntimeModel,
    /// Optional alternate detector; the Ocrs engine still handles line ordering.
    portable_detector: Option<PortableDetector>,
    /// Checkpoint-specific alphabet and input height.
    config: ModelConfig,
    /// File digests identifying the actual weights in use.
    identity: ModelIdentity,
    /// Cached Unicode class mapping shared by every line.
    alphabet: Vec<char>,
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
    /// Open the default English card recognizer and its text detector.
    ///
    /// Load once and reuse this instance for multiple images. `model_root`
    /// contains the repository's `models` directory; model choice and input
    /// normalization are supplied by its verified language registry.
    ///
    /// # Errors
    ///
    /// Returns errors for missing, invalid or incompatible model artifacts,
    /// registry/configuration files, or model/configuration digest mismatches.
    pub fn open(model_root: impl AsRef<Path>) -> Result<Self> {
        Self::open_language("en", model_root)
    }

    /// Open the default recognizer for one printed language, with text detection.
    ///
    /// `language` is a printed language code such as `en` or `fr`; callers do
    /// not choose a network size or preprocessing recipe.
    ///
    /// # Errors
    ///
    /// Returns errors for an unsupported language or for any model, registry,
    /// configuration or integrity failure described by [`Self::open`].
    pub fn open_language(language: &str, model_root: impl AsRef<Path>) -> Result<Self> {
        let root = model_root.as_ref();
        let detector = root.join("models/text-detection.onnx");
        Self::load_profile(language, root, Some(&detector))
    }

    /// Read an image using the default detection and recognition pipeline.
    ///
    /// Returns text, coordinates, method, uncalibrated confidence and retained
    /// preprocessing candidates without requiring recognition options.
    ///
    /// # Errors
    ///
    /// Returns errors for invalid or oversized images, missing text detection,
    /// or model execution and output-contract failures.
    pub fn read(&self, image: &DynamicImage) -> Result<Recognition> {
        self.recognize(image, Options::default())
    }

    /// Select one trained language profile for a whole card.
    ///
    /// Reads `models/profiles.json` below `model_root`. Only entries backed by
    /// trained models belong in this registry. Unknown languages return an
    /// error, so an English model cannot silently stand in for another script.
    /// Latin-script languages use the selected upstream colour recognizer.
    /// Non-Latin profiles are best effort; registry membership does not
    /// establish full coverage.
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
        Self::load_profile_with_backend(
            profile_id,
            model_root,
            detection_model,
            InferenceBackend::Rten,
            None,
        )
    }

    /// Load a verified language profile using an explicit inference runtime.
    ///
    /// `threads` optionally configures inference workers for either backend.
    /// `None` retains the default policy, including `RTEN_NUM_THREADS` for RTen.
    /// Reuse the loaded engine between camera frames.
    /// ONNX Runtime uses a shared process-lifetime environment; applications
    /// that manage their own environment should use
    /// `load_profile_with_onnxruntime_environment` (requires `onnxruntime`) from
    /// their first native load.
    ///
    /// # Errors
    ///
    /// Returns profile/integrity errors described by [`Self::load_profile`], or
    /// errors for unavailable runtime support, invalid threads or incompatible models.
    pub fn load_profile_with_backend(
        profile_id: &str,
        model_root: impl AsRef<Path>,
        detection_model: Option<&Path>,
        backend: InferenceBackend,
        threads: Option<usize>,
    ) -> Result<Self> {
        Self::load_profile_with_runtime_options(
            profile_id,
            model_root,
            detection_model,
            RuntimeOptions::new(backend, threads)?,
        )
    }

    /// Load a verified profile in the application's ONNX Runtime environment.
    ///
    /// Both networks retain a handle to `environment` and use independent CPU
    /// thread pools. No environment is stored globally by this loader. Share
    /// the same environment with other OCR engines and native sessions because
    /// ONNX Runtime permits only one environment to be created per process.
    ///
    /// # Errors
    ///
    /// Returns profile/integrity errors described by [`Self::load_profile`], or
    /// errors for invalid threads, session settings or incompatible ONNX models.
    #[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
    pub fn load_profile_with_onnxruntime_environment(
        profile_id: &str,
        model_root: impl AsRef<Path>,
        detection_model: Option<&Path>,
        environment: &ort::environment::Environment,
        threads: Option<usize>,
    ) -> Result<Self> {
        let mut options = RuntimeOptions::new(InferenceBackend::OnnxRuntime, threads)?;
        options.ort_environment = Some(environment.clone());
        Self::load_profile_with_runtime_options(profile_id, model_root, detection_model, options)
    }

    /// Verify a language profile and load it with the selected runtime settings.
    ///
    /// # Errors
    ///
    /// Returns profile, integrity and model-loading errors.
    fn load_profile_with_runtime_options(
        profile_id: &str,
        model_root: impl AsRef<Path>,
        detection_model: Option<&Path>,
        runtime_options: RuntimeOptions,
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
        let engine = Self::load_with_runtime_options(
            root.join(&profile.model),
            detection_model,
            config,
            runtime_options,
        )?;
        ensure!(
            engine.identity.sha256 == profile.sha256,
            "language profile recognition model SHA-256 mismatch"
        );
        Ok(engine)
    }

    /// Load recognition weights and an optional word detector.
    ///
    /// Input defaults to `[1,1,height,width]` grayscale in `[-0.5,0.5]` and
    /// output to `[time,1,alphabet_size+1]` log probabilities. Configuration can
    /// select RGB/BGR, different normalization, padding, NTC and probabilities.
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
        Self::load_with_backend(
            recognition_model,
            detection_model,
            config,
            InferenceBackend::Rten,
            None,
        )
    }

    /// Load models with an explicit CPU inference backend.
    ///
    /// ONNX Runtime requires the `onnxruntime` feature, a compatible native library,
    /// and ONNX weights for both networks. `threads` must be in `1..=i32::MAX`
    /// when supplied. RTen shares one private inference pool between both networks;
    /// ONNX Runtime configures each session with this worker count.
    ///
    /// `None` preserves the backend's default policy, including the process-level
    /// `RTEN_NUM_THREADS` setting for RTen. No environment variables are changed.
    /// Tensor preprocessing retains RTen's global pool policy, and RTen may also
    /// initialize that fallback pool during model execution. This is not a
    /// process-wide thread cap. RTen falls back to the calling thread if the
    /// platform cannot create the requested pool.
    ///
    /// ONNX Runtime creates a shared process-lifetime environment on first use.
    /// Applications managing their own environment should instead use
    /// `load_with_onnxruntime_environment` (requires `onnxruntime`) from their
    /// first native load.
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Self::load`], or errors for a missing
    /// runtime feature/library, invalid thread settings, or unsupported model files.
    pub fn load_with_backend(
        recognition_model: impl AsRef<Path>,
        detection_model: Option<&Path>,
        config: ModelConfig,
        backend: InferenceBackend,
        threads: Option<usize>,
    ) -> Result<Self> {
        Self::load_with_runtime_options(
            recognition_model,
            detection_model,
            config,
            RuntimeOptions::new(backend, threads)?,
        )
    }

    /// Load ONNX models in the application's ONNX Runtime environment.
    ///
    /// The sessions retain `environment` for their lifetime without storing it
    /// globally. Keep a handle if more engines or native sessions will be loaded:
    /// ONNX Runtime permits only one environment to be created per process.
    /// Each network uses an independent CPU pool with the optional worker count;
    /// environment execution providers and global thread pools are not inherited.
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Self::load`], or errors for invalid
    /// threads, session settings or incompatible ONNX models.
    #[cfg(all(feature = "onnxruntime", not(target_arch = "wasm32")))]
    pub fn load_with_onnxruntime_environment(
        recognition_model: impl AsRef<Path>,
        detection_model: Option<&Path>,
        config: ModelConfig,
        environment: &ort::environment::Environment,
        threads: Option<usize>,
    ) -> Result<Self> {
        let mut options = RuntimeOptions::new(InferenceBackend::OnnxRuntime, threads)?;
        options.ort_environment = Some(environment.clone());
        Self::load_with_runtime_options(recognition_model, detection_model, config, options)
    }

    /// Load validated model files using shared inference settings.
    ///
    /// # Errors
    ///
    /// Returns errors for invalid configuration, unreadable files or runtime setup.
    fn load_with_runtime_options(
        recognition_model: impl AsRef<Path>,
        detection_model: Option<&Path>,
        config: ModelConfig,
        runtime_options: RuntimeOptions,
    ) -> Result<Self> {
        let backend = runtime_options.backend;
        let threads = runtime_options.threads;
        ensure!(!config.alphabet.is_empty(), "recognition alphabet is empty");
        ensure!(
            !config.architecture.trim().is_empty(),
            "recognition architecture description is empty"
        );
        ensure!(
            (8..=256).contains(&config.input_height),
            "input height must be in 8..=256"
        );
        ensure!(
            config.input.scale.is_finite()
                && config.input.offset.is_finite()
                && (255. * config.input.scale + config.input.offset).is_finite(),
            "input normalization must be finite"
        );
        ensure!(
            config.input.padding_width <= 4096,
            "input padding width exceeds 4096"
        );
        ensure!(
            (1..=4096).contains(&config.input.min_content_width),
            "minimum content width must be in 1..=4096"
        );
        let recognition_bytes =
            fs::read(recognition_model.as_ref()).context("read recognition model")?;
        let recognizer = RuntimeModel::load(recognition_bytes.clone(), &runtime_options)
            .context("load recognition model")?;
        let detector_bytes = detection_model
            .map(fs::read)
            .transpose()
            .context("read detection model")?;
        let use_portable_detector = backend == InferenceBackend::OnnxRuntime || threads.is_some();
        let detector = detector_bytes
            .as_ref()
            .map(|bytes| -> Result<_> {
                OcrEngine::new(OcrEngineParams {
                    detection_model: if !use_portable_detector {
                        Some(Model::load(bytes.clone()).context("load detection model")?)
                    } else {
                        None
                    },
                    ..Default::default()
                })
            })
            .transpose()?;
        let portable_detector = if use_portable_detector {
            detector_bytes
                .as_ref()
                .map(|bytes| PortableDetector::load(bytes.clone(), &runtime_options))
                .transpose()?
        } else {
            None
        };
        let identity = ModelIdentity {
            backend,
            name: config.model_name.clone(),
            language: config.language.clone(),
            sha256: digest(&recognition_bytes),
            detector_sha256: detector_bytes.as_ref().map(|bytes| digest(bytes)),
        };
        Ok(Self {
            detector,
            recognizer,
            portable_detector,
            alphabet: config.alphabet.chars().collect(),
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
        let mut original = match image.as_rgb8() {
            Some(rgb) => Cow::Borrowed(rgb),
            None => Cow::Owned(image.to_rgb8()),
        };
        let card_layout = !options.single_line
            && match options.layout {
                Layout::Auto => {
                    height >= 300 && (0.68..=0.75).contains(&(width as f32 / height as f32))
                }
                Layout::StandardCard => true,
                Layout::Text => false,
            };
        if card_layout {
            mask_artwork(original.to_mut());
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
            let crop = imageops::crop_imm(image, x, y, w, h).to_image();
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
        let words = match &self.portable_detector {
            Some(portable) => portable.detect_words(image)?,
            None => detector.detect_words(&input)?,
        };
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

    /// Prepare a line with the checkpoint's pixel contract and decode CTC.
    ///
    /// # Errors
    ///
    /// Returns an error if the line is too wide or the recognition model fails
    /// or has an unexpected shape, datatype or alphabet size.
    fn recognize_line(&self, image: &RgbImage) -> Result<(String, f32)> {
        let input = recognition_input(image, &self.config)?;
        let output = self
            .recognizer
            .run(input.view())
            .context("recognition model inference")?;
        let output: NdTensor<f32, 3> = output
            .try_into()
            .context("recognition model must output a three-dimensional float CTC tensor")?;
        let [first, second, classes] = output.shape();
        let (steps, batch) = match self.config.output_layout {
            OutputLayout::Tnc => (first, second),
            OutputLayout::Ntc => (second, first),
        };
        ensure!(batch == 1, "recognition output batch must be one");
        // With a single batch item TNC and NTC have the same flattened order.
        let data = output.into_data();
        match self.config.output_values {
            OutputValues::Probabilities => {
                ctc::decode_probabilities(&data, steps, classes, &self.alphabet)
            }
            OutputValues::LogProbabilities => ctc::decode(&data, steps, classes, &self.alphabet),
        }
    }
}

/// Prepare planar input, retaining the trained channel order and padding value.
///
/// # Errors
///
/// Returns an error for empty input or aspect/padding widths exceeding 4096.
fn recognition_input(image: &RgbImage, config: &ModelConfig) -> Result<NdTensor<f32, 4>> {
    ensure!(
        image.width() > 0 && image.height() > 0,
        "text line has no pixels"
    );
    let height = config.input_height;
    let ideal_width = f64::from(image.width()) * f64::from(height) / f64::from(image.height());
    let width = match config.input.width_rounding {
        WidthRounding::NearestEven => ideal_width.round_ties_even(),
        WidthRounding::Ceil => ideal_width.ceil(),
    }
    .max(f64::from(config.input.min_content_width)) as u64;
    ensure!(
        width <= 4096 && config.input.padding_width <= 4096,
        "text line is too wide; split into smaller crops"
    );
    let width = width as u32;
    let padded_width = width.max(config.input.padding_width);
    let channels = match config.input.color_mode {
        ColorMode::Grayscale => 1,
        ColorMode::Rgb | ColorMode::Bgr => 3,
    };
    let plane = (height * padded_width) as usize;
    let mut data = vec![0.; channels * plane];
    let scale = config.input.scale;
    let offset = config.input.offset;
    match config.input.color_mode {
        ColorMode::Grayscale => {
            let resized = imageops::resize(&grayscale(image), width, height, FilterType::Triangle);
            for (x, y, pixel) in resized.enumerate_pixels() {
                data[(y * padded_width + x) as usize] = f32::from(pixel.0[0]) * scale + offset;
            }
        }
        ColorMode::Rgb | ColorMode::Bgr => {
            let resized = imageops::resize(image, width, height, FilterType::Triangle);
            for (x, y, pixel) in resized.enumerate_pixels() {
                for channel in 0..3 {
                    let source = match config.input.color_mode {
                        ColorMode::Bgr => 2 - channel,
                        _ => channel,
                    };
                    data[channel * plane + (y * padded_width + x) as usize] =
                        f32::from(pixel.0[source]) * scale + offset;
                }
            }
        }
    }
    Ok(NdTensor::from_data(
        [1, channels, height as usize, padded_width as usize],
        data,
    ))
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

#[cfg(test)]
mod tests {
    //! Recognition-contract regressions for original colour model exports.

    use super::recognition_input;
    use crate::{ColorMode, InputConfig, ModelConfig, WidthRounding};
    use image::{Rgb, RgbImage};
    use rten_tensor::Layout;

    #[test]
    fn bgr_planes_preserve_colour_and_pad_after_normalization() {
        let image = RgbImage::from_pixel(2, 1, Rgb([255, 128, 0]));
        let config = ModelConfig {
            input_height: 8,
            input: InputConfig {
                color_mode: ColorMode::Bgr,
                scale: 2. / 255.,
                offset: -1.,
                padding_width: 32,
                min_content_width: 1,
                width_rounding: WidthRounding::Ceil,
            },
            ..ModelConfig::default()
        };
        let tensor = recognition_input(&image, &config).unwrap();
        assert_eq!(tensor.shape(), [1, 3, 8, 32]);
        assert_eq!(tensor[[0, 0, 0, 0]], -1.);
        assert!((tensor[[0, 1, 0, 0]] - (256. / 255. - 1.)).abs() < 1e-6);
        assert_eq!(tensor[[0, 2, 0, 0]], 1.);
        for channel in 0..3 {
            assert_eq!(tensor[[0, channel, 0, 16]], 0.);
            assert_eq!(tensor[[0, channel, 7, 31]], 0.);
        }
    }

    #[test]
    fn checkpoint_width_rounding_and_limits_are_respected() {
        let image = RgbImage::new(33, 16);
        let mut config = ModelConfig {
            input_height: 8,
            ..ModelConfig::default()
        };
        assert_eq!(recognition_input(&image, &config).unwrap().shape()[3], 16);
        config.input.width_rounding = WidthRounding::Ceil;
        assert_eq!(recognition_input(&image, &config).unwrap().shape()[3], 17);
        assert!(recognition_input(&RgbImage::new(9000, 1), &config).is_err());
    }

    #[test]
    fn narrow_colour_glyphs_are_padded_without_stretching_their_content() {
        let image = RgbImage::from_pixel(3, 24, Rgb([255, 255, 255]));
        let config = ModelConfig {
            input_height: 48,
            input: InputConfig {
                color_mode: ColorMode::Bgr,
                scale: 2. / 255.,
                offset: -1.,
                padding_width: 320,
                min_content_width: 1,
                width_rounding: WidthRounding::Ceil,
            },
            ..ModelConfig::default()
        };
        let tensor = recognition_input(&image, &config).unwrap();
        assert_eq!(tensor.shape(), [1, 3, 48, 320]);
        assert_eq!(tensor[[0, 0, 20, 5]], 1.);
        assert_eq!(tensor[[0, 0, 20, 6]], 0.);
    }
}
