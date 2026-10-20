//! Detector preprocessing and word geometry shared with the alternate runtime.
//!
//! Adapted from Ocrs 0.13.1 `preprocess.rs` and `detection.rs` at
//! <https://github.com/robertknight/ocrs/tree/e5f1c5205326804637368422fe420aa1b9769676/ocrs/src>.
//! The default Ocrs path remains the reference for these pixel and box contracts.

// Copyright (c) 2023 Ocrs project contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

use anyhow::{Context, Result, ensure};
use image::RgbImage;
use rten::{FloatOperators, Operators};
use rten_imageproc::{RetrievalMode, RotatedRect, find_contours, min_area_rect, simplify_polygon};
use rten_tensor::{NdTensor, NdTensorView, Tensor, prelude::*};

use crate::runtime::{RuntimeModel, RuntimeOptions};

/// Normalized black value used by the pinned Ocrs detector.
const BLACK: f32 = -0.5;

/// Float RGB-to-luminance weights, including Ocrs' byte normalization.
const RGB_WEIGHTS: [f32; 3] = [0.299 / 255., 0.587 / 255., 0.114 / 255.];

/// Word detector with the pinned Ocrs pixel and postprocessing contract.
pub(crate) struct PortableDetector {
    /// Session executing the segmentation network.
    model: RuntimeModel,
    /// Fixed model input height and width.
    input_size: [usize; 2],
}

impl PortableDetector {
    /// Load a single-channel segmentation model with fixed spatial dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime cannot load the model or its input is not
    /// a single NCHW grayscale image with fixed, nonzero spatial dimensions.
    pub(crate) fn load(bytes: Vec<u8>, options: &RuntimeOptions) -> Result<Self> {
        let model = RuntimeModel::load(bytes, options)?;
        let dimensions = model.input_dimensions()?;
        let input_size = detector_size(&dimensions)?;
        Ok(Self { model, input_size })
    }

    /// Detect oriented word rectangles in original image coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error for empty images, tensor conversion failures, failed
    /// inference, or segmentation outputs incompatible with the model input.
    pub(crate) fn detect_words(&self, image: &RgbImage) -> Result<Vec<RotatedRect>> {
        let original_size = [image.height() as usize, image.width() as usize];
        let (input, padding) = detector_input(image, self.input_size)?;
        let output = self.model.run(input.view())?;
        let mask = restore_mask(output, self.input_size, original_size, padding)?;
        Ok(word_rectangles(mask.view()))
    }
}

/// Validate the fixed spatial dimensions and grayscale batch/channel contract.
///
/// # Errors
///
/// Returns an error if the input is incompatible with a single grayscale image.
fn detector_size(dimensions: &[Option<usize>]) -> Result<[usize; 2]> {
    let [batch, channels, Some(height), Some(width)] = dimensions else {
        anyhow::bail!("detector must have fixed NCHW spatial dimensions");
    };
    ensure!(
        batch.is_none_or(|value| value == 1) && channels.is_none_or(|value| value == 1),
        "detector input batch and channels must be one"
    );
    ensure!(
        *height > 0
            && *width > 0
            && i32::try_from(*height).is_ok()
            && i32::try_from(*width).is_ok(),
        "invalid detector spatial dimensions"
    );
    Ok([*height, *width])
}

/// Convert RGB bytes to Ocrs float grayscale, then pad and resize for detection.
///
/// Returns the NCHW tensor and bottom/right padding in model pixel coordinates.
///
/// # Errors
///
/// Returns an error for empty images, invalid dimensions or tensor operations.
fn detector_input(
    image: &RgbImage,
    input_size: [usize; 2],
) -> Result<(NdTensor<f32, 4>, [usize; 2])> {
    let [height, width] = [image.height() as usize, image.width() as usize];
    let [input_height, input_width] = input_size;
    ensure!(
        height > 0 && width > 0 && input_height > 0 && input_width > 0,
        "detector image dimensions must be positive"
    );
    let pixels: Vec<_> = image
        .as_raw()
        .as_chunks::<3>()
        .0
        .iter()
        .map(|rgb| {
            // Preserve Ocrs' operation order and floating-point luminance;
            // integer grayscale conversion changes detector probabilities.
            let mut value = BLACK;
            for channel in 0..3 {
                value += f32::from(rgb[channel]) * RGB_WEIGHTS[channel];
            }
            value
        })
        .collect();
    let mut input = NdTensor::from_data([1, 1, height, width], pixels).into_dyn();
    let pad_bottom = input_height.saturating_sub(height);
    let pad_right = input_width.saturating_sub(width);
    if pad_bottom > 0 || pad_right > 0 {
        let pads = [
            0,
            0,
            0,
            0,
            0,
            0,
            i32::try_from(pad_bottom).context("detector bottom padding is too large")?,
            i32::try_from(pad_right).context("detector right padding is too large")?,
        ];
        input = input.pad(pads.as_slice().into(), BLACK)?;
    }
    if input.size(2) != input_height || input.size(3) != input_width {
        input = input.resize_image(input_size)?;
    }
    let input = input
        .try_into()
        .context("detector input must be a four-dimensional tensor")?;
    Ok((input, [pad_bottom, pad_right]))
}

/// Crop detector padding and resample the probability mask to original pixels.
///
/// # Errors
///
/// Returns an error for incompatible output dimensions or tensor operations.
fn restore_mask(
    mask: Tensor<f32>,
    input_size: [usize; 2],
    original_size: [usize; 2],
    padding: [usize; 2],
) -> Result<NdTensor<f32, 2>> {
    let [input_height, input_width] = input_size;
    ensure!(
        mask.shape() == [1, 1, input_height, input_width],
        "detector must output a [1,1,height,width] probability mask"
    );
    ensure!(
        padding[0] < input_height
            && padding[1] < input_width
            && original_size[0] > 0
            && original_size[1] > 0,
        "invalid detector mask geometry"
    );
    let mask = mask
        .slice((
            ..,
            ..,
            ..(input_height - padding[0]),
            ..(input_width - padding[1]),
        ))
        .resize_image(original_size)?;
    Ok(mask.into_shape(original_size))
}

/// Match Ocrs' threshold, contour simplification, expansion and minimum area.
fn word_rectangles(mask: NdTensorView<f32, 2>) -> Vec<RotatedRect> {
    let binary = mask.map(|probability| *probability > 0.2);
    find_contours(binary.view(), RetrievalMode::External)
        .iter()
        .filter_map(|polygon| {
            let points: Vec<_> = polygon.iter().map(|point| point.to_f32()).collect();
            let simplified = simplify_polygon(&points, 2.);
            min_area_rect(&simplified).map(|mut rectangle| {
                rectangle.resize(rectangle.width() + 6., rectangle.height() + 6.);
                rectangle
            })
        })
        .filter(|rectangle| rectangle.area() >= 100.)
        .collect()
}

#[cfg(test)]
mod tests {
    //! Pixel and original-coordinate geometry contracts for alternate detection.

    use anyhow::Result;
    use image::{Rgb, RgbImage};
    use rten_imageproc::{BoundingRect, PointF};
    use rten_tensor::{NdTensor, prelude::*};

    use super::{
        BLACK, PortableDetector, detector_input, detector_size, restore_mask, word_rectangles,
    };

    #[test]
    fn input_uses_float_rgb_luminance_and_black_padding() {
        let image = RgbImage::from_fn(3, 1, |x, _| {
            let mut pixel = [0; 3];
            pixel[x as usize] = 255;
            Rgb(pixel)
        });
        let (input, padding) = detector_input(&image, [2, 4]).unwrap();
        assert_eq!(input.shape(), [1, 1, 2, 4]);
        assert_eq!(padding, [1, 1]);
        assert_eq!(input[[0, 0, 0, 0]], -0.5 + 255. * (0.299_f32 / 255.));
        assert_eq!(input[[0, 0, 0, 1]], -0.5 + 255. * (0.587_f32 / 255.));
        assert_eq!(input[[0, 0, 0, 2]], -0.5 + 255. * (0.114_f32 / 255.));
        assert_eq!(input[[0, 0, 0, 3]], BLACK);
        for x in 0..4 {
            assert_eq!(input[[0, 0, 1, x]], BLACK);
        }
    }

    #[test]
    fn input_resizes_large_images_after_padding_only_short_dimensions() {
        let image = RgbImage::from_fn(8, 2, |x, y| {
            if x < 4 && y == 0 {
                Rgb([255; 3])
            } else {
                Rgb([0; 3])
            }
        });
        let (input, padding) = detector_input(&image, [4, 4]).unwrap();
        assert_eq!(input.shape(), [1, 1, 4, 4]);
        assert_eq!(padding, [2, 0]);
        assert!(input[[0, 0, 0, 0]] > 0.49);
        assert_eq!(input[[0, 0, 0, 3]], BLACK);
        for y in 1..4 {
            for x in 0..4 {
                assert_eq!(input[[0, 0, y, x]], BLACK);
            }
        }
    }

    #[test]
    fn mask_restoration_discards_padding_before_resizing() {
        let mut mask = NdTensor::full([1, 1, 6, 8], 1.);
        for y in 0..3 {
            for x in 0..4 {
                mask[[0, 0, y, x]] = 0.;
            }
        }
        let restored = restore_mask(mask.into_dyn(), [6, 8], [3, 4], [3, 4]).unwrap();
        assert_eq!(restored.shape(), [3, 4]);
        assert!(restored.iter().all(|&value| value == 0.));

        let constant = NdTensor::full([1, 1, 3, 4], 0.75);
        let restored = restore_mask(constant.into_dyn(), [3, 4], [9, 12], [0, 0]).unwrap();
        assert_eq!(restored.shape(), [9, 12]);
        assert!(restored.iter().all(|&value| value == 0.75));
    }

    #[test]
    fn word_boxes_keep_threshold_expansion_and_original_pixel_coordinates() {
        let mut mask = NdTensor::full([40, 60], 0.2_f32);
        for y in 10..=20 {
            for x in 20..=40 {
                mask[[y, x]] = 0.2_f32.next_up();
            }
        }
        let rectangles = word_rectangles(mask.view());
        assert_eq!(rectangles.len(), 1);
        assert_eq!(rectangles[0].center(), PointF::from_yx(15., 30.));
        let bounds = rectangles[0].bounding_rect();
        assert_eq!(
            [bounds.top(), bounds.left(), bounds.bottom(), bounds.right()],
            [7., 17., 23., 43.]
        );
    }

    #[test]
    fn minimum_area_applies_after_expansion_and_includes_exact_boundary() {
        let mut mask = NdTensor::zeros([30, 40]);
        for y in 5..10 {
            for x in 5..10 {
                mask[[y, x]] = 1.;
            }
        }
        for y in 5..9 {
            for x in 25..30 {
                mask[[y, x]] = 1.;
            }
        }
        let rectangles = word_rectangles(mask.view());
        assert_eq!(rectangles.len(), 1);
        assert_eq!(rectangles[0].area(), 100.);
        assert_eq!(rectangles[0].center(), PointF::from_yx(7., 7.));
    }

    #[test]
    fn invalid_dimensions_and_mask_geometry_return_errors() {
        assert!(detector_size(&[Some(1), Some(1), Some(800), Some(600)]).is_ok());
        assert!(detector_size(&[None, None, Some(800), Some(600)]).is_ok());
        for dimensions in [
            vec![Some(1), Some(3), Some(800), Some(600)],
            vec![Some(2), Some(1), Some(800), Some(600)],
            vec![Some(1), Some(1), None, Some(600)],
            vec![Some(1), Some(1), Some(0), Some(600)],
            vec![Some(1), Some(800), Some(600)],
        ] {
            assert!(detector_size(&dimensions).is_err());
        }
        assert!(detector_input(&RgbImage::new(0, 2), [4, 4]).is_err());
        assert!(detector_input(&RgbImage::new(2, 2), [0, 4]).is_err());
        assert!(
            restore_mask(
                NdTensor::zeros([1, 2, 4, 4]).into_dyn(),
                [4, 4],
                [4, 4],
                [0, 0]
            )
            .is_err()
        );
        assert!(
            restore_mask(
                NdTensor::zeros([1, 1, 4, 4]).into_dyn(),
                [4, 4],
                [0, 4],
                [0, 0]
            )
            .is_err()
        );
        assert!(
            restore_mask(
                NdTensor::zeros([1, 1, 4, 4]).into_dyn(),
                [4, 4],
                [4, 4],
                [4, 0]
            )
            .is_err()
        );
    }

    /// Compare the port to Ocrs with the same RTen network, isolating pixel and
    /// geometric behavior from floating-point differences between runtimes.
    ///
    /// # Errors
    ///
    /// Returns an error if downloaded fixtures or detector inference fail.
    #[test]
    #[ignore = "requires downloaded detector and image fixtures; runs neural inference"]
    fn portable_detector_matches_ocrs_reference() -> Result<()> {
        let bytes = std::fs::read("models/text-detection.onnx")?;
        let options = crate::runtime::RuntimeOptions::new(crate::InferenceBackend::Rten, Some(2))?;
        let portable = PortableDetector::load(bytes.clone(), &options)?;
        let reference = ocrs::OcrEngine::new(ocrs::OcrEngineParams {
            detection_model: Some(rten::Model::load(bytes)?),
            ..Default::default()
        })?;
        let original = image::open("docs/gallery/assets/modern-rules.png")?.to_rgb8();
        let enlarged = image::imageops::resize(
            &original,
            original.width() * 3,
            original.height() * 3,
            image::imageops::FilterType::Triangle,
        );
        let mut canvas = RgbImage::from_pixel(1200, 900, Rgb([255; 3]));
        image::imageops::replace(&mut canvas, &enlarged, 100, 100);
        for image in [original, enlarged, canvas] {
            let source = ocrs::ImageSource::from_bytes(image.as_raw(), image.dimensions())?;
            let expected = reference.detect_words(&reference.prepare_input(source)?)?;
            let actual = portable.detect_words(&image)?;
            assert!(!expected.is_empty());
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.corners(), expected.corners());
            }
        }
        Ok(())
    }
}
