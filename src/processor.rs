use crate::config::ModelConfig;
use crate::error::TaggerError;
use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer, RgbImage, Rgba};
use ndarray::{Array, Axis, Ix4};

pub trait ImageProcessor {
    fn process(&self, iamge: &DynamicImage) -> Result<Array<f32, Ix4>, TaggerError>;
    fn process_batch(&self, images: Vec<DynamicImage>) -> Result<Array<f32, Ix4>, TaggerError> {
        let mut image_tensors = Vec::new();

        for image in images {
            image_tensors.push(self.process(&image)?);
        }

        if let Ok(batch_tensor) = ndarray::concatenate(
            Axis(0),
            &image_tensors
                .iter()
                .map(|tensor| tensor.view())
                .collect::<Vec<_>>(),
        ) {
            return Ok(batch_tensor);
        }

        Err(TaggerError::Processor(
            "Failed to process batch".to_string(),
        ))
    }
}

#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    channels: u32,
    height: u32,
    width: u32,
}

impl ImagePreprocessor {
    pub fn new(channels: u32, height: u32, width: u32) -> Self {
        Self {
            channels,
            height,
            width,
        }
    }

    pub fn from_config(config: &ModelConfig) -> Result<Self, TaggerError> {
        let input_size = &config.pretrained_cfg.input_size;
        // check if the input size is valid
        if input_size.len() != 3 {
            return Err(TaggerError::Processor("Invalid input size".to_string()));
        }

        Ok(Self {
            channels: input_size[0],
            height: input_size[1],
            width: input_size[2],
        })
    }

    pub fn from_pretrained(repo_id: &str) -> Result<Self, TaggerError> {
        let config = ModelConfig::from_pretrained(repo_id)?;
        Self::from_config(&config)
    }
}

impl ImageProcessor for ImagePreprocessor {
    /// Preprocess the image for the model input.
    /// Ref: https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py#L112-L162
    fn process(&self, image: &DynamicImage) -> Result<Array<f32, Ix4>, TaggerError> {
        let image_rgba = &image.to_rgba8();
        let (width, height) = image.dimensions().clone();

        // Create a canvas and composite image on top of it
        let canvas = ImageBuffer::from_fn(width, height, |x, y| {
            if let Some(pixel) = image_rgba.get_pixel_checked(x, y) {
                *pixel
            } else {
                Rgba([255, 255, 255, 255])
            }
        });
        let image_rgb = DynamicImage::ImageRgba8(canvas).to_rgb8();

        // Pad image to square
        let max_dim = std::cmp::max(width, height);
        let pad_left = (max_dim - width) / 2;
        let pad_top = (max_dim - height) / 2;

        let mut padded_image = RgbImage::new(max_dim, max_dim);

        for (x, y, pixel) in image_rgb.enumerate_pixels() {
            padded_image.put_pixel(x + pad_left, y + pad_top, *pixel);
        }

        // Resize the padded image
        let resized_image = DynamicImage::ImageRgb8(padded_image).resize(
            self.width.clone(),
            self.height.clone(),
            image::imageops::FilterType::CatmullRom,
        );

        // Convert to ndarray and switch RGB to BGR
        let resized_rgb = resized_image.to_rgb8();

        // Empty tensor (zeros)
        let mut image_tensor = Array::zeros((
            self.height as usize,
            self.width as usize,
            self.channels as usize,
        ));

        // Convert to BGR and normalize
        // float32[batch_size,448,448,3]
        for (x, y, pixel) in resized_rgb.enumerate_pixels() {
            let [r, g, b] = pixel.0;
            image_tensor[[y as usize, x as usize, 0]] = b as f32;
            image_tensor[[y as usize, x as usize, 1]] = g as f32;
            image_tensor[[y as usize, x as usize, 2]] = r as f32;
        }

        // Add batch dimension
        Ok(image_tensor.insert_axis(ndarray::Axis(0)))
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_process_image() {
        let image = image::open("assets/sample1_3x1024x1024.webp").unwrap();

        let processor = ImagePreprocessor::new(3, 448, 448);

        let tensor = processor.process(&image).unwrap();

        println!("{}", tensor);
        dbg!(tensor.shape());
    }
}
