# Image colorization with GANs

## Overview
This project implements a Deep Learning model to colorize grayscale images using PyTorch. It employs a Generative Adversarial Network (GAN) framework where a Generator predicts color channels (AB) from a grayscale input (L) in the CIELAB color space, and a Discriminator ensures the generated colors appear realistic.

## Dataset
The model is trained on the COCO 2017 dataset, managed via the FiftyOne library. The pipeline includes a preprocessing step that computes color distribution weights to handle class imbalance, ensuring rare colors are not underrepresented during training.

## Architecture

### Generator
A U-Net style Deep Convolutional Autoencoder with skip connections. 
- **Input**: 1-channel grayscale image (L channel).
- **Output**: 2-channel color map (AB channels).
- **Structure**: Encoder (downsampling) and Decoder (upsampling) with skip connections to preserve spatial details.

### Discriminator
A convolutional network that acts as a binary classifier, distinguishing between real images and those produced by the generator. It helps the generator learn more perceptually accurate color distributions.

### Loss Function
The training uses a combined loss:
- **L1 Loss**: Pixel-wise reconstruction error, weighted by a pre-computed color rarity map.
- **Adversarial Loss**: Binary Cross Entropy loss from the GAN framework.

## Requirements
- Python 3.x
- PyTorch
- FiftyOne
- scikit-image
- NumPy
- Matplotlib

## Usage
1. **Data Setup**: The script checks for the COCO dataset or downloads it using FiftyOne.
2. **Training**: Run the training loop. The model saves checkpoints and best-performing weights based on PSNR scores.
3. **Inference**: The trained generator can be used to colorize arbitrary grayscale images.
