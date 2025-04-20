# WCGANS_GP for Super-Resolution of Medical Images

## Overview

This repository contains the implementation of Wasserstein Conditional Generative Adversarial Networks with Gradient Penalty (WCGANS_GP) for medical image super-resolution, specifically optimized for chest X-ray images. The model transforms low-resolution medical images (64x64) into high-quality high-resolution (256x256) outputs while preserving critical diagnostic features essential for clinical interpretation.

Our approach combines the training stability of Wasserstein GANs with the enhanced feature preservation capabilities of self-attention mechanisms and spectral normalization, resulting in state-of-the-art performance for medical image enhancement.

## Model Architecture Diagram

![Figure-1: Model Architecture Diagram](images/Model_Flowchart.png)

## Key Features

- **Wasserstein Loss with Gradient Penalty**: Provides stable training dynamics and prevents mode collapse
- **Self-Attention Mechanism**: Captures long-range dependencies in medical images to preserve global anatomical context
- **Spectral Normalization**: Ensures Lipschitz continuity in the discriminator for robust adversarial learning
- **Multi-Scale Feature Extraction**: Preserves both global structures and local details crucial for diagnostic accuracy
- **Perceptual Loss**: Uses pre-trained VGG19 network to maintain perceptual quality and diagnostic features

## Architecture Details

### Generator Network

The generator transforms low-resolution images (64×64×3) to high-resolution outputs (256×256×3) through:

1. **Initial Convolutional Layer**: 64 filters with 9×9 kernel size
2. **Residual Blocks**: 16 residual blocks with skip connections to maintain feature propagation
3. **Self-Attention Layer**: Captures spatial dependencies across the entire image
4. **Upsampling Blocks**: Two upsampling blocks with L2 regularization to enhance resolution by 4×
5. **Output Layer**: Tanh activation to produce the final super-resolved image in [-1,1] range

### Discriminator Network

The discriminator evaluates the authenticity of generated images through:

1. **Convolutional Layers**: 8 convolutional layers with increasing filter sizes (64→512)
2. **Spectral Normalization**: Applied to each layer to enforce Lipschitz constraint
3. **LeakyReLU Activation**: Used with α=0.2 to prevent vanishing gradients
4. **Dense Layers**: Fully-connected layers leading to the final score prediction
5. **Linear Output**: Provides Wasserstein distance estimation without sigmoid activation

### Feature Extractor (VGG19)

A pre-trained VGG19 network serves as a feature extractor for perceptual loss calculation:

1. **Frozen Weights**: Pre-trained on ImageNet with frozen layers
2. **Feature Maps**: Extracts intermediate features for perceptual similarity assessment
3. **Content Loss**: Compares VGG features between generated and real high-resolution images

## Dataset

The model was trained on the Chest X-Ray Pneumonia dataset, containing:
- 5,216 chest X-ray images (both normal and pneumonia cases)
- Resized to 256×256 for high-resolution and 64×64 for low-resolution inputs
- Normalized to the range [-1, 1]

## Dataset Image
- **Input**: Low-resolution medical image.
![Figure-2: Low-Resolution](images/LR.jpg)
- **Output**: High-resolution reconstructed image.
![Figure-3: High-Resolution](images/HR.jpg)

## Training Methodology

Training incorporates several advanced techniques:

1. **Wasserstein Loss**: Measures Earth Mover's distance between real and generated distributions
2. **Gradient Penalty**: 10× penalty on gradient norm deviation from 1 for Lipschitz enforcement
3. **Adversarial Loss**: Encourages generator to produce realistic high-resolution images
4. **Perceptual Loss**: Ensures feature-level similarity using VGG19 activations
5. **Adam Optimizer**: Learning rate of 0.0002 and betas of (0.5, 0.9)
6. **Critic Iterations**: Discriminator trained 7× more frequently than generator for stability

Training progress was monitored through:
- Discriminator and generator loss curves
- PSNR (Peak Signal-to-Noise Ratio) metrics
- SSIM (Structural Similarity Index) metrics
- Visual assessment of generated samples


## Results

Visual results demonstrate significant improvement in image quality and detail preservation across training epochs, with early epochs (21-121) showing initial texture formation, mid-range epochs (221-421) developing clear anatomical structures, and later epochs (521-921) refining fine details and contrast.

Our model achieves state-of-the-art performance for medical image super-resolution:

- Peak PSNR: 30.19 dB
- Maximum SSIM: 0.793
- Realistic preservation of clinically important features

## Generated Sample Images

![Figure-4: Gnerated Sample Images](images/gen_img_1000_7.png)

## Progression of Training

The model shows clear quality improvement throughout training:

| Epoch | PSNR (dB) | SSIM | Visual Quality |
|-------|-----------|------|----------------|
| 21    | 5.82      | 0.207| Basic structure formation |
| 121   | 9.56      | 0.405| Improved contrast |
| 221   | 13.06     | 0.594| Clear lung fields |
| 321   | 20.48     | 0.642| Defined ribcage |
| 421   | 23.69     | 0.759| Enhanced vascular markings |
| 521   | 24.96     | 0.765| Improved edge definition |
| 621   | 26.55     | 0.764| Better tissue contrast |
| 721   | 26.70     | 0.779| Fine pulmonary detail |
| 821   | 26.81     | 0.778| Enhanced mediastinal structures |
| 921   | 28.74     | 0.793| Near-diagnostic quality |

![Figure-5: PSNR and SSIM Progression Images](images/PSNR-.png)

## Usage Instructions

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/WCGANS_GP-Medical-SR.git
cd WCGANS_GP-Medical-SR

# Install dependencies
pip install -r requirements.txt
```

### Training

```python
# Set data paths
TRAIN_PATH = '/path/to/chest_xray/train/'
VAL_PATH = '/path/to/chest_xray/val/'
TEST_PATH = '/path/to/chest_xray/test/'

# Run training script
python X3_WC.py
```

### Inference

```python
# Load trained model and perform super-resolution
import tensorflow as tf
from model import build_enhanced_generator

# Load model
generator = build_enhanced_generator()
generator.load_weights('checkpoints/generator_epoch_800.keras')

# Load and preprocess low-resolution image
lr_image = load_and_preprocess_image('path_to_lr_image.png')

# Generate super-resolution image
sr_image = generator.predict(lr_image)

# Save or display the result
save_image(sr_image, 'super_resolved_image.png')
```

## Model Performance Analysis

The training logs demonstrate steady improvement in both quantitative metrics and visual quality:

- **Early Phase (1-200 epochs)**: Rapid PSNR improvement from ~6dB to ~12dB
- **Middle Phase (200-500 epochs)**: Gradual refinement with PSNR reaching ~20dB
- **Later Phase (500-1000 epochs)**: Fine detail enhancement with final PSNR exceeding 28dB

The discriminator loss stabilizes around 9.67-9.68, indicating proper Wasserstein distance estimation, while generator loss decreases steadily from 0.30 to 0.17, showing continuous improvement in generating realistic images.

## Loss Curves ( Training Loss, PSNR, SSIM )

![Figure-6: Training Progress Chart (PSNR, SSIM, D & G_Loss)](images/charts_1000_7.png)

## Technical Implementation

The implementation utilizes TensorFlow 2.x and Keras 3.5.0 with the following components:

- **Self-Attention Block**: Implemented using Query-Key-Value transformations
- **Spectral Normalization**: Applied to discriminator weights using power iteration method
- **Residual Blocks**: 16 blocks with batch normalization and skip connections
- **Gradient Penalty**: Computed through interpolation between real and generated samples

The model architecture is designed to run efficiently on GPUs with at least 8GB VRAM, though 16GB is recommended for larger batch sizes.

## Medical Application

This super-resolution approach has several potential clinical applications:

- **Telemedicine**: Enhancing images transmitted over low-bandwidth connections
- **Legacy System Upgrade**: Improving images from older medical imaging equipment
- **Dose Reduction**: Maintaining diagnostic quality while reducing radiation exposure
- **Mobile Diagnostics**: Enabling better interpretation of portable X-ray images

Validation with medical professionals indicates the super-resolved images maintain diagnostic integrity while significantly improving visibility of subtle features.


## Acknowledgments

This project was developed by Pravinkumar Gohil

I acknowledge the creators of the Chest X-Ray Pneumonia dataset and the foundational work on Wasserstein GANs and image super-resolution that made this project possible.

## License
This project is licensed under the MIT License.
