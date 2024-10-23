# Text-to-3D-Face-Reconstruction
 integrates Stable Diffusion (for text-to-image generation) and PRNet (for 3D face reconstruction) to create a pipeline that generates 3D facial models from user-provided text descriptions.
# Text-to-3D Face Reconstruction Pipeline

## Project Overview

This project integrates **Stable Diffusion** for text-to-image generation and **PRNet** for 3D face reconstruction. The pipeline takes user-provided text descriptions, generates a corresponding 2D image, and converts that image into a 3D face model in `.obj` format. The models are deployed via **Gradio** on **Hugging Face Spaces** for public interaction.

## Models

### 1. Stable Diffusion (Fine-tuned)

- **Base Model**: [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- **Datasets**:
  - **LAION-400M**: A large-scale dataset of image-text pairs.
  - **FFHQ**: A high-quality dataset of human faces.
  - **COCO**: Common Objects in Context dataset used for general image-object generation.
- **Training**: Fine-tuned using Colab’s free GPU with a reduced batch size and dynamic learning rates.
  
### 2. PRNet (3D Face Reconstruction)

- **Base Model**: [PRNet GitHub](https://github.com/YadiraF/PRNet)
- **Datasets**:
  - **300W-LP**: A dataset for facial images with 3DMM parameters.
  - **AFLW2000-3D**: Dataset for testing and validation of 3D face models.
- **Training**: Trained locally due to TensorFlow 1.x dependencies and GPU requirements.

## How It Works

1. **Text Input**: User provides a text description (e.g., "A smiling woman with curly hair").
2. **2D Image Generation**: The text is passed to the **Stable Diffusion fine-tuned model** which generates a 2D image.
3. **3D Model Generation**: The 2D image is processed by the **PRNet model** to generate a 3D mesh in `.obj` format.
4. **Output**: The 3D `.obj` file is downloadable from the interactive Gradio interface.

## Installation

To run this project locally, clone the repository and install the necessary dependencies.

### Clone the Repository

```bash
git clone https://github.com/your-username/Text-to-3D-Face-Reconstruction.git
cd Text-to-3D-Face-Reconstruction
