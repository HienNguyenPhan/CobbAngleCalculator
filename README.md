# Cobb angle calculator
This project provides a deep learning-based tool for automatic Cobb angle estimation from cervical X-ray images. It leverages a custom Deep Learning model and utility functions to predict the Cobb angle and visualise keypoints, as well as vertebral masks on the input image.

## Features
- **Automatic Cobb Angle Prediction**: Uses a trained neural network to estimate Cobb angles from grayscale X-ray images.
- **Keypoint Detection**: Identifies anatomical landmarks required for Cobb angle calculation.
- **Vertabrae Segmentation**: Identifies anatomical masks required for Cobb angle calculation.
- **Visualisation**: Optionally displays the predicted keypoints and Cobb angle overlayed on the image.

## Directory Structure
├── infer.py # Main inference script         
├── model.py # Model definition (DeepLabV3Plus-based)  
├── utils.py # Utility functions for keypoint extraction and prediction  
├── main.py # Main training loop  
├── train.py # Train script  
├── eval.py # Evaluation script  
├── dataset.py # Dataset structure  
├── dataloader.py # Create and load dataset  
├── requirements.txt # Requirements   
├── README.md # Project documentation

## Requirements

- Python 3.12.10
- PyTorch
- torchvision
- numpy
- opencv-python
- Pillow
- matplotlib
- segmentation_models_pytorch

Install dependencies with:

```bash
pip install -r requirements.txt
```


## Usage

1.  **Prepare Model Checkpoint**

2.  **Ensure you have a trained model checkpoint (e.g., model_checkpoint.pth)**

3.  **Use the infer.py script to predict the Cobb angle for a given image:**
```bash
python infer.py [IMAGE_DIR]
```

## Model detail

- The model uses the MiT-B4 encoder with Imagenet weight alongside a cross-task attention mechanism.

- Accepts single-channel (grayscale) images resized to **256x256** pixels.

- The checkpoint used can be found here: [Google Drive](https://drive.google.com/drive/folders/12efh74VKyuJhhpzJ38f_FkVzCfvgBXBV?usp=sharing)



