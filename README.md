---
title: G
emoji: 🚀
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
---

# Face Liveness Detection

This project implements a Face Liveness Detection system using deep learning techniques. It leverages a pre-trained ResNet-18 model to classify input images as either "Live" or "Spoof." The model is trained on a custom dataset and exported to the ONNX format for inference.

## Project Overview

The system is designed to detect whether a face in a given image or video is real (live) or fake (spoofed) by analyzing facial features using a trained convolutional neural network (CNN).

### Key Features
- **Model Training**: Train a deep learning model using a custom dataset.
- **Inference**: Use the trained model (in ONNX format) to perform real-time face liveness detection.
- **Gradio Interface**: A user-friendly web interface to interact with the model for training and testing.

## Installation

1. Clone the Repository
Clone the repository to your local machine using:
```bash
git clone https://github.com/your-username/your-repository-name.git

2. Install Dependencies
Install the required libraries using pip:

pip install -r requirements.txt

3. Set Up Files
Ensure that the following files are present:

model.onnx - The trained model in ONNX format.
dataset/training - The directory containing the training images (if you're retraining the model).

Usage
1. Train the Model
To train the model on your custom dataset, run the following:

python app.py

2. Test the Model
Once the model is trained, you can test it by uploading images or videos in the Test Model tab. The system will classify the uploaded content as either "Live" or "Spoof."
/your-repository-name
│
├── app.py             # Main application file with model training and inference
├── model.onnx         # Trained model in ONNX format
├── dataset/           # Directory containing training dataset
│   ├── training/      # Folder with training images
├── requirements.txt   # List of required Python packages
└── README.md          # This README file

Dependencies
PyTorch: For training the model and model operations.
ONNX: For exporting the trained model and running inference.
Gradio: For creating the web-based user interface.
OpenCV: For image and video preprocessing.
Matplotlib/Seaborn: For visualizations (optional).

To install the dependencies:
pip install torch torchvision gradio opencv-python onnx onnxruntime matplotlib seaborn

Contributing:
Feel free to open issues and contribute to the repository. Pull requests are welcome!



