import gradio as gr
import numpy as np
import os
import time
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
import onnxruntime as ort
import psutil  # For memory usage tracking
import matplotlib.pyplot as plt
import seaborn as sns

# Check for the current working directory and files
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

# Model path and data directories
model_path = "model.onnx"
training_data_path = "dataset/training"
session = None

# Define device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to train a model
def train_model():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets with a subset to speed up training
    train_dataset = datasets.ImageFolder(training_data_path, transform=transform)
    
    # Further reduce the dataset size to speed up training
    small_subset_size = min(len(train_dataset), 100)  # Reduced to 100 samples for faster training
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [small_subset_size, len(train_dataset) - small_subset_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)  # Further reduce batch size
    
    # Use a pre-trained ResNet18 and modify the last layer
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    
    # Freeze layers for faster training
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Fine-tune only the final layer
    
    # Training loop with fewer epochs
    num_epochs = 1  # Reduce epochs to 1 for faster training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # Save the model as ONNX
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy_input, model_path)
    print("Model trained and saved as ONNX.")
    return f"Training completed. Final Accuracy: {epoch_acc:.2f}%", epoch_loss

# Load the ONNX model
def load_model():
    global session
    if session is None:
        session = ort.InferenceSession(model_path)
        print("Model loaded successfully.")

# Preprocess the input image
def preprocess_image(image, input_size=(224, 224)):
    image = cv2.resize(image, input_size)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change to CHW format
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image

# Detect liveness using the ONNX model
def detect_liveness(image):
    if session is None:
        return "Model not loaded", None
    
    input_image = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    start_time = time.time()
    outputs = session.run(None, {input_name: input_image})
    inference_time = time.time() - start_time
    
    result = "Live" if outputs[0][0][0] > 0.5 else "Spoof"
    return result, inference_time

# Gradio interface setup
with gr.Blocks() as app:
    gr.Markdown("# Face Liveness Detection")
    
    with gr.Tab("Train Model"):
        train_button = gr.Button("Train Model")
        train_output = gr.Textbox(label="Training Status")
        
        def train_model_button_action():
            train_status, loss = train_model()
            return train_status
        
        train_button.click(train_model_button_action, outputs=train_output)
    
    with gr.Tab("Test Model"):
        image_input = gr.Image(label="Upload Image", type="numpy")
        video_input = gr.Video(label="Upload Video")  # Removed source parameter
        
        detect_button = gr.Button("Detect Liveness")
        liveness_result = gr.Label(label="Liveness Result")
        inference_time_box = gr.Textbox(label="Inference Time (s)")
        
        def test_model_with_image(image):
            result, inference_time = detect_liveness(image)
            return result, f"{inference_time:.2f} s"
        
        detect_button.click(test_model_with_image, inputs=image_input, outputs=[liveness_result, inference_time_box])

# Load the model initially
load_model()

# Run the app
if __name__ == "__main__":
    app.launch()
