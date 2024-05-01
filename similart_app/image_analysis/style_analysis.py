from django.conf import settings
import torch
from torchvision import models, transforms
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights  # Make sure this import is correct

from PIL import Image
import os

import torch.nn as nn
from urllib.parse import urlparse


def find_similar_style_images(query_image_path, similar_images_paths):
    """
    Predict the style of the query image and return similar images from the same style directory.

    Args:
        query_image_path (str): Path to the query image.
        similar_images_paths (list): List of paths to images for similarity comparison.

    Returns:
        list: Paths to similar images within the same predicted style.
    """
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
  # Assuming ResNet50 architecture
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),  # Adding a dense layer with 512 units
        nn.ReLU(),                 # Activation function
        nn.Dropout(0.5),           # Dropout for regularization
        nn.Linear(512, 10)         # Final layer with 10 outputs
    )
    model.load_state_dict(torch.load(settings.MODEL_STYLE_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(query_image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Predict the style
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_style_index = predicted.item()

    # Assuming style directories are named with style names at the root level
    style_names = ['Abstract_Expressionism', 'Art_Nouveau_Modern', 'Baroque', 'Expressionism', 
                   'Impressionism', 'Northern_Renaissance', 'Post_Impressionism', 'Realism', 
                   'Romanticism', 'Symbolism']
    predicted_style = style_names[predicted_style_index]
    print(predicted_style)

    def extract_style_name(url):
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        style_name = path_parts[-2]  # Style name is second to last part of the path
        return style_name

    # Filter similar image paths to include only images from the predicted style
    style_image_paths = []
    for similar_image_path in similar_images_paths:
        style_name = extract_style_name(similar_image_path)
        print(style_name)
        if style_name == predicted_style:
            style_image_paths.append(similar_image_path)

    return style_image_paths,predicted_style
