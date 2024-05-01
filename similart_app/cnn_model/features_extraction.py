"""
This file contains functions to extract feature vectors from images using pre-trained models.
"""

from PIL import Image
import torchvision.transforms as transforms
from cnn_model.model import model_resnet18_layer4, model_resnet18_layer3, model_resnet18_layer2, model_resnet18_avgpool


# Path to the folder containing the images
db_path = '/media/'

def get_vector(image_url, model):
    """
    Extracts the feature vector of an image using a specified model.

    Args:
    - image_url: URL or path of the input image.
    - model: Name of the pretrained model to use.

    Returns:
    - Feature vector of the input image.
    """
    # Open the image
    img = Image.open(image_url)
    img = img.convert('RGB')

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    t_img = preprocess(img).unsqueeze(0)

    output_tensor = None  # Initialize the feature vector

    # Function to copy feature vectors from the desired layer
    def copy_data(m, i, o):
        nonlocal output_tensor
        output_tensor = o.clone()

    # Get the model and layer based on the specified model name
    if model == "RESNET18_AVGPOOL":
        model, layer = model_resnet18_avgpool()
    elif model == "VGG16":
        model, layer = model_vgg16()
    elif model == "RESNET18_LAYER4":
        model, layer = model_resnet18_layer4()
    elif model == "RESNET18_LAYER3":
        model, layer = model_resnet18_layer3()
    elif model == "RESNET18_LAYER2":
        model, layer = model_resnet18_layer2()

    # Register the hook to copy feature vectors from the layer
    h = layer.register_forward_hook(copy_data)

    # Pass the image through the model
    model(t_img)

    # Remove the hook
    h.remove()

    # Return the feature vector
    return output_tensor.squeeze().detach().cpu()  # Detach to prevent gradient tracking, CPU for non-GPU devices
