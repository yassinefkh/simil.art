import pytest
import torch
import os
from PIL import Image
from django.conf import settings
from cnn_model.features_extraction import get_vector

# Path to the test images folder
TEST_IMAGES_PATH = os.path.join(settings.BASE_DIR, 'cnn_model','tests')

def test_get_vector_resnet18_avgpool():
    """
    Test the get_vector function with the RESNET18_AVGPOOL model.
    """
    # Load a test image
    img_path = f"{TEST_IMAGES_PATH}/test_image.jpg"
    img = Image.open(img_path)
    img = img.convert('RGB')

    # Get the feature vector
    feature_vector = get_vector(img_path, "RESNET18_AVGPOOL")

    # Check that the feature vector is a PyTorch tensor with the expected shape
    assert isinstance(feature_vector, torch.Tensor)
    assert feature_vector.shape == (512,)

def test_get_vector_resnet18_layer4():
    """
    Test the get_vector function with the RESNET18_LAYER4 model.
    """
    # Load a test image
    img_path = f"{TEST_IMAGES_PATH}/test_image.jpg"
    img = Image.open(img_path)
    img = img.convert('RGB')

    # Get the feature vector
    feature_vector = get_vector(img_path, "RESNET18_LAYER4")

    # Check that the feature vector is a PyTorch tensor with the expected shape
    assert isinstance(feature_vector, torch.Tensor)
    assert feature_vector.shape == (512, 7, 7)

def test_get_vector_resnet18_layer3():
    """
    Test the get_vector function with the RESNET18_LAYER3 model.
    """
    # Load a test image
    img_path = f"{TEST_IMAGES_PATH}/test_image.jpg"
    img = Image.open(img_path)
    img = img.convert('RGB')

    # Get the feature vector
    feature_vector = get_vector(img_path, "RESNET18_LAYER3")

    # Check that the feature vector is a PyTorch tensor with the expected shape
    assert isinstance(feature_vector, torch.Tensor)
    assert feature_vector.shape == (256, 14, 14)

def test_get_vector_resnet18_layer2():
    """
    Test the get_vector function with the RESNET18_LAYER2 model.
    """
    # Load a test image
    img_path = f"{TEST_IMAGES_PATH}/test_image.jpg"
    img = Image.open(img_path)
    img = img.convert('RGB')

    # Get the feature vector
    feature_vector = get_vector(img_path, "RESNET18_LAYER2")

    # Check that the feature vector is a PyTorch tensor with the expected shape
    assert isinstance(feature_vector, torch.Tensor)
    assert feature_vector.shape == (128, 28, 28)
