"""
This file contains functions for creating feature vectors from images using pre-trained models, as well as some utility functions.
"""


import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.resnet import ResNet18_Weights


# Path to the folder containing the images
db_path = '/media/'

def model_resnet18_layer4():
    """
    Load ResNet-18 model and retrieve the fourth layer.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    layer = model._modules.get('layer4')

    model.eval()

    return model, layer

def model_resnet18_layer3():
    """
    Load ResNet-18 model and retrieve the third layer.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    layer = model.layer3

    model.eval()

    return model, layer

def model_resnet18_layer2():
    """
    Load ResNet-18 model and retrieve the second layer.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    layer = model.layer2

    model.eval()

    return model, layer

def model_resnet18_avgpool():
    """
    Load ResNet-18 model and retrieve the average pooling layer.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    layer = model._modules.get('avgpool')

    model.eval()

    return model, layer

