import pytest
import torch
import torchvision.models as models
from cnn_model.model import model_resnet18_avgpool, model_resnet18_layer3, model_resnet18_layer2, model_resnet18_layer4

def test_model_resnet18_layer4():
    """
    Test the model_resnet18_layer4 function.
    This function should return a ResNet-18 model and the fourth layer of the model as a Sequential module.
    The fourth layer should have 512 output channels.
    """
    model, layer = model_resnet18_layer4()
    assert isinstance(model, torch.nn.Module), "The model should be an instance of torch.nn.Module"
    assert isinstance(layer, torch.nn.Sequential), "The layer should be an instance of torch.nn.Sequential"
    assert layer[-1].conv2.out_channels == 512, "The layer should have 512 output channels"

def test_model_resnet18_layer3():
    """
    Test the model_resnet18_layer3 function.
    This function should return a ResNet-18 model and the third layer of the model as a Sequential module.
    The third layer should have 256 output channels.
    """
    model, layer = model_resnet18_layer3()
    assert isinstance(model, torch.nn.Module), "The model should be an instance of torch.nn.Module"
    assert isinstance(layer, torch.nn.Sequential), "The layer should be an instance of torch.nn.Sequential"
    assert layer[-1].conv2.out_channels == 256, "The layer should have 256 output channels"

def test_model_resnet18_layer2():
    """
    Test the model_resnet18_layer2 function.
    This function should return a ResNet-18 model and the second layer of the model as a Sequential module.
    The second layer should have 128 output channels.
    """
    model, layer = model_resnet18_layer2()
    assert isinstance(model, torch.nn.Module), "The model should be an instance of torch.nn.Module"
    assert isinstance(layer, torch.nn.Sequential), "The layer should be an instance of torch.nn.Sequential"
    assert layer[-1].conv2.out_channels == 128, "The layer should have 128 output channels"

def test_model_resnet18_avgpool():
    """
    Test the model_resnet18_avgpool function.
    This function should return a ResNet-18 model and the average pooling layer of the model as an AdaptiveAvgPool2d module.
    """
    model, layer = model_resnet18_avgpool()
    assert isinstance(model, torch.nn.Module), "The model should be an instance of torch.nn.Module"
    assert isinstance(layer, torch.nn.AdaptiveAvgPool2d), "The layer should be an instance of torch.nn.AdaptiveAvgPool2d"
