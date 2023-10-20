from torchvision import models

import torch.nn as nn


def _modify_resnet(model: models.ResNet, num_channels: int, num_classes: int) -> models.ResNet:
    # Modify first layer to accept specified number of bands
    bias = False if model.conv1.bias is None else True
    model.conv1 = nn.Conv2d(
        in_channels=num_channels,
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size, # type: ignore
        stride=model.conv1.stride, # type: ignore
        padding=model.conv1.padding, # type: ignore
        bias=bias)

    # Modify last layer to output specified number of classes
    bias = False if model.fc.bias is None else True # type: ignore
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,
        bias=bias)
    
    return model


def resnet18(num_channels: int, num_classes: int) -> models.ResNet:
    model = models.resnet18()
    return _modify_resnet(model, num_channels, num_classes)


def resnet34(num_channels: int, num_classes: int) -> models.ResNet:
    model = models.resnet34()
    return _modify_resnet(model, num_channels, num_classes)


def resnet50(num_channels: int, num_classes: int) -> models.ResNet:
    model = models.resnet50()
    return _modify_resnet(model, num_channels, num_classes)


def resnet101(num_channels: int, num_classes: int) -> models.ResNet:
    model = models.resnet101()
    return _modify_resnet(model, num_channels, num_classes)
