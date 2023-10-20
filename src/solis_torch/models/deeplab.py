from torchvision import models
from typing import Any

import torch.nn as nn


def _modify_deeplabv3_resnet(model: models.segmentation.DeepLabV3, num_channels: int) -> models.segmentation.DeepLabV3:
    # Modify first layer to accept specified number of bands
    bias = False if model.backbone.conv1.bias is None else True
    model.backbone.conv1 = nn.Conv2d(
        in_channels=num_channels,
        out_channels=model.backbone.conv1.out_channels,
        kernel_size=model.backbone.conv1.kernel_size, # type: ignore
        stride=model.backbone.conv1.stride, # type: ignore
        padding=model.backbone.conv1.padding, # type: ignore
        bias=bias)

    return model


def deeplabv3_resnet50(num_channels: int, num_classes: int, **kwargs: Any) -> models.segmentation.DeepLabV3:
    model = models.segmentation.deeplabv3_resnet50(num_classes=num_classes, **kwargs)
    return _modify_deeplabv3_resnet(model, num_channels)


def deeplabv3_resnet101(num_channels: int, num_classes: int, **kwargs: Any) -> models.segmentation.DeepLabV3:
    model = models.segmentation.deeplabv3_resnet101(num_classes=num_classes, **kwargs)
    return _modify_deeplabv3_resnet(model, num_channels)
