from solis_torch.models import resnet18, resnet34, resnet50, resnet101

import lightning.pytorch as pl
import torch
import torch.nn as nn


class _ResNet(pl.LightningModule):
    def __init__(self, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
    
    def forward(self, input):
        return self.model(input)
    
    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        loss = self.criterion(output, target)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_step_size,
            gamma=self.hparams.lr_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }


class ResNet18(_ResNet):
    def __init__(self, num_channels: int, num_classes: int, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__(learning_rate, momentum, weight_decay, lr_step_size, lr_gamma)
        self.model = resnet18(
            num_channels=self.hparams.num_channels,
            num_classes=self.hparams.num_classes)


class ResNet34(_ResNet):
    def __init__(self, num_channels: int, num_classes: int, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__(learning_rate, momentum, weight_decay, lr_step_size, lr_gamma)
        self.model = resnet34(
            num_channels=self.hparams.num_channels,
            num_classes=self.hparams.num_classes)


class ResNet50(_ResNet):
    def __init__(self, num_channels: int, num_classes: int, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__(learning_rate, momentum, weight_decay, lr_step_size, lr_gamma)
        self.model = resnet50(
            num_channels=self.hparams.num_channels,
            num_classes=self.hparams.num_classes)
        

class ResNet101(_ResNet):
    def __init__(self, num_channels: int, num_classes: int, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__(learning_rate, momentum, weight_decay, lr_step_size, lr_gamma)
        self.model = resnet101(
            num_channels=self.hparams.num_channels,
            num_classes=self.hparams.num_classes)
