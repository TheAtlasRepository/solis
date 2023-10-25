from solis_torch.models import deeplabv3_resnet50, deeplabv3_resnet101

import torch.nn as nn
import lightning.pytorch as pl
import torch


class _DeepLabV3(pl.LightningModule):
    def __init__(self, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def forward(self, input):
        return self.model(input)["out"]

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        loss = self.criterion(output[:, 1], target)
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


class DeepLabV3ResNet50(_DeepLabV3):
    def __init__(self, num_channels: int, num_classes: int, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__(learning_rate, momentum, weight_decay, lr_step_size, lr_gamma)
        self.model = deeplabv3_resnet50(
            num_channels=self.hparams.num_channels,
            num_classes=self.hparams.num_classes)


class DeepLabV3ResNet101(_DeepLabV3):
    def __init__(self, num_channels: int, num_classes: int, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__(learning_rate, momentum, weight_decay, lr_step_size, lr_gamma)
        self.model = deeplabv3_resnet101(
            num_channels=self.hparams.num_channels,
            num_classes=self.hparams.num_classes)
