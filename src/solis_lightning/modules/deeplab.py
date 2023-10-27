from solis_torch.models import deeplabv3_resnet50, deeplabv3_resnet101

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics.classification


class _DeepLabV3(pl.LightningModule):
    def __init__(self, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_precision = torchmetrics.classification.BinaryPrecision()
        self.train_recall = torchmetrics.classification.BinaryRecall()
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()

        self.val_precision = torchmetrics.classification.BinaryPrecision()
        self.val_recall = torchmetrics.classification.BinaryRecall()
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, input):
        return self.model(input)["out"]

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        loss = self.criterion(output[:, 1], target)
        return {"loss": loss, "output": output}
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        loss = self.criterion(output[:, 1], target)
        return {"loss": loss, "output": output}
    
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
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        input, target = batch
        loss = outputs["loss"]
        output = outputs["output"]

        self.train_precision(output[:, 1], target)
        self.train_recall(output[:, 1], target)
        self.train_accuracy(output[:, 1], target)
        self.train_f1(output[:, 1], target)

        self.log("training/loss", loss)
        self.log("training/precision", self.train_precision)
        self.log("training/recall", self.train_recall)
        self.log("training/accuracy", self.train_accuracy)
        self.log("training/f1", self.train_f1)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        input, target = batch
        loss = outputs["loss"]
        output = outputs["output"]

        self.val_precision(output[:, 1], target)
        self.val_recall(output[:, 1], target)
        self.val_accuracy(output[:, 1], target)
        self.val_f1(output[:, 1], target)

        self.log("hp_metric", self.val_f1)
        self.log("validation/loss", loss)
        self.log("validation/precision", self.val_precision)
        self.log("validation/recall", self.val_recall)
        self.log("validation/accuracy", self.val_accuracy)
        self.log("validation/f1", self.val_f1)


class DeepLabV3_ResNet50(_DeepLabV3):
    def __init__(self, num_channels: int, num_classes: int, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__(learning_rate, momentum, weight_decay, lr_step_size, lr_gamma)
        self.model = deeplabv3_resnet50(
            num_channels=self.hparams.num_channels,
            num_classes=self.hparams.num_classes)


class DeepLabV3_ResNet101(_DeepLabV3):
    def __init__(self, num_channels: int, num_classes: int, learning_rate: float, momentum: float, weight_decay: float, lr_step_size: int, lr_gamma: float):
        super().__init__(learning_rate, momentum, weight_decay, lr_step_size, lr_gamma)
        self.model = deeplabv3_resnet101(
            num_channels=self.hparams.num_channels,
            num_classes=self.hparams.num_classes)
