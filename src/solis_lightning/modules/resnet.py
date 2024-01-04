from solis_torch.models import resnet18, resnet34, resnet50, resnet101

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics.classification


class _ResNet(pl.LightningModule):
    def __init__(self):
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

        self.example_input_array = torch.zeros(
            1, self.hparams.num_channels, 224, 224)
    
    def forward(self, input):
        return self.model(input)[:, 0]
    
    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        loss = self.criterion(output, target)
        return {"loss": loss, "output": output}
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)
        loss = self.criterion(output, target)
        return {"loss": loss, "output": output}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        input, target = batch
        loss = outputs["loss"]
        output = outputs["output"]

        self.train_precision(output, target)
        self.train_recall(output, target)
        self.train_accuracy(output, target)
        self.train_f1(output, target)

        self.log("training/loss", loss)
        self.log("training/precision", self.train_precision)
        self.log("training/recall", self.train_recall)
        self.log("training/accuracy", self.train_accuracy)
        self.log("training/f1", self.train_f1)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        input, target = batch
        loss = outputs["loss"]
        output = outputs["output"]

        self.val_precision(output, target)
        self.val_recall(output, target)
        self.val_accuracy(output, target)
        self.val_f1(output, target)

        self.log("hp_metric", self.val_f1)
        self.log("validation/loss", loss)
        self.log("validation/precision", self.val_precision)
        self.log("validation/recall", self.val_recall)
        self.log("validation/accuracy", self.val_accuracy)
        self.log("validation/f1", self.val_f1)


class ResNet18(_ResNet):
    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.model = resnet18(
            num_channels=num_channels,
            num_classes=num_classes)


class ResNet34(_ResNet):
    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.model = resnet34(
            num_channels=num_channels,
            num_classes=num_classes)


class ResNet50(_ResNet):
    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.model = resnet50(
            num_channels=num_channels,
            num_classes=num_classes)
        

class ResNet101(_ResNet):
    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.model = resnet101(
            num_channels=num_channels,
            num_classes=num_classes)
