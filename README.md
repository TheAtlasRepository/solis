# Solis
Solis is Atlas' open-source library for binary classification and segmentation of multispectral images. It includes the following models, all of which have a configurable number of input channels:
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- DeepLabV3 with ResNet50 backbone
- DeepLabV3 with ResNet101 backbone

It also includes support for datasets created using [s2dataset](https://github.com/TheAtlasRepository/s2dataset). Models and datasets are available for both PyTorch and Lightning.

This project was originally intended to detect solar installations in satellite images, hence the name _solis_, but it makes no assumptions about the objects to be detected or the source of the images. This means it is equally well suited for, for example, aerial photographs or medical images.

## Installation
Solis is not yet available on PyPi, but can be installed directly from GitHub using pip:
```bash
pip install https://github.com/TheAtlasRepository/solis/archive/main.zip
```

## CLI Usage
To run the included CLI application, use `python -m solis`. This is simply a LightningCLI which has access to all the models and datasets included in this project. For more info on how to use the CLI, use `python -m solis --help` or check out the [LightningCLI documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html). The preferred way to interact with the CLI is using YAML-files. After inserting the path to your dataset in the example YAML-files in the `cfg` folder, a ResNet18 model can be trained using `python -m solis fit --config resnet18.yaml`. Similarly, a DeepLabV3 model can be trained using `python -m solis fit --config deeplabv3_resnet50`. Checkpoints and tensorboard logs will be saved in the `lightning_logs` folder. To view the logs in tensorboard, use `tensorboard --logdir lightning_logs`. To resume training from a checkpoint, use `python -m solis fit --config lightning_logs/version_x/config.yaml --ckpt_path lightning_logs/version_x/checkpoints/epoch=y-step=z.ckpt`.

The `solis` module also includes utilities for exporting a trained model to PyTorch or ONNX. These can be run using `python -m solis.export_torch` or `python -m solis.export_onnx` respectively. To export the DeepLabV3 model trained in the previous example to PyTorch, use `python -m solis.export_torch lightning_logs/version_x/checkpoints/epoch=y-step=z.ckpt my_model.pt`. To export the model to ONNX, use `python -m solis.export_onnx solis_lightning.modules.DeepLabV3_ResNet50 lightning_logs/version_x/checkpoints/epoch=y-step=z.ckpt my_model.onnx`.

## Python Usage
Models and datasets are available for both PyTorch and Lightning. The `solis_torch` module contains the PyTorch implementations, while the `solis_lightning` module contains the Lightning implementations.

### solis_torch
```python
from solis_torch.datasets import S2Segmentation
from solis_torch.models import deeplabv3_resnet50
from solis_torch.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
)
import torch

transform = Compose([
    ToTensor(),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
])
dataset = S2Segmentation("path/to/dataset", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, drop_last=True, shuffle=True)

model = deeplabv3_resnet50(num_channels=12, num_classes=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for batch in dataloader:
    x, y = batch
    optimizer.zero_grad()
    y_hat = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat["out"][:, 0], y)
    loss.backward()
    optimizer.step()
```

### solis_lightning
```python
from solis_lightning.datamodules import S2Segmentation
from solis_lightning.modules import DeepLabV3_ResNet50
import lightning.pytorch as pl

class MyModel(DeepLabV3_ResNet50):
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

datamodule = S2Segmentation("path/to/dataset", batch_size=32)
model = MyModel(num_channels=12, num_classes=1)
trainer = pl.Trainer()
trainer.fit(model, datamodule)
```
