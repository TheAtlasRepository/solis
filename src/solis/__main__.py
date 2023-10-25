from solis_lightning.datamodules import S2Classification, S2Segmentation
from solis_lightning.modules import ResNet18, ResNet34, ResNet50, ResNet101
from solis_lightning.modules import DeepLabV3ResNet50, DeepLabV3ResNet101
from lightning.pytorch.cli import LightningCLI


if __name__ == '__main__':
    LightningCLI()
