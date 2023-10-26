from .common import S2Dataset
from pathlib import Path

import numpy as np
import rasterio


class S2Segmentation(S2Dataset):
    def __init__(self, root_dir: str, cls: str, transform=None):
        super().__init__(root_dir, cls)
        self.transform = transform

        self.targets = list(self.targets.intersection(self.images))

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        filename = self.targets[index]

        with rasterio.open(self.get_image_path(filename)) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(self.get_target_path(filename)) as src:
            target = src.read(1).astype(np.float32)

        if self.transform:
            image, target = self.transform(image, target)

        return image, target