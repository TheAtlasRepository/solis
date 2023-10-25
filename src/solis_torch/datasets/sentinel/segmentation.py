from .common import S2Dataset
from pathlib import Path

import numpy as np
import rasterio


class S2Segmentation(S2Dataset):
    def __init__(self, root_dir: str, cls: str, transform=None):
        super().__init__(root_dir, cls)
        self.transform = transform

        self.images = list(self.image_dir.glob("*.tif"))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path, target_path = super().__getitem__(index)

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        if target_path.exists():
            with rasterio.open(target_path) as src:
                target = src.read(1).astype(np.float32)
        else:
            target = np.zeros(image.shape[1:], dtype=np.float32)

        if self.transform:
            image, target = self.transform(image, target)

        return image, target