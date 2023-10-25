from .common import S2Dataset

import numpy as np
import rasterio


class S2Classification(S2Dataset):
    def __init__(self, root_dir: str, cls: str, transform=None, target_transform=None):
        super().__init__(root_dir, cls)
        self.transform = transform
        self.target_transform = target_transform

        self.images = list(self.image_dir.glob("*.tif"))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path, target_path = super().__getitem__(index)

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        if target_path.exists():
            target = 1
        else:
            target = 0

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)

        return image, target