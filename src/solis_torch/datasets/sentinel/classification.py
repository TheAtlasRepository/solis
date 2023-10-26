from .common import S2Dataset

import numpy as np
import rasterio


class S2Classification(S2Dataset):
    def __init__(self, root_dir: str, cls: str, transform=None, target_transform=None):
        super().__init__(root_dir, cls)
        self.transform = transform
        self.target_transform = target_transform

        self.images = list(self.images)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        filename = self.images[index]

        with rasterio.open(self.get_image_path(filename)) as src:
            image = src.read().astype(np.float32)

        target = 1 if filename in self.targets else 0

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)

        return image, target