from torch.utils.data import Dataset
from pathlib import Path


class S2Dataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.image_dir = Path(root_dir) / "images"
        self.target_dir = Path(root_dir) / "targets"

        self.images = {image.name for image in self.image_dir.glob("*.tif")}
        self.targets = {target.name for target in self.target_dir.glob("*.tif")}

    def get_image_path(self, filename):
        return self.image_dir / filename
    
    def get_target_path(self, filename):
        return self.target_dir / filename
