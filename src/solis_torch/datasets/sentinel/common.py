from torch.utils.data import Dataset
from pathlib import Path


class S2Dataset(Dataset):
    def __init__(self, root_dir: str, cls: str):
        super().__init__()
        self.image_dir = Path(root_dir) / "images"
        self.target_dir = Path(root_dir) / "targets" / cls

        self.images = list(self.image_dir.glob("*.tif"))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        target_path = self.target_dir / image_path.name
        return image_path, target_path
