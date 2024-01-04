import lightning.pytorch as pl
import solis_torch.datasets as datasets
import torch
import torchvision.transforms.v2 as transforms


class S2Classification(pl.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int = 1, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        transform = transforms.Compose([
            torch.from_numpy,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

        dataset = datasets.S2Classification(
            root_dir, transform=transform)

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [0.8, 0.2])
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True)
