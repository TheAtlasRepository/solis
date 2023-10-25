import lightning.pytorch as pl
import solis_torch.datasets as datasets
import solis_torch.transforms as transforms
import torch


# mean = (1318.93, 1100.02, 1063.16, 1006.47, 1303.49, 2283.09, 2757.43, 2708.51, 3054.00, 876.21, 15.05, 2175.90, 1315.71)
# std  = ( 281.68,  354.13,  422.74,  673.23,  624.52,  718.17,  934.52,  939.38, 1027.30, 342.40,  8.75, 1025.45,  855.77)


class S2Segmentation(pl.LightningDataModule):
    def __init__(self, root_dir: str, cls: str, batch_size: int, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

        dataset = datasets.S2Segmentation(
            root_dir, cls,
            transform=transform)

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [0.8, 0.2])
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)
