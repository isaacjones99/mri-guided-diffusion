import lightning as L

from torch.utils.data import DataLoader
from .data_module import IXIDataset
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

class IXIDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./datasets/data",
        batch_size: int = 4,
        image_size: int = 128
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

    def setup(self, stage: str = "fit"):
        self.train_ds = IXIDataset(self.data_dir, train=True, image_size=self.image_size)
        self.validation_ds = IXIDataset(self.data_dir, train=False, image_size=self.image_size)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataloader,
            batch_size=self.batch_size,
            shuffle=True
        )
