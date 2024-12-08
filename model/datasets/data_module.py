import lightning as L

from PIL import Image
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms

class DataModule(L.LightningDataModule):
    
    def __init__(
        self,
        data_dir: str = "./datasets/data",
        batch_size: int = 4,
        image_size: int = 256
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ds = None
        self.val_ds = None
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage: str = "fit"):
        self.train_ds = None
        self.val_ds = None

        # Preprocess data
        

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )
    
    def val_dataloadder(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )