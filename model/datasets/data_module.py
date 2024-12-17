# import lightning as L

# from PIL import Image

# from torch.utils.data import DataLoader
# from torchvision import transforms

# class DataModule(L.LightningDataModule):
    
#     def __init__(
#         self,
#         data_dir: str = "./datasets/data",
#         batch_size: int = 4,
#         image_size: int = 256
#     ) -> None:
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.train_ds = None
#         self.val_ds = None
#         self.image_size = image_size
#         self.transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Resize((image_size, image_size)),
#                 transforms.Normalize(mean=[0.5], std=[0.5]),
#             ]
#         )

#     def prepare_data(self):
#         pass

#     def setup(self, stage: str = "fit"):
#         self.train_ds = None
#         self.val_ds = None

#         # Preprocess data
        

#     def train_dataloader(self) -> TRAIN_DATALOADERS:
#         return DataLoader(
#             self.train_ds,
#             batch_size=self.batch_size,
#             shuffle=True,
#         )
    
#     def val_dataloadder(self) -> EVAL_DATALOADERS:
#         return DataLoader(
#             self.val_ds,
#             batch_size=self.batch_size,
#             shuffle=False,
#         )

import os
from typing import List, Tuple

import blobfile as bf
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class IXIDataset(Dataset):
    def __init__(self, data_dir: str, train: bool = True, image_size: int = 128):
        self.data_dir = os.path.join(data_dir, "train" if train else "validation")
        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
            ]
        )

        # Load images and labels
        self.image_paths = self._get_image_paths()
        self.labels = self._get_image_labels()
    
    def _is_valid_image(self, filename: str) -> bool:
        valid_extensions = {"jpg", "jpeg", "png", "gif"}
        return "." in filename and filename.split(".")[-1].lower() in valid_extensions
    
    def _get_image_paths(self) -> List[str]:
        return [item for item in bf.listdir(self.data_dir) if self._is_valid_image(item)]
    
    def _get_image_labels(self) -> List[int]:
        labels = []
        for item in self.image_paths:
            try:
                label = int(item.split("_")[0])
                labels.append(label)
            except ValueError:
                raise ValueError(f"Invalid label format in file: {item}")
        return labels
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.data_dir, self.image_paths[idx])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image at img_path: {e}")
        
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
