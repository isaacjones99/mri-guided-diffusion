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
        # return image, label
        return {
            "images": image,
            "labels": label - 100, # Bit messy, but it works.
        }
    