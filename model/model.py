import lightning as L
import torch

from torch import optim

class DiffusionModel(L.LightningModule):

    def __init__(self):
        super().__init__()
        