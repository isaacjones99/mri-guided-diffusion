import argparse
import logging
import os

import numpy
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

# from datasets.data_module import IXIDataset
from .datasets.data_module import IXIDataset
from .model import DiffusionModel
from .utils import load_yaml_config, update_config, set_seeds, get_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="Batch size (default: 4)")
    parser.add_argument("--max_epochs", type=int, help="Number of epochs (default: 10)")
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of subprocesses to use for data loading (default: 0)"
    )
    parser.add_argument("--lr", type=float, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--pretrained",
        type=bool,
        help="Use a pretrained model (default: false)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = update_config(load_yaml_config, vars(args))

    set_seeds()

    config["device"] = device = get_device()
    logger.info(f"Deivce set to {device}")

    data = IXIDataset(**config["data"]["ixi"], train=True)
    train_loader = DataLoader(data, batch_size=4, shuffle=True)

    # Init diffusion model
    model = DiffusionModel(config)

    trainer = Trainer(accelerator=device, **config["trainer"])

    # Training loop
    logger.info("Training")
    trainer.fit(model, data)
    logger.info("Finished training")


if __name__ == "__main__":
    main()