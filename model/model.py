from typing import Any, Tuple

import lightning as L
import torch
from diffusers import DDPMScheduler
from torch import optim

# Local imports
from ..model.guided_diffusion import dist_util
from ..model.guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


class DiffusionModel(L.LightningModule):

    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        self.unet, self.diffusion = create_model_and_diffusion(
            **args_to_dict(config["model"], model_and_diffusion_defaults())
        )

        self.scheduler = DDPMScheduler(**self.config["scheduler"])

        self.set_weights()

        # Set precision
        if self.config["model"]["use_fp16"]:
            self.unet.convert_to_fp16()

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get images and label from batch
        images = batch["images"]
        labels = batch["labels"]
        batch_size = self.config["model"]["batch_size"]

        # Sample noise
        noise = torch.randn(images.shape).to(images.device)

    def set_weights(self) -> None:
        state_dict = dist_util.load_state_dict(
            self.config["weights"]["save_dir"], map_location="cpu"
        )
        self.unet.load_state_dict(state_dict)

