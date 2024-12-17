from typing import Any, Tuple

# import lightning as L
from pytorch_lightning import LightningModule

import torch
from diffusers import DDPMScheduler
from torch import optim
from torch.nn import functional as F

# Local imports
from .guided_diffusion import dist_util
from .guided_diffusion.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


class DiffusionModel(LightningModule):

    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        # Unet model and diffusion scheduler
        config["model"].update(model_and_diffusion_defaults())
        
        self.unet, self.diffusion = create_model_and_diffusion(
            **args_to_dict(config["model"], model_and_diffusion_defaults())
        )

        self.scheduler = DDPMScheduler(**self.config["scheduler"])

        # Set precision
        if self.config["model"]["use_fp16"]:
            self.unet.convert_to_fp16()

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get images and label from batch
        images = batch["images"]
        labels = batch["labels"]
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")

        batch_size = self.config["model"]["batch_size"]

        # Sample noise
        noise = torch.randn(images.shape).to(images.device)
        print(f"Noise shape: {noise.shape}")

        print(f"Forward batch size = {batch_size}")
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=images.device,
        ).long()
        print(f"Timesteps shape: {timesteps.shape}")

        # Forward diffusion process
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        print(f"Noisy images shape: {timesteps.shape}")

        # Predict the noise
        noise_pred = self.unet(noisy_images, timesteps, labels).to(images.device)

        return noise_pred, noise

    def _common_step(self, batch, batch_idx) -> torch.Tensor:
        noise_pred, noise = self.forward(batch)
        loss = F.mse_loss(noise_pred, noise)
        print(f"mse_loss = {loss}")
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        print(f"Training step loss: {loss}")
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._common_step(batch, batch_idx)
        print(f"Validation step loss: {loss}")
        return loss
    
    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.config["model"]["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def set_weights(self) -> None:
        state_dict = dist_util.load_state_dict(
            self.config["weights"]["save_dir"], map_location="cpu"
        )
        self.unet.load_state_dict(state_dict)

