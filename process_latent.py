
import sys

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict
import time

import torch
import wandb

from lightning_modules.lightning_cm import LightningConsistencyModel
from utils.callback_utils import get_callbacks, get_delete_checkpoints_callback
from utils.datamodule_utils import get_datamodule
from utils.naming_utils import get_run_name
from utils.model_utils import get_model
from wandb_config import key
from lightning.pytorch.utilities import rank_zero_only
from pathlib import Path
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.pytorch import seed_everything
from torchmetrics.image.fid import FrechetInceptionDistance
from utils.utils import rescaling_inv, adjust_channels

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    wandb.login(key=key)
    logger = WandbLogger()
    run_path = Path(cfg.run_path)
    run_path = run_path.with_name(f'model-{run_path.name}')
    checkpoint_reference = f'{run_path}:best'
    logger.download_artifact(checkpoint_reference, save_dir=cfg.root_dir, artifact_type="model")
    checkpoint_path = Path(cfg.root_dir) / "model.ckpt"
    model = LightningConsistencyModel.load_from_checkpoint(checkpoint_path)
    root_dir = cfg.root_dir
    data_dir = cfg.dataset.data_dir
    cfg = model.cfg
    cfg.root_dir = root_dir
    cfg.dataset.data_dir = data_dir
    L.seed_everything(cfg.seed, workers=True)

    dm = get_datamodule(cfg)
    dm.prepare_data()
    dm.setup(stage="fit")

    t_step = 0.05
    mus = []
    stds = []
    latent_samples = []
    with torch.no_grad():
        model.eval()
        seed_everything(32, workers=True)
        for batch in dm.train_dataloader():
            if isinstance(batch, list):
                data = batch[0].to(model.device)
            else:
                data = batch.to(model.device)
            t = torch.ones(data.shape[0], dtype=torch.float32, device=data.device) * t_step
            mu, std = model.model.encode(data, t, None)
            latent_sample = torch.distributions.Normal(mu, std).sample()
            mus.append(mu)
            stds.append(std)
            latent_samples.append(latent_sample)

        mus = torch.cat(mus, dim=0)
        stds = torch.cat(stds, dim=0)
        latent_samples = torch.cat(latent_samples, dim=0)
        mean = latent_samples.mean()
        var = ((latent_samples - mean) ** 2).mean()
        std = var.sqrt()

        # Scale factor to get unit std latents
        scale = std.item()

    out = {
        "mu": mus.cpu(),
        "std": stds.cpu(),
        "t": t_step,
        "scale": scale,
    }
    torch.save(out, Path('/home/gsilvestri/latent_data') / f"covae_latent_{t_step}.pt")


if __name__ == "__main__":
    main()