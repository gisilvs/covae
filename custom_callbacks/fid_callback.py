
import lightning as L
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.pytorch import seed_everything

from utils.utils import rescaling_inv, adjust_channels


class FIDCallback(L.Callback):

    def __init__(self, sample_shape, n_iters, n_dataset_samples, every_n_iterations=1, compute_rec_fid=False, rescale=True):
        super().__init__()
        self.sample_shape = sample_shape
        self.every_n_iterations = every_n_iterations  # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.n_iters = n_iters
        self.n_dataset_samples = n_dataset_samples
        with isolate_rng():
            self.fid = FrechetInceptionDistance(reset_real_features=False, normalize=True)
        self.best = torch.inf
        self.compute_rec_fid = compute_rec_fid
        self.rescale = rescale

    def on_train_start(self, trainer, pl_module):
        with isolate_rng():
            seed_everything(32, workers=True)
            with torch.no_grad():
                self.fid = self.fid.to(pl_module.device)
                for batch in trainer.datamodule.fid_dataloader():
                    if isinstance(batch, list):
                        data = batch[0].to(pl_module.device)
                    else:
                        data = batch.to(pl_module.device)
                    data = adjust_channels(data)
                    self.fid.update(data, real=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.step % self.every_n_iterations == 0:
            with isolate_rng():
                seed_everything(32, workers=True)
                with torch.no_grad():
                    pl_module.model.eval()
                    self.fid = self.fid.to(pl_module.device)
                    self.fid.reset()
                    total_n_samples = 0

                    while total_n_samples < self.n_dataset_samples:
                        samples = pl_module.sample(self.sample_shape, self.n_iters, use_ema=True)
                        n_samples = samples.shape[0]

                        # check how many samples are left to reach our target number, if too many take a subset of the latest batch
                        if total_n_samples + n_samples > self.n_dataset_samples:
                            n_samples = self.n_dataset_samples - total_n_samples
                            samples = samples[:n_samples]

                        if self.rescale:
                            samples = rescaling_inv(samples.clamp(-1, 1))
                        else:
                            samples = torch.nn.functional.sigmoid(samples)
                        samples = adjust_channels(samples)
                        self.fid.update(samples, real=False)
                        total_n_samples += n_samples
                fid = self.fid.compute()


                pl_module.model.train()

                if fid < self.best:
                    self.best = fid
                pl_module.log(f"best_FID_{self.n_iters}_iters", self.best, on_step=True, on_epoch=False, prog_bar=True, logger=True)

                pl_module.log(f"FID_{self.n_iters}_iters", fid, on_step=True, on_epoch=False, prog_bar=True, logger=True)

            if self.compute_rec_fid:
                with isolate_rng():
                    seed_everything(32, workers=True)
                    with torch.no_grad():
                        pl_module.model.eval()
                        self.fid = self.fid.to(pl_module.device)
                        self.fid.reset()
                        for batch in trainer.datamodule.train_dataloader():
                            if isinstance(batch, list):
                                data = batch[0].to(pl_module.device)
                            else:
                                data = batch.to(pl_module.device)
                            samples = pl_module.ema.eval().encode_decode(data)
                            if self.rescale:
                                samples = rescaling_inv(samples.clamp(-1, 1))
                            else:
                                samples = torch.nn.functional.sigmoid(samples)
                            samples = adjust_channels(samples)
                            self.fid.update(samples, real=False)

                    fid = self.fid.compute()

                    pl_module.model.train()

                    pl_module.log(f"rec_FID_{self.n_iters}_iters", fid, on_step=True, on_epoch=False, prog_bar=True,
                                  logger=True)
