
import lightning as L
import matplotlib.pyplot as plt
import torch
import torchvision
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.pytorch import seed_everything


class GenerateCallback(L.Callback):

    def __init__(self, sample_shape, n_iters, use_ema, every_n_iterations=1, plot_type='grid', plot_rec=False, rescale=True):
        super().__init__()
        self.sample_shape = sample_shape
        self.every_n_iterations = every_n_iterations  # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.n_iters = n_iters
        self.plot_type = plot_type
        self.use_ema = use_ema
        self.plot_rec = plot_rec
        self.rescale = rescale

    def log_samples(self, samples, iter, trainer, pl_module):
        if self.rescale:
            fig = torchvision.utils.make_grid(samples.clamp(-1, 1), nrow=16, value_range=(-1, 1), padding=0)
        else:
            fig = torchvision.utils.make_grid(torch.nn.functional.sigmoid(samples), nrow=16, value_range=(0, 1), padding=0)
        key = f"samples {iter} iters"
        if self.use_ema:
            key += " ema"
        trainer.logger.log_image(key=key, images=[fig], step=pl_module.step)

    def log_rec(self, samples, iter, trainer, pl_module):
        samples = samples[:64]
        if self.rescale:
            fig = torchvision.utils.make_grid(samples.clamp(-1, 1), nrow=8, value_range=(-1, 1), padding=0)
        else:
            fig = torchvision.utils.make_grid(torch.nn.functional.sigmoid(samples), nrow=8, value_range=(0, 1), padding=0)
        key = f"rec {iter} iters"
        trainer.logger.log_image(key=key, images=[fig], step=pl_module.step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.step % self.every_n_iterations == 0:
            with isolate_rng():
                seed_everything(32, workers=True)
                # Reconstruct images
                with torch.no_grad():
                    pl_module.model.eval()
                    samples = pl_module.sample(self.sample_shape, self.n_iters, self.use_ema)
                    pl_module.model.train()

                # Plot and add to tensorboard
                self.log_samples(samples, iter=self.n_iters, trainer=trainer, pl_module=pl_module)

            if self.plot_rec:
                with isolate_rng():
                    seed_everything(32, workers=True)
                    with torch.no_grad():
                        # Reconstruct images
                        if isinstance(batch, list):
                            data = batch[0].to(pl_module.device)
                        else:
                            data = batch.to(pl_module.device)
                        samples = pl_module.ema.eval().encode_decode(data)

                    # Plot and add to tensorboard
                    self.log_rec(samples, iter=self.n_iters, trainer=trainer, pl_module=pl_module)

