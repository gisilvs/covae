
import lightning as L
import matplotlib.pyplot as plt
import torch
import torchvision
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.pytorch import seed_everything
import numpy as np

class DiagnosticCallback(L.Callback):

    def __init__(self, every_n_iterations=1):
        super().__init__()
        self.every_n_iterations = every_n_iterations  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.step % self.every_n_iterations == 0:
            with isolate_rng():
                seed_everything(32, workers=True)
                if isinstance(batch, list):
                    inputs = batch[0]
                    labels = None
                else:
                    inputs = batch
                    labels = None
                # Reconstruct images
                with torch.no_grad():
                    batch_size = inputs.shape[0]
                    recs = [inputs]
                    means = []
                    stds = []
                    latents = []
                    pl_module.model.eval()
                    num_timesteps = pl_module.model._step_schedule(pl_module.step)
                    time_steps = pl_module.model._get_time_steps(num_timesteps, device=inputs.device)
                    #noise = torch.randn([batch_size] + pl_module.model.noise_shape, dtype=torch.float32).to(inputs.device)
                    noise = pl_module.model.sample_noise(batch_size, inputs.device)
                    for t in time_steps[1:]:
                        rec, mu, std, _ = pl_module.ema.eval().precond(inputs, t, noise, labels)
                        recs.append(rec.clamp(-1,1))
                        means.append(mu)
                        stds.append(std)
                        #latents.append(torch.linalg.norm((mu + noise * std).view(batch_size, -1), dim=1))
                        latents.append(torch.linalg.norm((pl_module.model._reparametrized_sample(mu, std, noise)).reshape(batch_size, -1), dim=1))
                    img_rec_diff = [((inputs - r)**2).view(batch_size, -1).sum(1) for r in recs[1:]]
                    #prior = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
                    prior = pl_module.model._get_distribution(torch.zeros_like(mu), torch.ones_like(std))
                    rec_diffs = [((recs[i] - recs[i+1])**2).view(batch_size, -1).sum(1) for i in range(len(recs)-1)]
                    '''kl_diffs = [torch.distributions.kl_divergence(torch.distributions.Normal(means[i], stds[i] + 1e-8),
                                                                  prior).view(batch_size, -1).sum(1) for i in
                                range(len(means))]'''
                    kl_diffs = [torch.distributions.kl_divergence(pl_module.model._get_distribution(means[i], stds[i] + 1e-8),
                                                                  prior).view(batch_size, -1).sum(1) for i in
                                range(len(means))]
                    rec_mean_diffs = np.array([np.mean(d.detach().cpu().numpy()) for d in rec_diffs])
                    rec_std_diffs = np.array([np.std(d.detach().cpu().numpy()) for d in rec_diffs])
                    kl_mean_diffs = np.array([np.mean(d.detach().cpu().numpy()) for d in kl_diffs])
                    kl_std_diffs = np.array([np.std(d.detach().cpu().numpy()) for d in kl_diffs])
                    img_rec_mean_diffs = np.array([np.mean(d.detach().cpu().numpy()) for d in img_rec_diff])
                    img_rec_std_diffs = np.array([np.std(d.detach().cpu().numpy()) for d in img_rec_diff])
                    latent_mean_diffs = np.array([np.mean(d.detach().cpu().numpy()) for d in latents])
                    latent_std_diffs = np.array([np.std(d.detach().cpu().numpy()) for d in latents])
                    time_steps = time_steps.cpu().numpy()

                    fig, ax = plt.subplots()
                    ax.plot(time_steps[1:], rec_mean_diffs)
                    ax.fill_between(time_steps[1:], rec_mean_diffs + rec_std_diffs, rec_mean_diffs - rec_std_diffs, alpha=0.2)
                    ax.set_title(f"Reconstruction Error iter {pl_module.step}")
                    if pl_module.model.time_scale != 'linear':
                        ax.set_xscale('log')
                    ax.set_yscale('log')
                    trainer.logger.log_image(key='reconstruction', images=[fig], step=pl_module.step)
                    plt.cla()
                    plt.close()

                    fig, ax = plt.subplots()
                    ax.plot(time_steps[1:], img_rec_mean_diffs)
                    ax.fill_between(time_steps[1:], img_rec_mean_diffs + img_rec_std_diffs, img_rec_mean_diffs - img_rec_std_diffs,
                                    alpha=0.2)
                    ax.set_title(f"Image Reconstruction Error iter {pl_module.step}")
                    if pl_module.model.time_scale != 'linear':
                        ax.set_xscale('log')
                    ax.set_yscale('log')
                    trainer.logger.log_image(key='img_reconstruction', images=[fig], step=pl_module.step)
                    plt.cla()
                    plt.close()

                    fig, ax = plt.subplots()
                    ax.plot(time_steps[1:], kl_mean_diffs)
                    ax.fill_between(time_steps[1:], kl_mean_diffs + kl_std_diffs, kl_mean_diffs - kl_std_diffs,
                                    alpha=0.2)
                    ax.set_title(f"KL Error iter {pl_module.step}")
                    if pl_module.model.time_scale != 'linear':
                        ax.set_xscale('log')
                    ax.set_yscale('log')
                    trainer.logger.log_image(key='kl', images=[fig], step=pl_module.step)
                    plt.cla()
                    plt.close()

                    fig, ax = plt.subplots()
                    ax.plot(time_steps[1:], latent_mean_diffs)
                    ax.fill_between(time_steps[1:], latent_mean_diffs + latent_std_diffs, latent_mean_diffs - latent_std_diffs,
                                    alpha=0.2)
                    ax.set_title(f"Latent magnitude {pl_module.step}")
                    if pl_module.model.time_scale != 'linear':
                        ax.set_xscale('log')
                    ax.set_yscale('log')
                    trainer.logger.log_image(key='latent_magnitude', images=[fig], step=pl_module.step)
                    plt.cla()
                    plt.close()

                    pl_module.model.train()
