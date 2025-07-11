import math

import torch
from lightning.pytorch.utilities.seed import isolate_rng
from models.covae_base import CoVAEBase

class CoVAESimple(CoVAEBase):
    def __init__(self,
                 kernel,
                 p_mean,
                 p_std,
                 sigma_data,
                 norm_strength,
                 mid_t,
                 norm_weight,
                 noise_schedule,
                 **cm_kwargs
                 ):

        super().__init__(**cm_kwargs)
        self.kernel = kernel
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.mid_t = mid_t
        self.norm_strength = norm_strength
        self.norm_weight = norm_weight
        self.noise_schedule = noise_schedule
        self.layer_norm = torch.nn.LayerNorm(self.noise_shape)
        assert norm_weight in ['fixed', 'adaptive']

    def _get_time_steps(self, num_timesteps, device):
        time_steps = self._get_sigmas_karras(num_timesteps, device)
        return time_steps

    def _lognormal_timestep_distribution(self, num_samples, sigmas):
        pdf = torch.erf((torch.log(sigmas[1:]) - self.p_mean) / (self.p_std * math.sqrt(2))) - torch.erf(
            (torch.log(sigmas[:-1]) - self.p_mean) / (self.p_std * math.sqrt(2))
        )
        indices = torch.multinomial(pdf, num_samples, replacement=True)

        return indices

    def _get_indices(self, num_timesteps, sigmas, device, batch_size):
        if self.noise_schedule == 'uniform':
            indices = torch.randint(
                0, num_timesteps - 1, (batch_size,), device=device
            )
        elif self.noise_schedule == 'lognormal':
            indices = self._lognormal_timestep_distribution(batch_size, sigmas)
        else:
            raise ValueError(f'Unknown noise schedule')

        return indices

    def _get_loss_weights(self, sigmas, indices):
        return (1 / (sigmas[1:] - sigmas[:-1]))[indices]

    def sample_noise(self, batch_size, device):
        return torch.randn([batch_size] + self.noise_shape, dtype=torch.float32).to(device)

    def decode(self, z, t, class_labels):
        z = z.to(torch.float32)
        t = self._append_dims(t, z.ndim).to(z).to(torch.float32)
        c_noise = t.log() / 4
        emb = self.model.time_embedding(c_noise.flatten(), class_labels)
        c_in, c_skip, c_out = self.kernel.get_scaling_factors(t)
        x = self.model.decoder(c_in * z, emb)
        if self.denoiser_loss_mode:
            x, denoiser_x = torch.chunk(x, 2, dim=1)
            x = denoiser_x.detach() + c_out * x.to(torch.float32)
        else:
            denoiser_x = None
        return x, denoiser_x

    def encode(self, x, t, class_labels):
        x = x.to(torch.float32)
        t = self._append_dims(t, x.ndim).to(x).to(torch.float32)
        c_noise = t.log() / 4
        emb = self.model.time_embedding(c_noise.flatten(), class_labels)
        mu, _ = self.model.encoder(x, torch.zeros_like(emb))
        if self.norm_strength > 0:
            return mu
        else:
            mu = torch.nn.functional.tanh(self.layer_norm(mu))
            return mu

    def encode_decode(self, x, idx=1, class_labels=None, noise=None):
        device = x.device
        batch_size = x.shape[0]
        time_steps = self._get_time_steps(self.end_scales + 1, device=device)
        t = time_steps[idx].to(device)
        if not torch.is_tensor(noise):
            noise = self.sample_noise(batch_size, device)
        z_0 = self.encode(x, t, class_labels)
        z_t = self.kernel.forward(z_0, t, noise)
        x, _ = self.decode(z_t, t, class_labels)
        return x

    def loss(self, x, step, labels=None):

        log_dict = {}
        dims = x.ndim  # keeps track of data dimensionality to work with both images and tabular
        device = x.device
        batch_size = x.shape[0]
        num_timesteps = self._step_schedule(step)
        time_steps = self._get_time_steps(num_timesteps, device=device)
        idxs = self._get_indices(len(time_steps) - 1, time_steps, device=device, batch_size=batch_size)
        t = time_steps[idxs + 1].to(device)
        r = time_steps[idxs].to(device)
        noise = self.sample_noise(batch_size, device)

        z_0 = self.encode(x, t, labels) # t is passed but not used
        t = self._append_dims(t, z_0.ndim).to(z_0).to(torch.float32)
        r = self._append_dims(r, z_0.ndim).to(z_0).to(torch.float32)
        z_t = self.kernel.forward(z_0, t, noise)
        z_r = self.kernel.forward(z_0, r, noise)

        with isolate_rng():
            x_t, denoiser_x = self.decode(z_t, t, labels)

        if (idxs == 0).all():
            # save time when training simple vae
            x_r = x
        else:
            with torch.no_grad():
                x_r, _ = self.decode(z_r, r, labels)

        # boundary condition
        x_r = torch.where(self._append_dims(idxs > 0, dims).to(device), x_r, x)

        rec_loss = self._loss_fn(x_t, x_r.detach(), self.loss_mode)
        log_dict['rec_loss'] = rec_loss.detach().view(batch_size, -1).sum(1).mean()
        rec_loss = (rec_loss).view(batch_size, -1).sum(1, keepdim=True)
        total_loss = rec_loss
        loss_weights = self._append_dims(self._get_loss_weights(time_steps, idxs), total_loss.ndim)
        total_loss = total_loss * loss_weights
        norm = (z_0 ** 2).flatten(1).sum(1, keepdim=True)
        std = torch.std(z_0.flatten(1), dim=1).detach().mean()
        log_dict['std'] = std
        log_dict['norm'] = norm.detach().mean()
        if self.norm_strength > 0:
            if self.norm_weight == 'fixed':
                norm_weights = self._append_dims(self._get_loss_weights(time_steps, torch.zeros_like(idxs)), total_loss.ndim)
            else:
                norm_weights = loss_weights

            norm *= self.norm_strength * norm_weights
            total_loss += norm

        if self.denoiser_loss_mode:
            denoiser_loss = self._loss_fn(denoiser_x, x, self.denoiser_loss_mode)
            log_dict['denoiser_loss'] = denoiser_loss.detach().view(batch_size, -1).sum(1).mean()
            c_in, c_skip, c_out = self.kernel.get_scaling_factors_bc(self._append_dims(t, x.ndim).to(torch.float32))
            denoiser_loss = denoiser_loss.view(batch_size, -1).sum(1, keepdim=True) * loss_weights
            total_loss = c_skip * denoiser_loss + total_loss

        if self.use_gan and step >= self.gan_warmup_steps:
            if self.denoiser_loss_mode:
                gan_input = torch.where(self._append_dims(idxs > 1, dims).to(device), x_t, denoiser_x)
            else:
                gan_input = x_t
            gan_loss = -torch.mean(self.discriminator(torch.clamp(gan_input, -1, 1).contiguous()).view(batch_size, -1), dim=1)
            mask_idx = int(self.end_scales / ((self.total_training_steps - self.gan_warmup_steps)  / (step + 1 - self.gan_warmup_steps)))
            all_timesteps = self._get_time_steps(self.end_scales + 1, device=device)
            mask = torch.where(t > all_timesteps[mask_idx], 0., 1.).to(device)
            gan_loss = gan_loss * mask
            log_dict['generator_loss'] = gan_loss.detach().mean()
            gan_loss = gan_loss * loss_weights * ((step + 1 - self.gan_warmup_steps)/ (self.total_training_steps - self.gan_warmup_steps)) * self.gan_lambda
            gan_loss = gan_loss.mean()
        else:
            gan_loss = 0.

        return total_loss.mean() + gan_loss, log_dict, x_t

    @torch.no_grad()
    def sample(self, sample_shape, n_iters, device, class_labels=None, idx=None, temperature=1):
        time_steps = self._get_time_steps(self.end_scales + 1, device)
        t = torch.ones(sample_shape[0], device=device) * time_steps[-1]
        noise = self.sample_noise(sample_shape[0], device)
        t = self._append_dims(t, noise.ndim).to(noise).to(torch.float32)
        z = noise * t
        x, _ = self.decode(z, t, class_labels)
        if idx is None:
            if time_steps.shape[0] > 2:
                idx = round((self.end_scales + 1) * 0.5)
            else:
                idx = 1
        for i in range(1, n_iters):
            t = torch.ones(sample_shape[0], device=device) * time_steps[idx].to(device)
            noise = self.sample_noise(sample_shape[0], device)
            z_0 = self.encode(x, t, class_labels)
            t = self._append_dims(t, z_0.ndim).to(z_0).to(torch.float32)
            z_t = self.kernel.forward(z_0, t, noise)
            x, _ = self.decode(z_t, t, class_labels)

        return x