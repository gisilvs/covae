import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

class CoVAEBase(nn.Module):
    def __init__(self,
                 model,
                 step_schedule,
                 sigma_min,
                 sigma_max,
                 rho,
                 start_scales,
                 end_scales,
                 total_training_steps,
                 noise_shape,
                 loss_mode,
                 denoiser_loss_mode,
                 use_gan,
                 gan_warmup_steps,
                 discriminator,
                 gan_lambda,
                 ):

        super().__init__()
        self.model = model
        self.step_schedule = step_schedule
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.start_scales = start_scales
        self.end_scales = end_scales
        self.total_training_steps = total_training_steps
        self.noise_shape = noise_shape
        self.loss_mode = loss_mode
        self.denoiser_loss_mode = denoiser_loss_mode
        self.use_gan = use_gan
        self.gan_warmup_steps = gan_warmup_steps
        self.discriminator = discriminator
        self.gan_lambda = gan_lambda

    def _append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    def _loss_fn(self, prediction, target, loss_mode):
        if loss_mode == 'l2':
            return (prediction - target) ** 2
        elif loss_mode == 'huber':
            c = 0.00054 * math.sqrt(prediction[0].numel())
            return torch.sqrt((prediction - target) ** 2 + c ** 2) - c
        elif loss_mode == 'bce':
            return F.binary_cross_entropy_with_logits(prediction, target, reduction='none')
        else:
            raise NotImplementedError

    def _get_sigmas_karras(
            self,
            num_timesteps: int,
            device: torch.device = None,
    ) -> Tensor:
        rho_inv = 1.0 / self.rho
        # Clamp steps to 1 so that we don't get nans
        steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
        sigmas = self.sigma_min ** rho_inv + steps * (
                self.sigma_max ** rho_inv - self.sigma_min ** rho_inv
        )
        sigmas = sigmas ** self.rho

        return sigmas

    def _step_schedule(self, step):

        if self.step_schedule == 'exp':
            k_prime = math.floor(
                self.total_training_steps
                / (math.log2(math.floor(self.end_scales / self.start_scales)) + 1)
            )
            num_timesteps = self.start_scales * math.pow(2, math.floor(step / k_prime))
            num_timesteps = min(num_timesteps, self.end_scales) + 1

        elif self.step_schedule == 'none':
            num_timesteps = self.end_scales + 1

        else:
            raise NotImplementedError

        return int(num_timesteps)

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def discriminator_loss(self, real, fake):
        logits_real = self.discriminator(real.contiguous().detach())
        logits_fake = self.discriminator(torch.clamp(fake.contiguous().detach(), -1,1))
        loss = self.hinge_d_loss(logits_real, logits_fake)
        return loss