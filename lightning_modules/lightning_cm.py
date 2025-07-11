
import lightning as L
import torch
from omegaconf import DictConfig
import copy


class LightningConsistencyModel(L.LightningModule):
    def __init__(self, cfg: DictConfig, model):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = model
        self.ema_rate = cfg.model.ema_rate
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.automatic_optimization = False
        self.step = 0

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.model.model.parameters(), lr=self.cfg.model.learning_rate,
                                 weight_decay=self.cfg.model.weight_decay)
        if self.cfg.model.use_gan:
            discriminator_opt = torch.optim.RAdam(self.model.discriminator.parameters(), lr=self.cfg.model.learning_rate,
                                 weight_decay=self.cfg.model.weight_decay)
            return [opt, discriminator_opt], []
        return [opt], []

    @torch.no_grad()
    def ema_update(self):
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, self.ema_rate))

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            inputs = batch[0]
            labels = None
        else:
            inputs = batch
            labels = None

        if self.cfg.model.use_gan:
            opt, discriminator_opt = self.optimizers()
        else:
            opt = self.optimizers()

        self.toggle_optimizer(opt)
        loss, log_dict, x_generated = self.model.loss(inputs, self.step, labels=labels)

        self.manual_backward(loss)
        if self.cfg.gradient_clip_val > 0:
            self.clip_gradients(opt, gradient_clip_val=self.cfg.gradient_clip_val, gradient_clip_algorithm="norm")
        opt.step()
        opt.zero_grad()
        self.untoggle_optimizer(opt)

        if self.cfg.model.use_gan and self.step >= self.model.gan_warmup_steps:
            self.toggle_optimizer(discriminator_opt)
            discriminator_loss = self.model.discriminator_loss(inputs, x_generated)
            log_dict['discriminator_loss'] = discriminator_loss
            self.manual_backward(discriminator_loss)
            discriminator_opt.step()
            discriminator_opt.zero_grad()
            self.untoggle_optimizer(discriminator_opt)

        self.ema_update()
        self.step += 1

        for key, value in log_dict.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    @torch.no_grad()
    def sample(self, sample_shape, n_iters, use_ema=True, class_labels=None, idx=None, temperature=1):
        if use_ema:
            model = self.ema.eval()
        else:
            model = self.model
        return model.sample(sample_shape, n_iters, self.device, class_labels=class_labels, idx=idx, temperature=temperature)

