from pickle import FALSE

from omegaconf import DictConfig

from models.covae import CoVAE
from models.covae_simple import CoVAESimple
from networks.autoencoder import AutoEncoder
from networks.discriminator import NLayerDiscriminator, weights_init
from kernels.linear_interpolant import LinearInterpolantKernel
from kernels.variance_exploding import VarianceExplodingKernel

def get_kernel(cfg: DictConfig):
    if cfg.model.kernel == 've':
        kernel = VarianceExplodingKernel(cfg.model.sigma_min, cfg.model.sigma_max, cfg.model.sigma_data)
    elif cfg.model.kernel == 'li':
        kernel = LinearInterpolantKernel(cfg.model.sigma_min, cfg.model.sigma_max, cfg.model.sigma_data)
    return kernel

def get_discriminator(cfg: DictConfig):
    return NLayerDiscriminator(
        input_nc=cfg.dataset.in_channels,
        n_layers=3,
        use_actnorm=False,
    ).apply(weights_init)

def get_neural_net(cfg: DictConfig, pretrained_net):
    if cfg.network.name == 'autoencoder':
        net = AutoEncoder(
            img_resolution=cfg.dataset.img_resolution,
            in_channels=cfg.dataset.in_channels,
            out_channels=cfg.dataset.out_channels,
            label_dim=0,
            model_channels=cfg.network.model_channels,
            channel_mult_enc=cfg.network.channel_mult_enc,
            num_blocks_enc=cfg.network.num_blocks_enc,
            channel_mult_dec=cfg.network.channel_mult_dec,
            num_blocks_dec=cfg.network.num_blocks_dec,
            attn_resolutions=cfg.network.attn_resolutions,
            dropout=cfg.network.dropout,
            z_channels=cfg.network.z_channels,
            final_dim=cfg.network.final_dim,
        )
    else:
        raise NotImplementedError

    return net

def get_model(cfg: DictConfig, pretrained_net=None):
    net = get_neural_net(cfg, pretrained_net)

    latent_size = cfg.dataset.img_resolution // (2 ** (len(cfg.network.channel_mult_enc) - 1))
    if isinstance(cfg.network.final_dim, int):
        noise_shape = [cfg.network.final_dim]
    else:
        noise_shape = [cfg.network.z_channels, latent_size, latent_size]

    if cfg.model.use_gan:
        discriminator = get_discriminator(cfg)
    else:
        discriminator = None

    cm_kwargs = {
        'model': net,
        'step_schedule': cfg.model.step_schedule,
        'sigma_min': cfg.model.sigma_min,
        'sigma_max': cfg.model.sigma_max,
        'rho': cfg.model.rho,
        'start_scales': cfg.model.start_scales,
        'end_scales': cfg.model.end_scales,
        'total_training_steps': cfg.model.total_training_steps,
        'noise_shape': noise_shape,
        'loss_mode': cfg.model.loss_mode,
        'denoiser_loss_mode': cfg.model.denoiser_loss_mode,
        'use_gan': cfg.model.use_gan,
        'gan_warmup_steps': cfg.model.gan_warmup_steps,
        'discriminator': discriminator,
        'gan_lambda': cfg.model.gan_lambda,
    }
    if cfg.model.name == 'covae':
        return CoVAE(**cm_kwargs,
                     time_scale=cfg.model.time_scale,
                     rec_weight_mode=cfg.model.rec_weight_mode,
                     kl_weight_mode=cfg.model.kl_weight_mode,
                     lambda_denoiser=cfg.model.lambda_denoiser,
                     latent_type=cfg.model.latent_type,
                     latent_shape=list(cfg.model.latent_shape),
                     )

    elif cfg.model.name == 'covae_simple':
        kernel = get_kernel(cfg)
        return CoVAESimple(**cm_kwargs,
                           kernel=kernel,
                           p_mean=cfg.model.p_mean,
                           p_std=cfg.model.p_std,
                           sigma_data=cfg.model.sigma_data,
                           norm_strength=cfg.model.norm_strength,
                           mid_t=cfg.model.mid_t,
                           noise_schedule=cfg.model.noise_schedule,
                           norm_weight=cfg.model.norm_weight,
                     )
