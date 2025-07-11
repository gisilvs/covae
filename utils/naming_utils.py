from omegaconf import OmegaConf, DictConfig


def get_run_name(cfg: DictConfig):
    name = f'{cfg.dataset.name}_{cfg.model.name}'
    if cfg.model.name == 'covae':
        name += f'_ss_{cfg.model.start_scales}_es_{cfg.model.end_scales}'
        name += f'_step_{cfg.model.step_schedule}'
        name += f'_smin_{cfg.model.sigma_min}_smax_{cfg.model.sigma_max}'
        if cfg.model.time_scale in ['karras']:
            name += f'_rho_{cfg.model.rho}'
        name += f'_ts_{cfg.model.time_scale}'
        name += f'_rwm_{cfg.model.rec_weight_mode}'
        name += f'_kwm_{cfg.model.kl_weight_mode}'
        name += f'_zch_{cfg.network.z_channels}'
        name += f'_bs_{cfg.dataset.batch_size * cfg.batch_multiplier}'
        name += f'_loss_mode_{cfg.model.loss_mode}'
        if cfg.model.denoiser_loss_mode:
            name += f'_denl'
            name += f'_ls_{cfg.model.lambda_denoiser}'
        if cfg.model.use_gan:
            name += f'_gan_{cfg.model.gan_lambda}'
    elif cfg.model.name == 'covae_simple':
            name += f'_ss_{cfg.model.start_scales}_es_{cfg.model.end_scales}'
            name += f'_step_{cfg.model.step_schedule}'
            name += f'_smin_{cfg.model.sigma_min}_smax_{cfg.model.sigma_max}'
            name += f'_rho_{cfg.model.rho}'
            name += f'_sd_{cfg.model.sigma_data}'
            name += f'_zch_{cfg.network.z_channels}'
            name += f'_bs_{cfg.dataset.batch_size * cfg.batch_multiplier}'
            name += f'_loss_mode_{cfg.model.loss_mode}'
            name += f'_ker_{cfg.model.kernel}'
            name += f'_ns_{cfg.model.noise_schedule}'
            if cfg.model.norm_strength > 0:
                name += f'_norm_{cfg.model.norm_strength}'
                name += f'_nw_{cfg.model.norm_weight}'
            if cfg.model.denoiser_loss_mode:
                name += f'_denl'
    return name
