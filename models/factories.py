from kernels.linear_interpolant import LinearInterpolantKernel
from kernels.variance_exploding import VarianceExplodingKernel
from networks.discriminator import NLayerDiscriminator, weights_init


MODEL_TARGETS = {
    "covae": "models.covae.CoVAE",
    "covae_simple": "models.covae_simple.CoVAESimple",
}


def _optional_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in {"none", "null", ""}:
            return None
        return int(value)
    return None


def build_noise_shape(img_resolution, channel_mult_enc, z_channels, final_dim=None):
    final_dim = _optional_int(final_dim)
    if final_dim is not None:
        return [final_dim]

    latent_size = int(img_resolution) // (2 ** (len(channel_mult_enc) - 1))
    return [int(z_channels), latent_size, latent_size]


def build_discriminator(enabled, input_nc, n_layers=3, use_actnorm=False):
    if not enabled:
        return None

    return NLayerDiscriminator(
        input_nc=int(input_nc),
        n_layers=int(n_layers),
        use_actnorm=use_actnorm,
    ).apply(weights_init)


def build_kernel(name, sigma_min, sigma_max, sigma_data):
    if name == "ve":
        return VarianceExplodingKernel(sigma_min, sigma_max, sigma_data)
    if name == "li":
        return LinearInterpolantKernel(sigma_min, sigma_max, sigma_data)

    raise ValueError(f"Unknown kernel: {name}")


def validate_model_config(cfg):
    if "instance" not in cfg.model:
        raise ValueError(
            "cfg.model.instance is missing. Select one of the Hydra model configs "
            "(for example `model=covae` or `model=covae_simple`)."
        )

    expected_target = MODEL_TARGETS.get(cfg.model.name)
    actual_target = cfg.model.instance.get("_target_")
    if expected_target is None:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    if actual_target != expected_target:
        raise ValueError(
            f"Model config mismatch: model.name={cfg.model.name!r} expects "
            f"{expected_target!r}, but cfg.model.instance._target_ is "
            f"{actual_target!r}. Select the matching Hydra config group, e.g. "
            "`model=covae_simple` instead of overriding `model.name`."
        )
