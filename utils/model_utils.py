import hydra
from omegaconf import DictConfig

from models.factories import validate_model_config


def get_model(cfg: DictConfig, pretrained_net=None):
    """Compatibility wrapper around the Hydra model config."""
    validate_model_config(cfg)

    kwargs = {}
    if pretrained_net is not None:
        kwargs["model"] = pretrained_net

    return hydra.utils.instantiate(cfg.model.instance, _convert_="all", **kwargs)
