from omegaconf import DictConfig

from datamodules.cifar10_datamodule import CIFAR10DataModule
from datamodules.mnist_datamodule import MNISTDataModule
from datamodules.celeba64_datamodule import CelebA64DataModule

def get_datamodule(cfg: DictConfig):

    if cfg.dataset.name in ['mnist', 'binary_mnist']:
        binary =  'binary' in cfg.dataset.name
        dm = MNISTDataModule(batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers,
                                    data_dir=cfg.dataset.data_dir, binary=binary)
    elif cfg.dataset.name == 'cifar10':
        dm = CIFAR10DataModule(batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers, data_dir=cfg.dataset.data_dir)

    elif cfg.dataset.name == 'celeba64':
        dm = CelebA64DataModule(batch_size=cfg.dataset.batch_size, size=cfg.dataset.size, num_workers=cfg.dataset.num_workers, data_dir=cfg.dataset.data_dir)
    else:
        raise NotImplementedError

    return dm