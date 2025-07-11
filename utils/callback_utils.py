
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from custom_callbacks.diagnostic_callback import DiagnosticCallback
from custom_callbacks.fid_callback import FIDCallback
from custom_callbacks.generate_callback import GenerateCallback
from custom_callbacks.delete_checkpoints_callback import DeleteCheckpointsCallback

def get_delete_checkpoints_callback(cfg, path):
    return DeleteCheckpointsCallback(path, cfg.log_frequency)

def get_callbacks(cfg: DictConfig):
    callbacks = []
    rescale = 'binary' not in cfg.dataset.name
    #callbacks.append(ModelCheckpoint(save_last=True))  # saves at the end of the training?
    if cfg.log_samples:
        callbacks.append(
            GenerateCallback(tuple(cfg.dataset.sample_shape), n_iters=1, use_ema=True,
                             every_n_iterations=cfg.log_frequency, plot_type=cfg.dataset.plot_type, plot_rec=cfg.compute_rec_fid, rescale=rescale))
        callbacks.append(
            GenerateCallback(tuple(cfg.dataset.sample_shape), n_iters=2, use_ema=True,
                             every_n_iterations=cfg.log_frequency, plot_type=cfg.dataset.plot_type, rescale=rescale))
        callbacks.append(
            GenerateCallback(tuple(cfg.dataset.sample_shape), n_iters=1, use_ema=False,
                             every_n_iterations=cfg.log_frequency, plot_type=cfg.dataset.plot_type, rescale=rescale))
        callbacks.append(
            GenerateCallback(tuple(cfg.dataset.sample_shape), n_iters=2, use_ema=False,
                             every_n_iterations=cfg.log_frequency, plot_type=cfg.dataset.plot_type, rescale=rescale))

    if cfg.compute_fid:
        callbacks.append(
            FIDCallback(tuple(cfg.dataset.fid_sample_shape), n_iters=1, n_dataset_samples=cfg.dataset.n_dataset_samples,
                        every_n_iterations=cfg.log_frequency, compute_rec_fid=cfg.compute_rec_fid, rescale=rescale))
        callbacks.append(
            FIDCallback(tuple(cfg.dataset.fid_sample_shape), n_iters=2, n_dataset_samples=cfg.dataset.n_dataset_samples,
                        every_n_iterations=cfg.log_frequency, rescale=rescale))
        callbacks.append(ModelCheckpoint(every_n_train_steps=cfg.log_frequency,
                                         save_top_k=1,
                                         monitor="FID_1_iters",
                                         mode="min",
                                         save_last=True,
                                         save_on_train_epoch_end=False,
                                         enable_version_counter=False,
                                         ))

    if cfg.log_rec:
        callbacks.append(DiagnosticCallback(every_n_iterations=cfg.log_frequency))

    return callbacks
