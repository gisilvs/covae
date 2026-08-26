from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightning_modules.lightning_cm import LightningConsistencyModel
from utils.datamodule_utils import get_datamodule
from utils.utils import adjust_channels, rescaling, rescaling_inv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one-step post-training CoVAE diagnostics for one checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model.ckpt.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where diagnostics are saved.")
    parser.add_argument("--num-samples", type=int, default=10_000, help="Number of examples/samples to evaluate.")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers from checkpoint cfg.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override cfg.dataset.data_dir from checkpoint.")
    parser.add_argument("--skip-prepare-data", action="store_true", help="Do not call datamodule.prepare_data().")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--num-t", type=int, default=17, help="Number of linearly spaced positive time indices.")
    parser.add_argument(
        "--t-indices",
        type=str,
        default=None,
        help="Comma-separated positive time indices into model._get_time_steps(end_scales + 1), e.g. 1,2,4,8,257.",
    )
    parser.add_argument("--all-t", action="store_true", help="Evaluate every positive time index.")
    parser.add_argument("--prior-seed", type=int, default=1234, help="Seed for generation prior noise.")
    parser.add_argument("--posterior-seed", type=int, default=5678, help="Seed for posterior sampling noise.")
    parser.add_argument("--skip-fid", action="store_true", help="Skip FID computation and collect only KL/MSE/visual tensors.")
    parser.add_argument(
        "--num-visual-samples",
        type=int,
        default=16,
        help="Number of fixed real/generated/reconstructed images to store for later visualizations.",
    )
    parser.add_argument(
        "--no-save-per-sample",
        action="store_true",
        help="Only save summarized statistics, not per-sample KL/MSE arrays.",
    )
    return parser.parse_args()


def select_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device):
    lit_model = LightningConsistencyModel.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    cfg = lit_model.cfg
    model = lit_model.ema
    model.eval().requires_grad_(False).to(device)
    del lit_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cfg, model


def build_eval_loader(
    cfg,
    batch_size: int,
    num_workers: int | None,
    data_dir: Path | None,
    skip_prepare_data: bool,
) -> DataLoader:
    if num_workers is not None:
        cfg.dataset.num_workers = num_workers
    if data_dir is not None:
        cfg.dataset.data_dir = str(data_dir)
    datamodule = get_datamodule(cfg)
    if not skip_prepare_data:
        datamodule.prepare_data()
    datamodule.setup("fit")
    return DataLoader(
        datamodule.fid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def positive_t_indices(model, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    time_steps = model._get_time_steps(model.end_scales + 1, device=device)
    max_idx = time_steps.numel() - 1
    if args.all_t:
        indices = torch.arange(1, max_idx + 1, dtype=torch.long)
    elif args.t_indices is not None:
        indices = torch.tensor([int(value) for value in args.t_indices.split(",")], dtype=torch.long)
    else:
        indices = torch.linspace(1, max_idx, args.num_t).round().to(torch.long).unique(sorted=True)

    if indices.numel() == 0:
        raise ValueError("No time indices selected.")
    if indices.min() < 1 or indices.max() > max_idx:
        raise ValueError(f"Time indices must be in [1, {max_idx}], got {indices.tolist()}.")
    return indices


def split_batch(batch) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def limited_eval_batches(loader: DataLoader, num_samples: int):
    seen = 0
    for batch in loader:
        images = split_batch(batch)
        remaining = num_samples - seen
        if remaining <= 0:
            break
        if images.shape[0] > remaining:
            images = images[:remaining]
        seen += images.shape[0]
        yield images
    if seen < num_samples:
        raise ValueError(f"Requested {num_samples} samples, but dataloader only yielded {seen}.")


def uses_rescaling(cfg) -> bool:
    return "binary" not in cfg.dataset.name


def model_input_from_real(real_images: torch.Tensor, rescale: bool) -> torch.Tensor:
    if rescale:
        return rescaling(real_images)
    return real_images


def model_output_to_image(model_output: torch.Tensor, rescale: bool) -> torch.Tensor:
    if rescale:
        images = rescaling_inv(model_output.clamp(-1, 1))
    else:
        images = torch.sigmoid(model_output)
    return adjust_channels(images)


def sample_model_noise(model, batch_size: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    if model.latent_type == "gaussian":
        shape = [batch_size] + list(model.noise_shape)
        return torch.randn(shape, dtype=torch.float32, generator=generator).to(device)
    if model.latent_type == "categorical":
        shape = [batch_size] + list(model.latent_shape)
        return torch.rand(shape, dtype=torch.float32, generator=generator).to(device)
    raise NotImplementedError(f"Unsupported latent_type={model.latent_type}")


def prior_decode(model, t_value: torch.Tensor, batch_size: int, device: torch.device, generator: torch.Generator):
    noise = sample_model_noise(model, batch_size, device, generator)
    mu = torch.zeros([batch_size] + list(model.noise_shape), dtype=torch.float32, device=device)
    std = torch.ones_like(mu)
    z = model._reparametrized_sample(mu, std, noise)
    t = torch.full((batch_size,), float(t_value.item()), dtype=torch.float32, device=device)
    decoded, _ = model.decode(z, t, class_labels=None)
    return decoded


def posterior_decode(model, x: torch.Tensor, t_value: torch.Tensor, noise: torch.Tensor):
    batch_size = x.shape[0]
    t = torch.full((batch_size,), float(t_value.item()), dtype=torch.float32, device=x.device)
    decoded, mu, std, _ = model.precond(x, t, noise, class_labels=None)
    return decoded, mu, std


def kl_to_standard_prior(model, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    posterior = model._get_distribution(mu, std + 1e-8)
    prior = model._get_distribution(torch.zeros_like(mu), torch.ones_like(std))
    return torch.distributions.kl_divergence(posterior, prior).reshape(mu.shape[0], -1).sum(1)


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.to(torch.float64)
    std = values.std(unbiased=False)
    return {
        "mean": values.mean().item(),
        "std": std.item(),
        "sem": (std / math.sqrt(values.numel())).item(),
    }


@torch.no_grad()
def initialize_fid_real_features(
    loader: DataLoader,
    num_samples: int,
    device: torch.device,
) -> FrechetInceptionDistance:
    fid = FrechetInceptionDistance(reset_real_features=False, normalize=True).to(device)
    for real_images in limited_eval_batches(loader, num_samples):
        real_images = adjust_channels(real_images.to(device, non_blocking=True))
        fid.update(real_images, real=True)
    return fid


@torch.no_grad()
def collect_generation_fid(
    model,
    fid: FrechetInceptionDistance | None,
    t_value: torch.Tensor,
    cfg,
    args: argparse.Namespace,
    device: torch.device,
):
    rescale = uses_rescaling(cfg)
    generator = torch.Generator(device="cpu").manual_seed(args.prior_seed)
    if fid is not None:
        fid.reset()
    total = 0
    visual_images = []
    visual_count = 0
    while total < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - total)
        decoded = prior_decode(model, t_value, batch_size, device, generator)
        images = model_output_to_image(decoded, rescale)
        if fid is not None:
            fid.update(images, real=False)
        if visual_count < args.num_visual_samples:
            take = min(args.num_visual_samples - visual_count, images.shape[0])
            visual_images.append(images[:take].cpu())
            visual_count += take
        total += batch_size
    fid_value = float("nan") if fid is None else fid.compute().item()
    visuals = torch.cat(visual_images, dim=0) if visual_images else torch.empty(0)
    return fid_value, visuals


@torch.no_grad()
def collect_reconstruction_diagnostics(
    model,
    fid: FrechetInceptionDistance | None,
    loader: DataLoader,
    t_value: torch.Tensor,
    prev_t_value: torch.Tensor | None,
    cfg,
    args: argparse.Namespace,
    device: torch.device,
):
    rescale = uses_rescaling(cfg)
    generator = torch.Generator(device="cpu").manual_seed(args.posterior_seed)
    if fid is not None:
        fid.reset()
    kl_values = []
    recon_mse_values = []
    recon_mse_image_values = []
    consistency_mse_values = []
    visual_real = []
    visual_recon = []
    visual_count = 0

    for real_images in limited_eval_batches(loader, args.num_samples):
        real_images = real_images.to(device, non_blocking=True)
        x = model_input_from_real(real_images, rescale)
        noise = sample_model_noise(model, x.shape[0], device, generator)
        decoded, mu, std = posterior_decode(model, x, t_value, noise)
        decoded_images = model_output_to_image(decoded, rescale)
        real_fid_images = adjust_channels(real_images)
        if fid is not None:
            fid.update(decoded_images, real=False)

        kl_values.append(kl_to_standard_prior(model, mu, std).cpu())
        recon_mse_values.append(((decoded - x) ** 2).reshape(x.shape[0], -1).mean(1).cpu())
        recon_mse_image_values.append(
            ((decoded_images - real_fid_images) ** 2).reshape(x.shape[0], -1).mean(1).cpu()
        )

        if prev_t_value is None:
            prev_decoded = x
        else:
            prev_decoded, _, _ = posterior_decode(model, x, prev_t_value, noise)
        consistency_mse_values.append(((decoded - prev_decoded) ** 2).reshape(x.shape[0], -1).mean(1).cpu())

        if visual_count < args.num_visual_samples:
            take = min(args.num_visual_samples - visual_count, decoded_images.shape[0])
            visual_real.append(real_fid_images[:take].cpu())
            visual_recon.append(decoded_images[:take].cpu())
            visual_count += take

    fid_value = float("nan") if fid is None else fid.compute().item()
    per_sample = {
        "kl": torch.cat(kl_values, dim=0),
        "reconstruction_mse": torch.cat(recon_mse_values, dim=0),
        "reconstruction_mse_image": torch.cat(recon_mse_image_values, dim=0),
        "consistency_mse": torch.cat(consistency_mse_values, dim=0),
    }
    visuals = {
        "real": torch.cat(visual_real, dim=0) if visual_real else torch.empty(0),
        "reconstruction": torch.cat(visual_recon, dim=0) if visual_recon else torch.empty(0),
    }
    return fid_value, per_sample, visuals


def write_metrics_csv(output_dir: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg, model = load_model(args.checkpoint, device)
    eval_loader = build_eval_loader(
        cfg=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        skip_prepare_data=args.skip_prepare_data,
    )
    t_indices = positive_t_indices(model, args, device)
    time_steps = model._get_time_steps(model.end_scales + 1, device=device)

    print(f"Loaded EMA model from {args.checkpoint}")
    print(f"Dataset: {cfg.dataset.name}; samples: {args.num_samples}; device: {device}")
    print(f"Evaluating time indices: {t_indices.tolist()}")

    fid = None if args.skip_fid else initialize_fid_real_features(eval_loader, args.num_samples, device)

    generation_fids = []
    reconstruction_fids = []
    rows = []
    per_sample_by_key = {
        "kl": [],
        "reconstruction_mse": [],
        "reconstruction_mse_image": [],
        "consistency_mse": [],
    }
    visual_generation = []
    visual_reconstruction = []
    visual_real = None

    for position, t_idx in enumerate(t_indices.tolist()):
        t_value = time_steps[t_idx]
        prev_t_value = None if t_idx == 1 else time_steps[t_idx - 1]
        print(f"[{position + 1}/{len(t_indices)}] t_idx={t_idx}, t={t_value.item():.6g}")

        gen_fid, gen_visuals = collect_generation_fid(model, fid, t_value, cfg, args, device)
        rec_fid, per_sample, rec_visuals = collect_reconstruction_diagnostics(
            model=model,
            fid=fid,
            loader=eval_loader,
            t_value=t_value,
            prev_t_value=prev_t_value,
            cfg=cfg,
            args=args,
            device=device,
        )

        generation_fids.append(gen_fid)
        reconstruction_fids.append(rec_fid)
        visual_generation.append(gen_visuals)
        visual_reconstruction.append(rec_visuals["reconstruction"])
        if visual_real is None:
            visual_real = rec_visuals["real"]

        stats = {key: summarize(value) for key, value in per_sample.items()}
        for key, value in per_sample.items():
            per_sample_by_key[key].append(value)

        ratio = stats["consistency_mse"]["mean"] / max(stats["reconstruction_mse"]["mean"], 1e-12)
        rows.append(
            {
                "position": position,
                "t_index": t_idx,
                "t_value": t_value.item(),
                "generation_fid": gen_fid,
                "reconstruction_fid": rec_fid,
                "kl_mean": stats["kl"]["mean"],
                "kl_std": stats["kl"]["std"],
                "kl_sem": stats["kl"]["sem"],
                "reconstruction_mse_mean": stats["reconstruction_mse"]["mean"],
                "reconstruction_mse_std": stats["reconstruction_mse"]["std"],
                "reconstruction_mse_sem": stats["reconstruction_mse"]["sem"],
                "reconstruction_mse_image_mean": stats["reconstruction_mse_image"]["mean"],
                "reconstruction_mse_image_std": stats["reconstruction_mse_image"]["std"],
                "reconstruction_mse_image_sem": stats["reconstruction_mse_image"]["sem"],
                "consistency_mse_mean": stats["consistency_mse"]["mean"],
                "consistency_mse_std": stats["consistency_mse"]["std"],
                "consistency_mse_sem": stats["consistency_mse"]["sem"],
                "consistency_to_reconstruction_ratio": ratio,
            }
        )

    metrics = {
        "generation_fid": torch.tensor(generation_fids, dtype=torch.float32),
        "reconstruction_fid": torch.tensor(reconstruction_fids, dtype=torch.float32),
    }
    for key, values in per_sample_by_key.items():
        stacked = torch.stack(values, dim=0)
        metrics[f"{key}_mean"] = stacked.mean(dim=1)
        metrics[f"{key}_std"] = stacked.std(dim=1, unbiased=False)
        metrics[f"{key}_sem"] = metrics[f"{key}_std"] / math.sqrt(stacked.shape[1])

    payload = {
        "metadata": {
            "checkpoint": str(args.checkpoint),
            "weights": "ema",
            "sampling_steps": 1,
            "generation_definition": "sample z from the prior and decode once at t",
            "reconstruction_definition": "encode x at t, sample z from q_t(z|x), and decode once at t",
            "consistency_definition": "compare one-step reconstructions at consecutive time indices using the same posterior noise",
            "dataset": cfg.dataset.name,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "device": str(device),
            "prior_seed": args.prior_seed,
            "posterior_seed": args.posterior_seed,
            "skip_fid": args.skip_fid,
            "use_consistency_loss": bool(getattr(model, "use_consistency_loss", True)),
            "latent_type": model.latent_type,
            "noise_shape": list(model.noise_shape),
        },
        "t_indices": t_indices.cpu(),
        "t_values": time_steps[t_indices].detach().cpu(),
        "metrics": metrics,
        "visuals": {
            "real": visual_real if visual_real is not None else torch.empty(0),
            "generation": torch.stack(visual_generation, dim=0) if visual_generation else torch.empty(0),
            "reconstruction": torch.stack(visual_reconstruction, dim=0) if visual_reconstruction else torch.empty(0),
        },
    }
    if not args.no_save_per_sample:
        payload["per_sample"] = {key: torch.stack(values, dim=0) for key, values in per_sample_by_key.items()}

    torch.save(payload, args.output_dir / "diagnostics.pt")
    write_metrics_csv(args.output_dir, rows)
    (args.output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    (args.output_dir / "checkpoint_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    print(f"Saved diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
