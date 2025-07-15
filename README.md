# Official Pytorch implementation for "CoVAE: Consistency Training of Variational Autoencoders" [https://arxiv.org/abs/2507.09103](https://arxiv.org/abs/2507.09103)

## Requirements
This code uses Weights & Biases, and assumes that you have your wandb key in a file named `wandb_config.py` in a variable named `key=your_wandb_key`.
This code uses the following libraries:
```angular2html
pytorch 
torchvision 
torchaudio 
pytorch-cuda
lightning
torchmetrics
torch-fidelity
scipy 
scikit-learn 
matplotlib 
wandb
hydra-core
```
## Training
The batch size is specified as batch per device, so adjust according to the number of GPUs you intend to use.
### CoVAE MNIST
```angular2html
python main.py log_samples=True compute_fid=True log_rec=True compute_rec_fid=True dataset=mnist dataset.name=mnist model=covae project=covae-mnist dataset.num_workers=16 dataset.batch_size=128 model.total_training_steps=400000 model.step_schedule=exp model.start_scales=2 model.end_scales=256 model.sigma_min=0.05 model.sigma_max=3 model.time_scale=karras model.rho=7 network=autoencoder gradient_clip_val=200 model.rec_weight_mode=linear model.kl_weight_mode=square network.attn_resolutions=[14] deterministic=True network.z_channels=1 network.model_channels=64 network.channel_mult_enc=[2,2,2] network.channel_mult_dec=[2,2,2] model.denoiser_loss_mode=l2 dataset.out_channels=2 model.loss_mode=huber model.lambda_denoiser=0.1 model.use_gan=False
```

### CoVAE CIFAR-10
```angular2html
python main.py dataset=cifar10 strategy=ddp_find_unused_parameters_true model=covae log_frequency=10000 project=covae-cifar dataset.num_workers=16 dataset.batch_size=512 model.total_training_steps=400000 compute_rec_fid=True model.step_schedule=exp model.start_scales=2 model.end_scales=256 model.sigma_min=0.05 model.sigma_max=3 model.time_scale=karras model.rho=7 network=autoencoder gradient_clip_val=200 model.rec_weight_mode=linear model.kl_weight_mode=square log_rec=True network.attn_resolutions=[16,8] deterministic=True network.z_channels=16 network.model_channels=128 network.channel_mult_enc=[2,2,4] network.channel_mult_dec=[2,2,4] network.num_blocks_enc=4 network.num_blocks_dec=4 model.denoiser_loss_mode=l2 dataset.out_channels=6 model.loss_mode=huber model.lambda_denoiser=0.1 model.use_gan=False
```

### CoVAE CIFAR-10 w/ L adv
```angular2html
python main.py dataset=cifar10 strategy=ddp_find_unused_parameters_true model=covae log_frequency=10000 project=covae-cifar dataset.num_workers=16 dataset.batch_size=512 model.total_training_steps=400000 compute_rec_fid=True model.step_schedule=exp model.start_scales=2 model.end_scales=256 model.sigma_min=0.05 model.sigma_max=3 model.time_scale=karras model.rho=7 network=autoencoder gradient_clip_val=200 model.rec_weight_mode=linear model.kl_weight_mode=square log_rec=True network.attn_resolutions=[16,8] deterministic=True network.z_channels=16 network.model_channels=128 network.channel_mult_enc=[2,2,4] network.channel_mult_dec=[2,2,4] network.num_blocks_enc=4 network.num_blocks_dec=4 model.denoiser_loss_mode=l2 dataset.out_channels=6 model.loss_mode=huber model.lambda_denoiser=0.1 model.use_gan=True model.gan_warmup_steps=200000 model.gan_lambda=0.05
```

### CoVAE CelebA-64 w/ L adv
```angular2html
python main.py dataset=celeba64 strategy=ddp_find_unused_parameters_true model=covae project=covae-celeba64 dataset.num_workers=16 dataset.batch_size=400 model.total_training_steps=400000 compute_rec_fid=True model.step_schedule=exp model.start_scales=2 model.end_scales=256 model.sigma_min=0.05 model.sigma_max=3 model.time_scale=karras model.rho=7 network=autoencoder gradient_clip_val=200 model.rec_weight_mode=linear model.kl_weight_mode=square log_rec=True network.attn_resolutions=[16,8] deterministic=True network.z_channels=64 network.model_channels=128 network.channel_mult_enc=[1,2,2,4] network.channel_mult_dec=[1,2,2,4] network.num_blocks_enc=2 network.num_blocks_dec=2 model.denoiser_loss_mode=l2 dataset.out_channels=6 model.loss_mode=huber model.lambda_denoiser=0.1 model.use_gan=True model.gan_warmup_steps=200000 model.gan_lambda=0.05
```

### CoVAE Binary MNIST
```angular2html
python main.py log_samples=True compute_fid=True log_rec=True compute_rec_fid=True dataset=mnist dataset.name=binary_mnist model=covae project=covae-mnist dataset.num_workers=16 dataset.batch_size=128 model.total_training_steps=400000 model.step_schedule=exp model.start_scales=2 model.end_scales=256 model.sigma_min=0.5 model.sigma_max=3 model.time_scale=karras model.rho=7 network=autoencoder gradient_clip_val=200 model.rec_weight_mode=linear model.kl_weight_mode=square network.attn_resolutions=[14] deterministic=True network.z_channels=1 network.model_channels=64 network.channel_mult_enc=[2,2,2] network.channel_mult_dec=[2,2,2] model.denoiser_loss_mode=bce dataset.out_channels=2 model.loss_mode=bce model.lambda_denoiser=0.1 model.use_gan=False
```

### s-CoVAE CIFAR-10 (with $\gamma$ regularization)
```angular2html
python main.py dataset=cifar10 model=covae model.name=covae_simple log_frequency=10000 model.start_scales=10 model.end_scales=1280 project=covae-cifar dataset.num_workers=18 dataset.batch_size=128 model.total_training_steps=400000 compute_rec_fid=True model.step_schedule=exp network=autoencoder log_rec=False network.attn_resolutions=[16,8] deterministic=False network.z_channels=16 network.model_channels=128 model.sigma_data=0.5 model.sigma_min=0.002 model.sigma_max=80 model.loss_mode=huber model.norm_strength=0.001 model.kernel=ve model.rho=7 model.noise_schedule=lognormal gradient_clip_val=200 model.norm_weight=fixed dataset.out_channels=6 model.use_gan=False model.denoiser_loss_mode=l2
```
### s-CoVAE CIFAR-10 (with normalization and tanh)
```angular2html
python main.py dataset=cifar10 model=covae model.name=covae_simple log_frequency=10000 model.start_scales=10 model.end_scales=1280 project=covae-cifar dataset.num_workers=18 dataset.batch_size=128 model.total_training_steps=400000 compute_rec_fid=True model.step_schedule=exp network=autoencoder log_rec=False network.attn_resolutions=[16,8] deterministic=False network.z_channels=16 network.model_channels=128 model.sigma_data=0.5 model.sigma_min=0.002 model.sigma_max=80 model.loss_mode=huber model.norm_strength=0 model.kernel=ve model.rho=7 model.noise_schedule=lognormal gradient_clip_val=200 model.norm_weight=fixed dataset.out_channels=6 model.use_gan=False model.denoiser_loss_mode=l2
```
## References
Parts of the code were adapted from the following codebases:
- [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm)
- [https://github.com/locuslab/ect](https://github.com/locuslab/ect)
- [https://github.com/sony/vct](https://github.com/sony/vct)

## Contact
- Gianluigi Silvestri: gianlu.silvestri@gmail.com