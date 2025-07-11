
# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import silu

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def pixel_norm(self, v):
        """
        Karras et al. pixel‐norm: normalize each feature‐vector at each (n, h, w)
        across the C channels.

        Args:
          v: tensor of shape (N, C, H, W) or (N, C)

        Returns:
          same shape as v, but normalized:
            v / sqrt( mean_c v^2 + eps )
        """
        # compute mean-of-squares over channel dim
        # keepdim so we can broadcast back
        mean_sq = v.pow(2).mean(dim=1, keepdim=True)
        return v / torch.sqrt(mean_sq + 1e-8)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        '''if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))'''
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            s_norm = self.pixel_norm(scale)
            b_norm = self.pixel_norm(shift)
            x_norm = self.norm1(x)
            x = silu(x_norm * s_norm + b_norm)
        else:
            x = silu(self.norm1(x.add_(params)))


        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            #w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class AutoEncoder(nn.Module):
    def __init__(self,
                 img_resolution,
                 in_channels,  # Number of color channels at input.
                 out_channels,
                 label_dim=0,
                 model_channels=128,
                 channel_mult_enc=[1, 2, 2, 2],
                 channel_mult_dec=[1, 2, 2, 2],
                 num_blocks_enc=4,
                 num_blocks_dec=4,
                 attn_resolutions=[16],
                 dropout=0.10,
                 z_channels=4,
                 final_dim=None,
                 ):
        super().__init__()

        init = dict(init_mode='xavier_uniform')

        emb_channels = model_channels * 4
        noise_channels = model_channels

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels,
                                             endpoint=True)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        self.temb_ch = model_channels * 4
        self.ch = model_channels

        self.encoder = Encoder(
            img_resolution=img_resolution,
            in_channels=in_channels,
            model_channels=model_channels,
            channel_mult=channel_mult_enc,
            num_blocks=num_blocks_enc,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            z_channels=z_channels,
            temb_ch=self.temb_ch,
            final_dim=final_dim,
        )

        self.decoder = Decoder(
            img_resolution=img_resolution,
            out_channels=out_channels,
            model_channels=model_channels,
            channel_mult=channel_mult_dec,
            num_blocks=num_blocks_dec,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            z_channels=z_channels,
            temb_ch=self.temb_ch,
            final_dim=final_dim,
        )

    def time_embedding(self, t, class_labels=None):
        assert t is not None
        emb = self.map_noise(t)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([t.shape[0], 1], device=t.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))
        return emb

class Encoder(nn.Module):
    def __init__(self,
                 img_resolution,
                 in_channels,
                 model_channels,
                 channel_mult,
                 num_blocks,
                 attn_resolutions,
                 dropout,
                 z_channels,
                 temb_ch,
                 final_dim=None,
                 ):
        super().__init__()

        self.z_channels = z_channels
        self.final_dim = final_dim
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=temb_ch, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=[1, 1], resample_proj=True, adaptive_scale=True,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True,
                                                          **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                **block_kwargs)


        self.mid_block_1 = UNetBlock(in_channels=cout, out_channels=cout, attention=True,
                                                                **block_kwargs)
        self.mid_block_2 = UNetBlock(in_channels=cout, out_channels=cout, attention=False, **block_kwargs)
        self.norm_out = GroupNorm(cout)
        self.conv_out = Conv2d(cout, 2 * z_channels, kernel=3, **init)
        self.quant_conv = Conv2d(2 * z_channels, 2 * z_channels, 1, **init)
        if isinstance(final_dim, int):
            assert final_dim == z_channels
            #final_res = int(img_resolution/(2**(len(channel_mult)-1)))
            #pre_dim = z_channels * 2 * final_res ** 2
            self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, temb):
        # Encoder.
        for name, block in self.enc.items():
            x = block(x, temb) if isinstance(block, UNetBlock) else block(x)

        x = self.mid_block_1(x, temb)
        x = self.mid_block_2(x, temb)
        x = self.norm_out(x)
        x = silu(x)
        x = self.conv_out(x)
        x = self.quant_conv(x)
        if isinstance(self.final_dim, int):
            # x = x.flatten(1)
            x = self.avg_pooling(x).view(x.shape[0], -1)
            mu = x[:, :self.final_dim]
            std = nn.functional.softplus(x[:, self.final_dim:])
        else:
            mu = x[:, :self.z_channels]
            std = nn.functional.softplus(x[:, self.z_channels:])
        return mu, std

class Decoder(nn.Module):
    def __init__(self,
                 img_resolution,
                 out_channels,
                 model_channels,
                 channel_mult,
                 num_blocks,
                 attn_resolutions,
                 dropout,
                 z_channels,
                 temb_ch,
                 final_dim=None,
                 ):
        super().__init__()

        self.z_channels = z_channels
        self.img_resolution = img_resolution
        self.final_dim = final_dim
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=temb_ch, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=[1, 1], resample_proj=True, adaptive_scale=True,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        cout = channel_mult[-1] * model_channels
        self.conv_in = Conv2d(z_channels, cout, kernel=3, **init)
        self.post_quant_conv = Conv2d(z_channels, z_channels, 1, **init)

        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True,
                                                         **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                **block_kwargs)
            if level == 0:
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3,
                                                           **init_zero)

        if isinstance(final_dim, int):
            self.final_res = int(img_resolution/(2**(len(channel_mult)-1)))
            pre_dim = z_channels * self.final_res ** 2
            self.final_dense = Linear(in_features=final_dim, out_features=pre_dim, **init)

    def forward(self, z, temb):
        if isinstance(self.final_dim, int):
            z = silu(self.final_dense(z))
            z = z.view(-1, self.z_channels, self.final_res, self.final_res)
        # timestep embedding
        z = self.post_quant_conv(z)
        # z to block_in
        x = self.conv_in(z)

        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                x = block(x, temb)
        return aux

    def get_last_layer(self):
        return self.dec[f'{self.img_resolution}x{self.img_resolution}_aux_conv'].weight