import math
import time
from functools import partial

import torch
import torch.nn as nn
from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.loss import StandardDiffusionLoss
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (append_dims, default, disabled_train, get_obj_from_str,
                      instantiate_from_config)
from torch import nn
from torchdiffeq import odeint


class FlowEngine(nn.Module):

    def __init__(self, args, **kwargs):
        super().__init__()
        model_config = args.model_config
        log_keys = model_config.get('log_keys', None)
        input_key = model_config.get('input_key', 'mp4')
        network_config = model_config.get('network_config', None)
        network_wrapper = model_config.get('network_wrapper', None)
        denoiser_config = model_config.get('denoiser_config', None)
        sampler_config = model_config.get('sampler_config', None)
        conditioner_config = model_config.get('conditioner_config', None)
        first_stage_config = model_config.get('first_stage_config', None)
        loss_fn_config = model_config.get('loss_fn_config', None)
        scale_factor = model_config.get('scale_factor', 1.0)
        latent_input = model_config.get('latent_input', False)
        disable_first_stage_autocast = model_config.get(
            'disable_first_stage_autocast', False)
        no_cond_log = model_config.get('disable_first_stage_autocast', False)
        not_trainable_prefixes = model_config.get(
            'not_trainable_prefixes', ['first_stage_model', 'conditioner'])
        compile_model = model_config.get('compile_model', False)
        en_and_decode_n_samples_a_time = model_config.get(
            'en_and_decode_n_samples_a_time', None)
        lr_scale = model_config.get('lr_scale', None)
        lora_train = model_config.get('lora_train', False)
        self.use_pd = model_config.get('use_pd', False)

        self.log_keys = log_keys
        self.input_key = input_key
        self.not_trainable_prefixes = not_trainable_prefixes
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lr_scale = lr_scale
        self.lora_train = lora_train
        self.noised_image_input = model_config.get('noised_image_input', False)
        self.noised_image_all_concat = model_config.get(
            'noised_image_all_concat', False)
        self.noised_image_dropout = model_config.get('noised_image_dropout',
                                                     0.0)
        if args.fp16:
            dtype = torch.float16
            dtype_str = 'fp16'
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = 'bf16'
        else:
            dtype = torch.float32
            dtype_str = 'fp32'
        self.dtype = dtype
        self.dtype_str = dtype_str

        network_config['params']['dtype'] = dtype_str
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(
            default(network_wrapper,
                    OPENAIUNETWRAPPER))(model,
                                        compile_model=compile_model,
                                        dtype=dtype)

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(
            sampler_config) if sampler_config is not None else None
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG))

        self._init_first_stage(first_stage_config)

        self.loss_fn = instantiate_from_config(
            loss_fn_config) if loss_fn_config is not None else None

        self.latent_input = latent_input
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

    def disable_untrainable_params(self):
        pass

    def reinit(self, parent_model=None):
        pass

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        return batch[self.input_key].to(self.dtype)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast('cuda',
                            enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {
                        'timesteps': len(z[n * n_samples:(n + 1) * n_samples])
                    }
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples:(n + 1) * n_samples], **kwargs)
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, batch):
        frame = x.shape[2]

        if frame > 1 and self.latent_input:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            return x * self.scale_factor  # already encoded

        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast('cuda',
                            enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(x[n * n_samples:(n + 1) *
                                                      n_samples])
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def save_memory_encode_first_stage(self, x, batch):
        splits_x = torch.split(x, [13, 12, 12, 12], dim=2)

        all_out = []

        with torch.autocast('cuda', enabled=False):
            for idx, input_x in enumerate(splits_x):
                if idx == len(splits_x) - 1:
                    clear_fake_cp_cache = True
                else:
                    clear_fake_cp_cache = False
                out = self.first_stage_model.encode(
                    input_x.contiguous(),
                    clear_fake_cp_cache=clear_fake_cp_cache)
                all_out.append(out)

        z = torch.cat(all_out, dim=2)
        z = self.scale_factor * z
        return z

    def single_function_evaluation(self,
                                   t,
                                   x,
                                   cond=None,
                                   uc=None,
                                   cfg=1,
                                   **kwargs):
        start_time = time.time()
        # for CFG
        x = torch.cat([x] * 2)
        t = t.reshape(1).to(x.dtype).to(x.device)
        t = torch.cat([t] * 2)
        idx = 1000 - (t * 1000)

        real_cond = dict()
        for k, v in cond.items():
            uncond_v = uc[k]
            real_cond[k] = torch.cat([v, uncond_v])

        vt = self.model(x, t=idx, c=real_cond, idx=idx)
        vt, uc_vt = vt.chunk(2)
        vt = uc_vt + cfg * (vt - uc_vt)
        end_time = time.time()
        print(f'single_function_evaluation time at {t}', end_time - start_time)
        return vt

    @torch.no_grad()
    def sample(
        self,
        ref_x,
        cond,
        uc,
        **sample_kwargs,
    ):
        """Stage 2 Sampling, start from the first stage results `ref_x`

        Args:
            ref_x (_type_): Stage1 low resolution video
            cond (dict): Dict contains condtion embeddings
            uc (dict):  Dict contains  uncondition embedding

        Returns:
            Tensor: Secondary stage results
        """

        sample_kwargs = sample_kwargs or {}
        print('sample_kwargs', sample_kwargs)
        # timesteps
        num_steps = sample_kwargs.get('num_steps', 4)
        t = torch.linspace(0, 1, num_steps + 1,
                           dtype=ref_x.dtype).to(ref_x.device)
        print(self.share_cache['shift_t'])
        shift_t = float(self.share_cache['shift_t'])
        t = 1 - shift_t * (1 - t) / (1 + (shift_t - 1) * (1 - t))

        print('sample:', t)
        t = t
        single_function_evaluation = partial(self.single_function_evaluation,
                                             cond=cond,
                                             uc=uc,
                                             cfg=sample_kwargs.get('cfg', 1))

        ref_noise_step = self.share_cache['sample_ref_noise_step']
        print(f'ref_noise_step : {ref_noise_step}')

        ref_alphas_cumprod_sqrt = self.loss_fn.sigma_sampler.idx_to_sigma(
            torch.zeros(ref_x.shape[0]).fill_(ref_noise_step).long())
        ref_alphas_cumprod_sqrt = ref_alphas_cumprod_sqrt.to(ref_x.device)
        ori_dtype = ref_x.dtype

        ref_noise = torch.randn_like(ref_x)
        print('weight', ref_alphas_cumprod_sqrt, flush=True)

        ref_noised_input = ref_x * append_dims(ref_alphas_cumprod_sqrt, ref_x.ndim) \
                + ref_noise * append_dims(
        (1 - ref_alphas_cumprod_sqrt**2) ** 0.5, ref_x.ndim
        )
        ref_x = ref_noised_input.to(ori_dtype)
        self.share_cache['ref_x'] = ref_x

        results = odeint(single_function_evaluation,
                         ref_x,
                         t,
                         method=sample_kwargs.get('method', 'euler'),
                         atol=1e-6,
                         rtol=1e-3)[-1]

        return results


class FlowVideoDiffusionLoss(StandardDiffusionLoss):

    def __init__(self,
                 block_scale=None,
                 block_size=None,
                 min_snr_value=None,
                 fixed_frames=0,
                 **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        self.schedule = None
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        pass
