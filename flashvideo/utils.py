import math
from typing import List, Union

import numpy as np
import torch
from omegaconf import ListConfig
from sgm.util import instantiate_from_config


def read_from_file(p, rank=0, world_size=1):
    with open(p) as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def disable_all_init():
    """Disable all redundant torch default initialization to accelerate model
    creation."""
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    setattr(torch.nn.modules.sparse.Embedding, 'reset_parameters',
            lambda self: None)
    setattr(torch.nn.modules.conv.Conv2d, 'reset_parameters',
            lambda self: None)
    setattr(torch.nn.modules.normalization.GroupNorm, 'reset_parameters',
            lambda self: None)


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list({x.input_key for x in conditioner.embedders})


def get_batch(keys,
              value_dict,
              N: Union[List, ListConfig],
              T=None,
              device='cuda'):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == 'txt':
            batch['txt'] = np.repeat([value_dict['prompt']],
                                     repeats=math.prod(N)).reshape(N).tolist()
            batch_uc['txt'] = np.repeat(
                [value_dict['negative_prompt']],
                repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch['num_video_frames'] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def decode(first_stage_model, latent):
    first_stage_model.to(torch.float16)
    latent = latent.to(torch.float16)
    recons = []
    T = latent.shape[2]
    if T > 2:
        loop_num = (T - 1) // 2
        for i in range(loop_num):
            if i == 0:
                start_frame, end_frame = 0, 3
            else:
                start_frame, end_frame = i * 2 + 1, i * 2 + 3
            if i == loop_num - 1:
                clear_fake_cp_cache = True
            else:
                clear_fake_cp_cache = False
            with torch.no_grad():
                recon = first_stage_model.decode(
                    latent[:, :, start_frame:end_frame].contiguous(),
                    clear_fake_cp_cache=clear_fake_cp_cache)

            recons.append(recon)
    else:

        clear_fake_cp_cache = True
        if latent.shape[2] > 1:
            for m in first_stage_model.modules():
                m.force_split = True
        recon = first_stage_model.decode(
            latent.contiguous(), clear_fake_cp_cache=clear_fake_cp_cache)
        recons.append(recon)
    recon = torch.cat(recons, dim=2).to(torch.float32)
    samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    samples = (samples * 255).squeeze(0).permute(0, 2, 3, 1)
    save_frames = samples

    return save_frames


def save_mem_decode(first_stage_model, latent):

    l_h, l_w = latent.shape[3], latent.shape[4]
    T = latent.shape[2]
    F = 8
    # split spatial along h w
    num_h_splits = 1
    num_w_splits = 2
    ori_video = torch.zeros((1, 3, 1 + 4 * (T - 1), l_h * 8, l_w * 8),
                            device=latent.device)
    for h_idx in range(num_h_splits):
        for w_idx in range(num_w_splits):
            start_h = h_idx * latent.shape[3] // num_h_splits
            end_h = (h_idx + 1) * latent.shape[3] // num_h_splits
            start_w = w_idx * latent.shape[4] // num_w_splits
            end_w = (w_idx + 1) * latent.shape[4] // num_w_splits

            latent_overlap = 16
            if (start_h - latent_overlap >= 0) and (num_h_splits > 1):
                real_start_h = start_h - latent_overlap
                h_start_overlap = latent_overlap * F
            else:
                h_start_overlap = 0
                real_start_h = start_h
            if (end_h + latent_overlap <= l_h) and (num_h_splits > 1):
                real_end_h = end_h + latent_overlap
                h_end_overlap = latent_overlap * F
            else:
                h_end_overlap = 0
                real_end_h = end_h

            if (start_w - latent_overlap >= 0) and (num_w_splits > 1):
                real_start_w = start_w - latent_overlap
                w_start_overlap = latent_overlap * F
            else:
                w_start_overlap = 0
                real_start_w = start_w

            if (end_w + latent_overlap <= l_w) and (num_w_splits > 1):
                real_end_w = end_w + latent_overlap
                w_end_overlap = latent_overlap * F
            else:
                w_end_overlap = 0
                real_end_w = end_w

            latent_slice = latent[:, :, :, real_start_h:real_end_h,
                                  real_start_w:real_end_w]
            recon = decode(first_stage_model, latent_slice)

            recon = recon.permute(3, 0, 1, 2).contiguous()[None]

            recon = recon[:, :, :,
                          h_start_overlap:recon.shape[3] - h_end_overlap,
                          w_start_overlap:recon.shape[4] - w_end_overlap]
            ori_video[:, :, :, start_h * 8:end_h * 8,
                      start_w * 8:end_w * 8] = recon
    ori_video = ori_video.squeeze(0)
    ori_video = ori_video.permute(1, 2, 3, 0).contiguous().cpu()
    return ori_video


def prepare_input(text, model, T, negative_prompt=None, pos_prompt=None):

    if negative_prompt is None:
        negative_prompt = ''
    if pos_prompt is None:
        pos_prompt = ''
    value_dict = {
        'prompt': text + pos_prompt,
        'negative_prompt': negative_prompt,
        'num_frames': torch.tensor(T).unsqueeze(0),
    }
    print(value_dict)
    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        value_dict, [1])

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            print(key, batch[key].shape)
        elif isinstance(batch[key], list):
            print(key, [len(l) for l in batch[key]])
        else:
            print(key, batch[key])
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=['txt'],
    )

    for k in c:
        if not k == 'crossattn':
            c[k], uc[k] = map(lambda y: y[k][:math.prod([1])].to('cuda'),
                              (c, uc))
    return c, uc


def save_memory_encode_first_stage(x, model):
    splits_x = torch.split(x, [17, 16, 16], dim=2)
    all_out = []

    with torch.autocast('cuda', enabled=False):
        for idx, input_x in enumerate(splits_x):
            if idx == len(splits_x) - 1:
                clear_fake_cp_cache = True
            else:
                clear_fake_cp_cache = False
            out = model.first_stage_model.encode(
                input_x.contiguous(), clear_fake_cp_cache=clear_fake_cp_cache)
            all_out.append(out)

    z = torch.cat(all_out, dim=2)
    z = model.scale_factor * z
    return z


def seed_everything(seed: int = 42):
    import os
    import random

    import numpy as np
    import torch

    # Python random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # If using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # # CuDNN
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # OS environment
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_time_slice_vae():
    vae_config = {
        'target': 'vae_modules.autoencoder.VideoAutoencoderInferenceWrapper',
        'params': {
            'cp_size': 1,
            'ckpt_path': './checkpoints/3d-vae.pt',
            'ignore_keys': ['loss'],
            'loss_config': {
                'target': 'torch.nn.Identity'
            },
            'regularizer_config': {
                'target':
                'vae_modules.regularizers.DiagonalGaussianRegularizer'
            },
            'encoder_config': {
                'target':
                'vae_modules.cp_enc_dec.SlidingContextParallelEncoder3D',
                'params': {
                    'double_z': True,
                    'z_channels': 16,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 2, 2, 4],
                    'attn_resolutions': [],
                    'num_res_blocks': 3,
                    'dropout': 0.0,
                    'gather_norm': False
                }
            },
            'decoder_config': {
                'target': 'vae_modules.cp_enc_dec.ContextParallelDecoder3D',
                'params': {
                    'double_z': True,
                    'z_channels': 16,
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 2, 2, 4],
                    'attn_resolutions': [],
                    'num_res_blocks': 3,
                    'dropout': 0.0,
                    'gather_norm': False
                }
            }
        }
    }

    vae = instantiate_from_config(vae_config).eval().half().cuda()
    return vae
