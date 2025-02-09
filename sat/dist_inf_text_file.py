import argparse
import copy
import os

import numpy as np
import torch
from arguments import get_args
from diffusion_video import SATVideoDiffusionEngine
from flow_video import FlowEngine
from torchvision.io.video import write_video
from utils import (decode, disable_all_init, prepare_input, save_mem_decode,
                   save_memory_encode_first_stage, seed_everything)

from sat import mpu
from sat.model.base_model import get_model

disable_all_init()


def init_model(model, second_model, args, second_args):

    share_cache = dict()
    second_share_cache = dict()
    if hasattr(args, 'share_cache_config'):
        for k, v in args.share_cache_config.items():
            share_cache[k] = v
    if hasattr(second_args, 'share_cache_config'):
        for k, v in second_args.share_cache_config.items():
            second_share_cache[k] = v

    for n, m in model.named_modules():
        m.share_cache = share_cache
        if hasattr(m, 'register_new_modules'):
            m.register_new_modules()
    for n, m in second_model.named_modules():
        m.share_cache = second_share_cache
        if hasattr(m, 'register_new_modules'):
            m.register_new_modules()

    if os.environ.get('SKIP_LOAD', None) is not None:
        print('skip load for speed debug')
    else:
        weight_path = args.inf_ckpt
        weight = torch.load(weight_path, map_location='cpu')
        if 'model.diffusion_model.mixins.pos_embed.freqs_sin' in weight[
                'module']:
            del weight['module'][
                'model.diffusion_model.mixins.pos_embed.freqs_sin']
            del weight['module'][
                'model.diffusion_model.mixins.pos_embed.freqs_cos']
        msg = model.load_state_dict(weight['module'], strict=False)
        print(msg)
        second_weight_path = args.inf_ckpt2
        second_weight = torch.load(second_weight_path, map_location='cpu')

        if 'model.diffusion_model.mixins.pos_embed.freqs_sin' in second_weight[
                'module']:
            del second_weight['module'][
                'model.diffusion_model.mixins.pos_embed.freqs_sin']
            del second_weight['module'][
                'model.diffusion_model.mixins.pos_embed.freqs_cos']
        second_msg = second_model.load_state_dict(second_weight['module'],
                                                  strict=False)
        print(second_msg)
    for n, m in model.named_modules():
        if hasattr(m, 'merge_lora'):
            m.merge_lora()
            print(f'merge lora of {n}')


def get_first_results(model, text, num_frames, H, W):
    """Get first Stage results.

    Args:
        model (nn.Module): first stage model.
        text (str): text prompt
        num_frames (int): number of frames
        H (int): height of the first stage results
        W (int): width of the first stage results

    Returns:
        Tensor: first stage video.
    """
    device = 'cuda'
    T = 1 + (num_frames - 1) // 4
    F = 8
    motion_text_prefix = [
        'very low motion,',
        'low motion,',
        'medium motion,',
        'high motion,',
        'very high motion,',
    ]
    neg_prompt = ''
    pos_prompt = ''
    with torch.no_grad():
        model.to('cuda')
        input_negative_prompt = motion_text_prefix[
            0] + ', ' + motion_text_prefix[1] + neg_prompt
        c, uc = prepare_input(text,
                              model,
                              T,
                              negative_prompt=input_negative_prompt,
                              pos_prompt=pos_prompt)
        with torch.no_grad(), torch.amp.autocast(enabled=True,
                                                 device_type='cuda',
                                                 dtype=torch.bfloat16):
            samples_z = model.sample(
                c,
                uc=uc,
                batch_size=1,
                shape=(T, 16, H // F, W // F),
                num_steps=model.share_cache.get('first_sample_step', None),
            )
        samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

        model.to('cpu')
        torch.cuda.empty_cache()
        first_stage_model = model.first_stage_model
        first_stage_model = first_stage_model.to(device)

        latent = 1.0 / model.scale_factor * samples_z

        samples = decode(first_stage_model, latent)
    model.to('cpu')
    return samples


def get_second_results(model, text, first_stage_samples, num_frames):
    """Get second Stage results.

    Args:
        model (nn.Module): second stage model.
        text (str): text prompt
        first_stage_samples (Tensor): first stage results
        num_frames (int): number of frames
    Returns:
        Tensor: second stage results.
    """

    t, h, w, c = first_stage_samples.shape
    first_stage_samples = first_stage_samples[:num_frames]
    first_stage_samples = (first_stage_samples / 255.)
    first_stage_samples = (first_stage_samples - 0.5) / 0.5

    target_size = model.share_cache.get('target_size', None)
    if target_size is None:
        upscale_factor = model.share_cache.get('upscale_factor', 8)
        H = int(h * upscale_factor) // 16 * 16
        W = int(w * upscale_factor) // 16 * 16
    else:
        H, W = target_size
        H = H // 16 * 16
        W = W // 16 * 16

    first_stage_samples = first_stage_samples.permute(0, 3, 1, 2).to('cuda')

    ref_x = torch.nn.functional.interpolate(first_stage_samples,
                                            size=(H, W),
                                            mode='bilinear',
                                            align_corners=False,
                                            antialias=True)
    ref_x = ref_x[:num_frames][None]

    ref_x = ref_x.permute(0, 2, 1, 3, 4).contiguous()

    first_stage_model = model.first_stage_model
    print(f'start encoding first stage results to high resolution')
    with torch.no_grad():
        first_stage_dtype = next(model.first_stage_model.parameters()).dtype
        model.first_stage_model.cuda()
        ref_x = save_memory_encode_first_stage(
            ref_x.contiguous().to(first_stage_dtype).cuda(), model)

    ref_x = ref_x.permute(0, 2, 1, 3, 4).contiguous()
    ref_x = ref_x.to(model.dtype)
    print(f'finish encoding first stage results, and starting stage II')

    device = 'cuda'

    model.to(device)
    motion_text_prefix = [
        'very low motion,',
        'low motion,',
        'medium motion,',
        'high motion,',
        'very high motion,',
    ]

    pos_prompt = None
    input_negative_prompt = None
    text = 'medium motion,' + text
    c, uc = prepare_input(text,
                          model,
                          num_frames,
                          negative_prompt=input_negative_prompt,
                          pos_prompt=pos_prompt)

    T = 1 + (num_frames - 1) // 4
    F = 8
    with torch.no_grad(), torch.amp.autocast(enabled=True,
                                             device_type='cuda',
                                             dtype=torch.bfloat16):
        samples_z = model.sample(
            ref_x,
            c,
            uc=uc,
            batch_size=1,
            shape=(T, 16, H // F, W // F),
            num_steps=model.share_cache.get('sample_step', 5),
            method='euler',
            cfg=model.share_cache.get('cfg', 7.5),
        )
        samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

        model.to('cpu')
        torch.cuda.empty_cache()
        first_stage_model = model.first_stage_model
        first_stage_model = first_stage_model.to(device)

        latent = 1.0 / model.scale_factor * samples_z
        print(f'start spatiotemporal slice decoding')
        samples = save_mem_decode(first_stage_model, latent)
        print(f'finish spatiotemporal slice decoding')
        model.to('cpu')
    return samples


def sampling_main(args, second_args, model_cls, second_model_cls):

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    second_model = get_model(second_args, second_model_cls)

    model = get_model(args, model_cls)

    init_model(model, second_model, args, second_args)

    model.eval()
    second_model.eval()
    rank, world_size = mpu.get_data_parallel_rank(
    ), mpu.get_data_parallel_world_size()

    rank, world_size = mpu.get_data_parallel_rank(
    ), mpu.get_data_parallel_world_size()
    print('rank and world_size', rank, world_size)

    text_file = args.input_file

    num_sample_perprompt = 1

    with open(text_file) as fin:
        all_prompt = []
        for single_line in fin:
            all_prompt.append(single_line.strip())
    print(f'load from {text_file} with {len(all_prompt)}')

    image_size = [270, 480]

    image_size = (image_size[0] // 16 * 16, image_size[1] // 16 * 16)
    second_img_size = [1080, 1920]

    second_img_size = (second_img_size[0] // 16 * 16,
                       second_img_size[1] // 16 * 16)

    num_frames = 49
    second_num_frames = 49

    # 6-8
    model.share_cache['cfg'] = 8

    second_model.share_cache['target_size'] = second_img_size

    # range from 650 to 750
    second_model.share_cache['ref_noise_step'] = 675
    second_model.share_cache['sample_ref_noise_step'] = 675

    # range from 2 to 3.5
    second_model.share_cache['shift_t'] = 2.5

    # range from 4 to 6
    second_model.share_cache['sample_step'] = 5
    # range from 10 to 13
    second_model.share_cache['cfg'] = 13

    second_model.share_cache.pop('ref_noise_step_range', None)

    second_model.share_cache['time_size_embedding'] = True
    _, H, W, _, _ = num_frames, image_size[0], image_size[
        1], args.latent_channels, 8

    save_dir = f''''''
    if args.output_dir:
        save_dir = args.output_dir
        print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    all_ids = range(len(all_prompt))
    local_ids = all_ids[rank::world_size]

    for enu_index in local_ids:

        text = all_prompt[enu_index]
        print(f'rank {rank} processing {enu_index}')
        for inter_index in range(num_sample_perprompt):
            seed_everything(enu_index + inter_index * 1000)
            seed = enu_index + inter_index * 1000
            first_stage_samples = get_first_results(model, text, num_frames, H,
                                                    W)
            file_name = f'{save_dir}/{enu_index}_{inter_index}_seed_{seed}.mp4'
            second_file_name = f'{save_dir}/{enu_index}_{inter_index}_seed_{seed}_second.mp4'
            joint_file_name = f'{save_dir}/{enu_index}_{inter_index}_seed_{seed}_joint.mp4'
            print(f'save to {file_name}')
            write_video(filename=file_name,
                        fps=8,
                        video_array=first_stage_samples,
                        options={'crf': '5'})

            if not args.skip_second:

                second_stage_samples = get_second_results(
                    second_model, text, first_stage_samples, second_num_frames)

                write_video(filename=second_file_name,
                            fps=8,
                            video_array=second_stage_samples.cpu(),
                            options={'crf': '5'})

                # save joint video
                part_first_stage = first_stage_samples[:second_num_frames]

                target_h, target_w = second_stage_samples.shape[
                    1], second_stage_samples.shape[2]
                part_first_stage = torch.nn.functional.interpolate(
                    part_first_stage.permute(0, 3, 1, 2).contiguous(),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                    antialias=True)
                part_first_stage = part_first_stage.permute(0, 2, 3,
                                                            1).contiguous()

                joint_video = torch.cat(
                    [part_first_stage.cpu(),
                     second_stage_samples.cpu()],
                    dim=-2)

                write_video(filename=joint_file_name,
                            fps=8,
                            video_array=joint_video.cpu(),
                            options={'crf': '15'})


if __name__ == '__main__':
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()
    second_args_list = copy.deepcopy(args_list)

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    second_args_list[1] = args.second[0]
    second_args = get_args(second_args_list)
    second_args = argparse.Namespace(**vars(second_args), **vars(known))
    del second_args.deepspeed_config
    second_args.model_config.first_stage_config.params.cp_size = 1
    second_args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    second_args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    second_args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    sampling_main(args,
                  second_args,
                  model_cls=SATVideoDiffusionEngine,
                  second_model_cls=FlowEngine)
