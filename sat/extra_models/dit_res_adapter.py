import copy
import random

import torch
from dit_video_concat import (DiffusionTransformer,
                              Rotary3DPositionEmbeddingMixin, broadcat,
                              rotate_half)
from einops import rearrange, repeat
from sgm.modules.diffusionmodules.util import timestep_embedding
from torch import nn

from sat.transformer_defaults import HOOKS_DEFAULT


class ScaleCropRotary3DPositionEmbeddingMixin(Rotary3DPositionEmbeddingMixin):

    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        hidden_size_head,
        text_length,
        theta=10000,
        h_interp_ratio=1.0,
        w_interp_ratio=1.0,
        t_interp_ratio=1.0,
        rot_v=False,
        learnable_pos_embed=False,
    ):
        super(Rotary3DPositionEmbeddingMixin, self).__init__()
        self.rot_v = rot_v
        print(f'theta is {theta}')
        dim_t = hidden_size_head // 4
        dim_h = hidden_size_head // 8 * 3
        dim_w = hidden_size_head // 8 * 3

        self.freqs_t = (1.0 / (theta**(
            torch.arange(0, dim_t, 2)[:(dim_t // 2)].float() / dim_t))).cuda()
        self.freqs_h = (1.0 / (theta**(
            torch.arange(0, dim_h, 2)[:(dim_h // 2)].float() / dim_h))).cuda()
        self.freqs_w = (1.0 / (theta**(
            torch.arange(0, dim_w, 2)[:(dim_w // 2)].float() / dim_w))).cuda()

        self.compressed_num_frames = compressed_num_frames
        self.height = height
        self.width = width
        self.text_length = text_length
        if learnable_pos_embed:
            num_patches = height * width * compressed_num_frames + text_length
            self.pos_embedding = nn.Parameter(torch.zeros(
                1, num_patches, int(hidden_size)),
                                              requires_grad=True)
        else:
            self.pos_embedding = None

    def online_sin_cos(self, real_t, real_h, real_w, dy_interpolation=None):

        grid_t = torch.arange(real_t, dtype=torch.float32, device='cuda')
        grid_h = torch.arange(real_h, dtype=torch.float32, device='cuda')
        grid_w = torch.arange(real_w, dtype=torch.float32, device='cuda')
        freqs_t = self.freqs_t
        freqs_h = self.freqs_h
        freqs_w = self.freqs_w

        freqs_t = torch.einsum('..., f -> ... f', grid_t, freqs_t)
        freqs_h = torch.einsum('..., f -> ... f', grid_h, freqs_h)
        freqs_w = torch.einsum('..., f -> ... f', grid_w, freqs_w)

        freqs_t = repeat(freqs_t, '... n -> ... (n r)', r=2)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)

        freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :],
                          freqs_w[None, None, :, :]),
                         dim=-1)
        freqs = rearrange(freqs, 't h w d -> (t h w) d')

        temp_layer_id = self.share_cache['temp_layer_id']
        emb = self.share_cache['emb']
        if f'rope_layer_{temp_layer_id}' in self.share_cache:
            m = self.share_cache[f'rope_layer_{temp_layer_id}']

            dy_interpolation = m(emb)
        else:
            dy_interpolation = freqs.new_ones(emb.shape[0], freqs.shape[-1])

        b, dim = dy_interpolation.shape
        dy_interpolation = dy_interpolation[:, None]
        freqs = freqs[None]
        freqs = freqs.repeat(b, 1, 1)
        if dy_interpolation.shape[-1] != freqs.shape[-1]:
            freqs[...,-dy_interpolation.shape[-1]:] =  \
                freqs[...,-dy_interpolation.shape[-1]:] * dy_interpolation

        else:
            freqs = freqs * dy_interpolation

        freqs_sin = torch.sin(freqs)
        freqs_cos = torch.cos(freqs)

        return freqs_cos, freqs_sin

    def rotary(self, t, **kwargs):
        if 'freqs_cos' in self.share_cache:
            freqs_cos = self.share_cache['freqs_cos']
            freqs_sin = self.share_cache['freqs_sin']
        else:
            real_t, real_h, real_w = self.share_cache['shape_info']
            freqs_cos, freqs_sin = self.online_sin_cos(real_t,
                                                       real_h,
                                                       real_w,
                                                       dy_interpolation=None)
            freqs_cos = freqs_cos.unsqueeze(1)
            freqs_cos = freqs_cos.to(t.dtype)
            freqs_sin = freqs_sin.unsqueeze(1)
            freqs_sin = freqs_sin.to(t.dtype)
            self.share_cache['freqs_cos'] = freqs_cos
            self.share_cache['freqs_sin'] = freqs_sin

        return t * freqs_cos + rotate_half(t) * freqs_sin

    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        **kwargs,
    ):
        attention_fn_default = HOOKS_DEFAULT['attention_fn']

        img_query_layer = self.rotary(query_layer[:, :, self.text_length:])
        query_layer = torch.cat(
            [query_layer[:, :, :self.text_length], img_query_layer], dim=2)
        query_layer = query_layer.to(value_layer.dtype)
        img_key_layer = self.rotary(key_layer[:, :, self.text_length:])
        key_layer = torch.cat(
            [key_layer[:, :, :self.text_length], img_key_layer], dim=2)
        key_layer = key_layer.to(value_layer.dtype)

        if self.rot_v:
            value_layer[:, :, self.text_length:] = self.rotary(
                value_layer[:, :, self.text_length:])

        return attention_fn_default(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )


from sat.model.finetune.lora2 import *


class ResLoraMixin(LoraMixin):

    def reinit(self, parent_model):
        for i in self.layer_range:
            print_rank0(f'replacing layer {i} attention with lora')
            parent_model.transformer.layers[
                i].attention.dense = replace_linear_with_lora(
                    parent_model.transformer.layers[i].attention.dense,
                    1,
                    self.r,
                    self.lora_alpha,
                    self.lora_dropout,
                    qlora=self.qlora,
                    in_size=parent_model.transformer.hidden_size,
                    out_size=None)
            parent_model.transformer.layers[
                i].attention.query_key_value = replace_linear_with_lora(
                    parent_model.transformer.layers[i].attention.
                    query_key_value,
                    parent_model.transformer.layers[i].attention.stride,
                    self.r,
                    self.lora_alpha,
                    self.lora_dropout,
                    qlora=self.qlora,
                    in_size=parent_model.transformer.hidden_size,
                    out_size=None
                    if not parent_model.transformer.num_multi_query_heads else
                    parent_model.transformer.layers[i].attention.
                    inner_hidden_size + parent_model.transformer.layers[i].
                    attention.hidden_size_per_attention_head * parent_model.
                    transformer.layers[i].attention.num_multi_query_heads * 2)
            if self.cross_attention and parent_model.transformer.layers[
                    i].is_decoder:
                print_rank0(f'replacing layer {i} cross attention with lora')
                kv_size = parent_model.transformer.layers[
                    i].cross_attention.inner_hidden_size * 2 if not parent_model.transformer.cross_num_multi_query_heads else parent_model.transformer.layers[
                        i].cross_attention.hidden_size_per_attention_head * parent_model.transformer.layers[
                            i].cross_attention.cross_num_multi_query_heads * 2
                parent_model.transformer.layers[
                    i].cross_attention.dense = replace_linear_with_lora(
                        parent_model.transformer.layers[i].cross_attention.
                        dense,
                        1,
                        self.r,
                        self.lora_alpha,
                        self.lora_dropout,
                        qlora=self.qlora,
                        in_size=parent_model.transformer.layers[i].
                        cross_attention.inner_hidden_size,
                        out_size=parent_model.transformer.hidden_size)
                parent_model.transformer.layers[
                    i].cross_attention.query = replace_linear_with_lora(
                        parent_model.transformer.layers[i].cross_attention.
                        query,
                        1,
                        self.r,
                        self.lora_alpha,
                        self.lora_dropout,
                        qlora=self.qlora,
                        in_size=parent_model.transformer.hidden_size,
                        out_size=parent_model.transformer.layers[i].
                        cross_attention.inner_hidden_size)
                parent_model.transformer.layers[
                    i].cross_attention.key_value = replace_linear_with_lora(
                        parent_model.transformer.layers[i].cross_attention.
                        key_value,
                        2,
                        self.r,
                        self.lora_alpha,
                        self.lora_dropout,
                        qlora=self.qlora,
                        in_size=parent_model.transformer.layers[i].
                        cross_attention.cross_attn_hidden_size,
                        out_size=kv_size)

        for m in parent_model.mixins.adaln_layer.adaLN_modulations:
            m[1] = replace_linear_with_lora(m[1],
                                            1,
                                            self.r,
                                            self.lora_alpha,
                                            self.lora_dropout,
                                            qlora=self.qlora,
                                            in_size=512,
                                            out_size=36864)

    def merge_lora(self):
        for i in self.layer_range:
            print_rank0(f'merge layer {i} lora attention back to linear')
            self.transformer.layers[i].attention.dense = merge_linear_lora(
                self.transformer.layers[i].attention.dense)
            self.transformer.layers[
                i].attention.query_key_value = merge_linear_lora(
                    self.transformer.layers[i].attention.query_key_value)
            if self.cross_attention and self.transformer.layers[i].is_decoder:
                print_rank0(
                    f'merge layer {i} lora cross attention back to linear')
                self.transformer.layers[
                    i].cross_attention.dense = merge_linear_lora(
                        self.transformer.layers[i].cross_attention.dense)
                self.transformer.layers[
                    i].cross_attention.query = merge_linear_lora(
                        self.transformer.layers[i].cross_attention.query)
                self.transformer.layers[
                    i].cross_attention.key_value = merge_linear_lora(
                        self.transformer.layers[i].cross_attention.key_value)


class SMALLDiffusionTransformer(DiffusionTransformer):

    def register_new_modules(self):

        if 'sample_ref_noise_step' in self.share_cache:
            self.ref_step_time_embedding = copy.deepcopy(self.time_embed)
            # zero init last linear in the self.ref_step_time_embedding
            for n, p in self.ref_step_time_embedding[-1].named_parameters():
                nn.init.constant_(p, 0)
                p.requires_grad = True
        if 'time_size_embedding' in self.share_cache:
            self.time_size_embedding = copy.deepcopy(self.time_embed)

            # zero init the fuse linear
            for n, p in self.time_size_embedding.named_parameters():
                nn.init.constant_(p, 0)
                p.requires_grad = True

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        b, t, d, h, w = x.shape

        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        if 'ref_x' in self.share_cache:
            ref_x = self.share_cache['ref_x']
            if ref_x.dtype != self.dtype:
                ref_x = ref_x.to(self.dtype)
            self.share_cache['ref_x'] = ref_x
        # This is not use in inference
        if 'concat_images' in kwargs and kwargs['concat_images'] is not None:
            if kwargs['concat_images'].shape[0] != x.shape[0]:
                concat_images = kwargs['concat_images'].repeat(2, 1, 1, 1, 1)
            else:
                concat_images = kwargs['concat_images']
            x = torch.cat([x, concat_images], dim=2)

        assert (y is not None) == (
            self.num_classes is not None
        ), 'must specify y if and only if the model is class-conditional'
        t_emb = timestep_embedding(timesteps,
                                   self.model_channels,
                                   repeat_only=False,
                                   dtype=self.dtype)
        emb = self.time_embed(t_emb)

        if 'time_size_embedding' in self.share_cache:
            num_t = torch.zeros_like(timesteps).fill_(int(t))
            time_size_emb = timestep_embedding(num_t,
                                               self.model_channels,
                                               repeat_only=False,
                                               dtype=self.dtype)
            time_size_emb = self.time_size_embedding(time_size_emb)
            emb = emb + time_size_emb

        if 'sample_ref_noise_step' in self.share_cache:
            print(
                f'''sample_ref_noise_step {self.share_cache["sample_ref_noise_step"]}'''
            )
            # bf 16
            ref_time_step = copy.deepcopy(timesteps).fill_(
                self.share_cache['sample_ref_noise_step'])
            ref_step_time_emb = timestep_embedding(ref_time_step,
                                                   self.model_channels,
                                                   repeat_only=False,
                                                   dtype=self.dtype)
            ref_step_time_emb = self.ref_step_time_embedding(ref_step_time_emb)
            if not self.training:
                print(f'{ref_time_step} get {ref_step_time_emb.sum()}')
            emb = emb + ref_step_time_emb

        if self.num_classes is not None:
            assert x.shape[0] % y.shape[0] == 0
            y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
            emb = emb + self.label_emb(y)

        self.share_cache['shape_info'] = (t, h // (self.patch_size),
                                          w // (self.patch_size))
        self.share_cache['timesteps'] = int(timesteps[0])
        self.share_cache['emb'] = emb

        kwargs['seq_length'] = t * h * w // (self.patch_size**2)
        kwargs['images'] = x
        kwargs['emb'] = emb
        kwargs['encoder_outputs'] = context
        kwargs['text_length'] = context.shape[1]

        kwargs['input_ids'] = kwargs['position_ids'] = kwargs[
            'attention_mask'] = torch.ones((1, 1)).to(x.dtype)
        output = super(DiffusionTransformer, self).forward(**kwargs)[0]
        return output
