import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
from skimage.measure import label
from scipy.ndimage import uniform_filter
from scipy import ndimage

from typing import Optional
import torch
import torch.nn.functional as F
import math
import re

from diffusers.models.attention_processor import Attention
from utils.mask_save_function import visualize_bottom_mask_and_matches_cv, top_k_center_shrink_delete_peak
from utils.utils_function import get_cc_watershed


def get_normalization(tmp_torch):
    tmp_torch = (tmp_torch - tmp_torch.min()) / (tmp_torch.max() - tmp_torch.min() + 1e-8)
    return tmp_torch


class EmoStoryAttentionProcessor:
    def __init__(self, is_enhance_elements=False, save_cross_attn_weights=False, emotion_token_indices=None,
                 p_mask=None, boost_factor=1.5, save_step=[]):

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.is_enhance_elements = is_enhance_elements
        self.save_cross_attn_weights = save_cross_attn_weights
        self.emotion_token_indices = emotion_token_indices
        self.boost_factor = boost_factor
        self.attn_weights_bottom_to_prompt = {}
        self.attn_weights_top_to_prompt = {}
        self.save_step = save_step
        self.p_mask = p_mask

    def elements_attention_enhance(
            self,
            attn,
            query, key, value,
            subject_attn_mask_list,
            n_prompt_tokens,
            n_panel_tokens,
    ):
        B, H, Tq, D = query.shape
        #print("query shape: ", query.shape)
        # query shape:  torch.Size([4, 24, 4608, 128])

        scores = torch.stack([attn.get_attention_scores(query=query[i], key=key[i])
                                             for i in range(B)])
        #scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        # scores: [B, H, T, T], T = 4608 = 512 + 2048 * 2

        device = scores.device
        dtype = scores.dtype

        # step1 subject mask
        subject_mask = torch.stack([
            torch.as_tensor(m, device=device, dtype=dtype).flatten()
            for m in subject_attn_mask_list
        ], dim=0)  # [B, N]
        subject_mask = subject_mask[:, None, :, None]  # [B,1,N,1]
        non_subject_mask = 1 - subject_mask

        # step2 element token mask
        p_mask = self.p_mask.to(dtype=dtype, device=device)  # [B,1,1,T]

        # step3 attention bias
        attn_bias = torch.zeros(B, 1, Tq, Tq, device=device, dtype=dtype)
        img_slice = slice(n_prompt_tokens + n_panel_tokens, Tq)
        penalty = self.boost_factor * non_subject_mask * p_mask  # [B,1,N,1] * [B,1,1,T] -> [B,1,N,T]
        attn_bias[:, :, img_slice, :] += penalty

        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )
        return hidden_states

    def elements_attention_weaken_on_subject(
            self,
            attn,
            query, key, value,
            subject_attn_mask_list,
            n_prompt_tokens,
            n_panel_tokens,
    ):
        B, H, Tq, D = query.shape
        #print("query shape: ", query.shape)
        # query shape:  torch.Size([4, 24, 4608, 128])

        scores = torch.stack([attn.get_attention_scores(query=query[i], key=key[i])
                                             for i in range(B)])
        # scores: [B, H, T, T], T = 4608 = 512 + 2048 * 2
        device = scores.device
        dtype = scores.dtype

        # step1 subject mask
        subject_mask = torch.stack([
            torch.as_tensor(m, device=device, dtype=dtype).flatten()
            for m in subject_attn_mask_list
        ], dim=0)  # [B, N]
        subject_mask = subject_mask[:, None, :, None]  # [B,1,N,1]

        # step2 element token mask
        p_mask = self.p_mask.to(dtype=dtype, device=device)  # [B,1,1,T]

        # step3 attention bias
        attn_bias = torch.zeros(B, 1, Tq, Tq, device=device, dtype=dtype)
        img_slice = slice(n_prompt_tokens + n_panel_tokens, Tq)
        penalty = -self.boost_factor * subject_mask * p_mask  # [B,1,N,1] * [B,1,1,T] -> [B,1,N,T]
        attn_bias[:, :, img_slice, :] += penalty

        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )
        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attn_store=None,
        n_prompt_tokens=None,
        n_image_tokens=None,
        ravm_mixing_coef=None,
        first_mixing_block=None,
        last_mixing_block=None,
        first_mixing_denoising_step=None,
        last_mixing_denoising_step=None,
    ) -> torch.FloatTensor:

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        n_panel_tokens = n_image_tokens // 2

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        attn_scores = torch.stack([attn.get_attention_scores(query=query[i], key=key[i]).mean(dim=0)
                                    for i in range(batch_size)])
        # attn_scores -> (B, Tq, Tk)

        # message
        curr_block = attn_store._get_curr_trans_block()
        curr_step = attn_store._get_curr_diffusion_step()

        attn_to_top = attn_scores[:, n_prompt_tokens + n_panel_tokens:, n_prompt_tokens:n_prompt_tokens + n_panel_tokens]
        attn_to_bottom = attn_scores[:, n_prompt_tokens:n_prompt_tokens + n_panel_tokens,  n_prompt_tokens + n_panel_tokens:]

        mut_attn_bt = torch.minimum(attn_to_top, attn_to_bottom.transpose(-1, -2)) # (bsz, 2048, 2048)

        bottom_to_prompt = attn_scores[:, n_prompt_tokens + n_panel_tokens:, :n_prompt_tokens]
        attn_store.store_attention_map(mut_attn_bt, bottom_to_prompt)

        # obtain the subject mask
        subject_attn_mask_list = []
        if first_mixing_denoising_step <= curr_step <= last_mixing_denoising_step:
            # 1-21
            mut_attn_bt = attn_store.aggregate_attn_maps(device=hidden_states.device)  # (bsz, 2048, 2048)
            cross_avg_attn_map = attn_store.get_cross_avg_attn_map()

            if self.is_enhance_elements:
                cross_avg_attn_map = get_normalization(cross_avg_attn_map)

                for batch in range(batch_size):
                    prompt_to_area = cross_avg_attn_map[batch]  # (T_bottom, T_prompt)
                    prompt_to_area = prompt_to_area.detach().cpu()  # (T_prompt, T_bottom)
                    # Convert to float32 and then to numpy to avoid bfloat16 errors.

                    subject_token_idx = [4, 5, 6, 7]
                    for subject in self.emotion_token_indices["subject"]:
                        subject_token_idx = self.emotion_token_indices["subject"][subject][batch]

                    heat = prompt_to_area[:, subject_token_idx].mean(dim=-1).to(torch.float32).reshape(32, 64).numpy()
                    norm_heat = get_normalization(heat)  # 归一化

                    mask = top_k_center_shrink_delete_peak(norm_heat)
                    subject_attn_mask_list.append(mask)

            for i in range(batch_size):
                # MUTUAL ATTENTION
                bottom = mut_attn_bt[i, :, :].max(dim=-1).values

                # Convert to numpy so that we can use OpenCV
                bottom = bottom.to(torch.float32).cpu().numpy()
                bottom_labels = get_cc_watershed(bottom.reshape(32, 64)) # 2048 就得换成 64

                # Constrained by the subject semantic attention map
                if self.is_enhance_elements:
                    bottom_labels = (bottom_labels > 0) & (subject_attn_mask_list[i] > 0)
                    subject_attn_mask_list[i] = bottom_labels
                    if curr_block == first_mixing_block and i == 0:
                        print(f"The subject area of interest constraint is applied when the denoising step = {curr_step}")

                # Back to GPU-acceleration
                bottom_labels = torch.from_numpy(bottom_labels).to(device=hidden_states.device, dtype=torch.uint8)

                if first_mixing_block <= curr_block <= last_mixing_block:
                    # Get flattened token indices
                    bottom_flat = bottom_labels.flatten()
                    bottom_indices = torch.where(bottom_flat > 0)[0]
                    # Slice mutual attention (B x T): [bottom_indices, top_indices]
                    attention_scores = mut_attn_bt[i, :, :].index_select(0, bottom_indices)  # shape: (k, 2048)
                    # For each bottom token, find top token it attends to most
                    top_matches = attention_scores.argmax(dim=-1)  # (k,)
                    # argmax(dim=-1) take the index of the maximum value of top on each row

                    # Map matched top indices
                    matched_top_indices = top_matches  # (k,)

                    # save top and bottom matched mask
                    if curr_block == 40 and curr_step in self.save_step:
                        image_output_dir = f"{attn_store.return_output_dir()}/mutual_attention_mask"
                        visualize_bottom_mask_and_matches_cv(
                            bottom_labels,
                            matched_top_indices,
                            image_output_dir,
                            curr_step,
                            curr_block,
                            batch=i,
                        )

                    save = value[i, :, n_prompt_tokens + n_panel_tokens + bottom_indices, :]  # (H, k, D)
                    paste = value[i, :, n_prompt_tokens + matched_top_indices, :]  # (H, k, D)
                    blended = (1 - ravm_mixing_coef) * save + ravm_mixing_coef * paste
                    value[i, :, n_prompt_tokens + n_panel_tokens + bottom_indices, :] = blended

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # decouples the constraint element from the body
        if self.is_enhance_elements and first_mixing_denoising_step <= curr_step <= 10 \
                and first_mixing_block <= curr_block <= last_mixing_block:
            hidden_states = self.elements_attention_enhance(attn, query, key, value, subject_attn_mask_list,
                                                    n_prompt_tokens, n_panel_tokens)
            if curr_block == first_mixing_block:
                print(f"The elements attention is enhanced when the denoising step = {curr_step}")
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        #hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
       
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

