import torch
from einops import pack, reduce

class AttentionStore:
    def __init__(self, batch_size, n_diff_steps, n_trans_blocks, n_image_tokens, n_prompt_tokens, n_attn_heads,
                 dtype, device, output_dir, emotion_token_indices):
        self.prompt_token_ids = None
        self.n_diff_steps = n_diff_steps
        self.n_trans_blocks = n_trans_blocks
        self.n_image_tokens = n_image_tokens
        self.n_prompt_tokens = n_prompt_tokens
        self.n_attn_heads = n_attn_heads
        self.curr_iter = -1
        self.internal_device = device
        self.dtype = dtype
        self.output_dir = output_dir
        self.emotion_token_indices = emotion_token_indices

        #self.attn_map_shape = (batch_size, self.n_image_tokens // 2, self.n_image_tokens // 2)
        # (bsz, 1024, 1024)
        self.attn_map_shape = (batch_size, self.n_image_tokens // 2, self.n_image_tokens // 2)
        self.avg_attn_map = torch.zeros(self.attn_map_shape, device=self.internal_device, dtype=dtype)

        # save cross attn
        self.cross_attn_list = []
        self.cross_avg_attn_map = None

        self.curr_diff_step_maps = []
        self.attn_map_decay = 0.5

    def return_output_dir(self):
        return self.output_dir

    def increment(self):
        self.curr_iter += 1

    def _get_curr_diffusion_step(self):
        return self.curr_iter // self.n_trans_blocks
    # 显然从 0 开始

    def _is_first_layer(self):
        return self.curr_iter % self.n_trans_blocks == 0

    def _is_last_block(self):
        return self.curr_iter % self.n_trans_blocks == self.n_trans_blocks - 1

    def _get_curr_trans_block(self):
        return self.curr_iter % self.n_trans_blocks

    def get_emotion_token_indices(self):
        return self.emotion_token_indices

    # online averaging method saves GPU memory
    def store_attention_map(self, attn_map, cross_attn_map):
        assert attn_map.shape == self.attn_map_shape, \
            "Attention map dimensions are incorrect"

        attn_map = attn_map.to(device=self.internal_device) # 传设备
        cross_attn_map = cross_attn_map.to(device=self.internal_device)
        self.curr_diff_step_maps.append(attn_map) # 收集注意力图
        self.cross_attn_list.append(cross_attn_map)

        if self._is_last_block():
            # 最后一个块，则求注意力图平均
            # step_avg_attn_map, _ = pack(self.curr_diff_step_maps, 'c * v_toks v_toks2')
            # # pack 只是分别平均 top_to_bottom 和 bottom_to_top
            # step_avg_attn_map = reduce(step_avg_attn_map, 'channel layer v_toks v_toks2 -> channel v_toks v_toks2', 'mean')
            # # reduce(..., 'mean') 对 block 维度取平均

            # Calculate mutual attention
            step_maps = torch.stack(self.curr_diff_step_maps, dim=0)
            step_avg_attn_map = step_maps.mean(dim=0)

            # Calculate the BtoP attention
            cross_step_maps = torch.stack(self.cross_attn_list, dim=0)
            self.cross_avg_attn_map = cross_step_maps.mean(dim=0)

            curr_step = self._get_curr_diffusion_step()

            self.curr_diff_step_maps = []
            self.cross_attn_list = []

            new_observation = step_avg_attn_map - self.avg_attn_map

            self.avg_attn_map = (self.attn_map_decay * self.avg_attn_map
                                       + (1 - self.attn_map_decay) * new_observation / (curr_step + 1))

    def aggregate_attn_maps(self, device):
        return self.avg_attn_map.to(device)

    def get_cross_avg_attn_map(self):
        return self.cross_avg_attn_map
