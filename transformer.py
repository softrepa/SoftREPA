from typing import Any, Dict, List, Optional, Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import is_torch_version

import torch
import torch.nn as nn


class SD3Transformer2DModel_DC(nn.Module):
    def __init__(self, base_model, n_dc_tokens:int=8, use_dc_t=True, n_dc_layers:int=6): #check
        super().__init__()

        self.base_model = base_model
        self.n_dc_layers = n_dc_layers if n_dc_layers != -1 else self.base_model.config.num_layers
        self.n_dc_tokens = n_dc_tokens
        self.use_dc_t = use_dc_t
        if use_dc_t:
            self.dc_t_tokens = nn.Embedding(100, self.base_model.config.caption_projection_dim*self.n_dc_layers)

        # self.dc_tokens = nn.Parameter(torch.randn(self.n_dc_layers, self.n_dc_tokens, self.base_model.config.caption_projection_dim)) # (nblocks, 1, embed_dims)
        self.dc_tokens = nn.Parameter(torch.randn(self.n_dc_layers, self.n_dc_tokens, 4096))
        # initialize
        nn.init.normal_(self.dc_tokens, mean=0, std=0.02)
        if use_dc_t:
            nn.init.normal_(self.dc_t_tokens.weight, mean=0, std=0.02)
        '''
        FrozenDict([('_class_name', 'SD3Transformer2DModel'), ('_diffusers_version', '0.29.0.dev0'), 
            ('_name_or_path', '/mnt/data1/jaayeon/cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671/transformer'), 
            ('sample_size', 128), ('patch_size', 2), ('in_channels', 16), ('num_layers', 24), 
            ('attention_head_dim', 64), ('num_attention_heads', 24), ('joint_attention_dim', 4096), 
            ('caption_projection_dim', 1536), ('pooled_projection_dim', 2048), ('out_channels', 16), 
            ('pos_embed_max_size', 192)])
        '''

    def to(self, device):
        super().to(device)
        self.device = device
        return self
    
    def initialize_dc(self, dc_tokens, dc_t_tokens=None):
        self.dc_tokens.data = dc_tokens.type(self.dc_tokens.data.dtype)
        if dc_t_tokens is not None:
            self.dc_t_tokens.weight.data = dc_t_tokens.type(self.dc_tokens.data.dtype)
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        dc_tokens: bool = False, # added
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        '''
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of input conditions.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        '''
        height, width = hidden_states.shape[-2:]

        hidden_states = self.base_model.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.base_model.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.base_model.context_embedder(encoder_hidden_states)

        # dc time embeddings for each timestep (should be added to the dc_tokens)
        if self.use_dc_t:
            int_t = timestep.type(torch.int32)
            if torch.sum(int_t>=1000)>0: # index supported [0, 999]
                int_t-=1
            dctemb = self.dc_t_tokens(int_t//10).contiguous().to(hidden_states.device)
            dctemb = dctemb.chunk(self.n_dc_layers, dim=-1)

        n=encoder_hidden_states.size(-2)
        for index_block, block in enumerate(self.base_model.transformer_blocks):
            # Skip specified layers
            is_skip = True if skip_layers is not None and index_block in skip_layers else False

            if dc_tokens: # remove learnable prompts from previous layers
                encoder_hidden_states = encoder_hidden_states[:, -n:, :] #text

            # Use Contrastive(discriminative) prompts
            if dc_tokens and index_block<self.n_dc_layers: #check
                dc1 = self.dc_tokens[index_block].expand(encoder_hidden_states.shape[0], -1, -1)
                if dc1.shape[-1] == 4096:
                    dc1 = self.base_model.context_embedder(dc1)
                if self.use_dc_t:
                    dc2 = dctemb[index_block].unsqueeze(1).expand(-1, self.n_dc_tokens, -1)
                    dc_add = dc1 + dc2
                else:
                    dc_add = dc1
                encoder_hidden_states = torch.cat([dc_add.type(encoder_hidden_states.dtype), encoder_hidden_states], dim=1) # (B,S,C)+(B,N,C) -> (B,S+N,C)

            # not used
            if torch.is_grad_enabled() and self.base_model.gradient_checkpointing and not is_skip:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                    **ckpt_kwargs,
                )
            # used
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    # joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = self.base_model.norm_out(hidden_states, temb)
        hidden_states = self.base_model.proj_out(hidden_states)

        # unpatchify
        patch_size = self.base_model.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.base_model.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.base_model.out_channels, height * patch_size, width * patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

class UNet2DConditionModel_DC(nn.Module):
    def __init__(self, base_model, n_dc_tokens, use_dc_t=True, apply_dc=[True, True, False]): #check
        super().__init__()

        self.base_model = base_model
        self.n_dc_tokens = n_dc_tokens
        self.use_dc_t = use_dc_t
        self.apply_dc = apply_dc
        if use_dc_t:
            self.dc_t_tokens = nn.Embedding(100, self.base_model.config.cross_attention_dim)

        self.dc_tokens = nn.Parameter(torch.randn(self.n_dc_tokens, self.base_model.config.cross_attention_dim))
        # initialize
        nn.init.normal_(self.dc_tokens, mean=0, std=0.02)
        if use_dc_t:
            nn.init.normal_(self.dc_t_tokens.weight, mean=0, std=0.02)
        '''
        FrozenDict([('_class_name', 'SD3Transformer2DModel'), ('_diffusers_version', '0.29.0.dev0'), 
            ('_name_or_path', '/mnt/data1/jaayeon/cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/ea42f8cef0f178587cf766dc8129abd379c90671/transformer'), 
            ('sample_size', 128), ('patch_size', 2), ('in_channels', 16), ('num_layers', 24), 
            ('attention_head_dim', 64), ('num_attention_heads', 24), ('joint_attention_dim', 4096), 
            ('caption_projection_dim', 1536), ('pooled_projection_dim', 2048), ('out_channels', 16), 
            ('pos_embed_max_size', 192)])
        '''

    def initialize_dc(self, dc_tokens, dc_t_tokens=None):
        self.dc_tokens.data = dc_tokens.type(self.dc_tokens.data.dtype)
        if dc_t_tokens is not None:
            self.dc_t_tokens.weight.data = dc_t_tokens

    def to(self, device):
        super().to(device)
        self.device = device
        return self
    
    def forward(
        self,
        sample: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        return_dict: bool = True,
        dc_tokens: bool = False, # added
        timestep_cond = None,
        class_labels = None,
        added_cond_kwargs = None,
        attention_mask=None,
        encoder_attention_mask=None,
        cross_attention_kwargs=None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        '''
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of input conditions.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        '''
        apply_dc = self.apply_dc
        if self.use_dc_t:
            int_t = timestep.type(torch.int32)
            if torch.sum(int_t>=1000)>0:
                int_t-=1
            dctemb = self.dc_t_tokens(int_t//10).contiguous().to(sample.device)

        if dc_tokens: 
            dc1 = self.dc_tokens.unsqueeze(0).expand(encoder_hidden_states.shape[0], -1, -1)
            if self.use_dc_t:
                dc2 = dctemb.unsqueeze(1).expand(-1, self.n_dc_tokens, -1)
            else:
                dc2 = torch.zeros_like(dc1, dtype=dc1.dtype, device=dc1.device)
            dc_add = dc1 + dc2
            modified_encoder_hidden_states = torch.cat([encoder_hidden_states[:, 0:1], dc_add.type(encoder_hidden_states.dtype), encoder_hidden_states[:, 1:]], dim=1) # (B,1,C)+(B,S,C) -> (B,S+1,C)
        else:
            modified_encoder_hidden_states = encoder_hidden_states

        default_overall_up_factor = 2**self.base_model.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.base_model.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.base_model.get_time_embed(sample=sample, timestep=timestep)
        emb = self.base_model.time_embedding(t_emb, timestep_cond)

        class_emb = self.base_model.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.base_model.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.base_model.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        modified_aug_emb =  self.base_model.get_aug_embed(
            emb=emb, encoder_hidden_states=modified_encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.base_model.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.base_model.time_embed_act is not None:
            emb = self.base_model.time_embed_act(emb)

        encoder_hidden_states = self.base_model.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        modified_encoder_hidden_states = self.base_model.process_encoder_hidden_states(
            encoder_hidden_states=modified_encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.base_model.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.base_model.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        down_block_res_samples = (sample,)
        for downsample_block in self.base_model.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=(modified_encoder_hidden_states if apply_dc[0] else encoder_hidden_states),
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                
            down_block_res_samples += res_samples

        # 4. mid
        if self.base_model.mid_block is not None:
            if hasattr(self.base_model.mid_block, "has_cross_attention") and self.base_model.mid_block.has_cross_attention:
                sample = self.base_model.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=(modified_encoder_hidden_states if apply_dc[1] else encoder_hidden_states),
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.base_model.mid_block(sample, emb)

        # 5. up
        for i, upsample_block in enumerate(self.base_model.up_blocks):
            is_final_block = i == len(self.base_model.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=(modified_encoder_hidden_states if apply_dc[2] else encoder_hidden_states),
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.base_model.conv_norm_out:
            sample = self.base_model.conv_norm_out(sample)
            sample = self.base_model.conv_act(sample)
        sample = self.base_model.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

class UNet2DConditionModel_DC3(nn.Module):
    def __init__(self, base_model, n_dc_tokens, use_dc_t=True, apply_dc=[True, True, False]): #check
        super().__init__()

        self.base_model = base_model
        self.n_dc_tokens = n_dc_tokens
        self.use_dc_t = use_dc_t
        self.apply_dc = apply_dc
        if use_dc_t:
            self.dc_t_tokens = nn.Embedding(100, self.base_model.config.cross_attention_dim)
        self.dc_tokens = nn.Parameter(torch.randn(self.n_dc_tokens, self.base_model.config.cross_attention_dim))
        # initialize
        nn.init.normal_(self.dc_tokens, mean=0, std=0.02)
        if use_dc_t:
            nn.init.normal_(self.dc_t_tokens.weight, mean=0, std=0.02)

    def initialize_dc(self, dc_tokens, dc_t_tokens=None):
        self.dc_tokens.data = dc_tokens.type(self.dc_tokens.data.dtype)
        if dc_t_tokens is not None:
            self.dc_t_tokens.weight.data = dc_t_tokens

    def to(self, device):
        super().to(device)
        self.device = device
        return self
    
    def forward(
        self,
        sample: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        return_dict: bool = True,
        dc_tokens: bool = False, # added
        timestep_cond = None,
        class_labels = None,
        added_cond_kwargs = None,
        attention_mask=None,
        encoder_attention_mask=None,
        cross_attention_kwargs=None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        '''
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of input conditions.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        '''
        apply_dc = self.apply_dc
        if self.use_dc_t:
            int_t = timestep.type(torch.int32)
            if torch.sum(int_t>=1000)>0:
                int_t-=1
            dctemb = self.dc_t_tokens(int_t//10).contiguous().to(sample.device)

        if dc_tokens: #check
            dc1 = self.dc_tokens.unsqueeze(0).expand(encoder_hidden_states.shape[0], -1, -1)
            if self.use_dc_t:
                dc2 = dctemb.unsqueeze(1).expand(-1, self.n_dc_tokens, -1)
            else:
                dc2 = torch.zeros_like(dc1, dtype=dc1.dtype, device=dc1.device)
            dc_add = dc1 + dc2
            modified_encoder_hidden_states = torch.cat([encoder_hidden_states[:, 0:1], dc_add.type(encoder_hidden_states.dtype), encoder_hidden_states[:, 1:]], dim=1) # (B,1,C)+(B,S,C) -> (B,S+1,C)
        else:
            modified_encoder_hidden_states = encoder_hidden_states

        default_overall_up_factor = 2**self.base_model.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.base_model.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.base_model.get_time_embed(sample=sample, timestep=timestep)
        emb = self.base_model.time_embedding(t_emb, timestep_cond)

        class_emb = self.base_model.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.base_model.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.base_model.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        modified_aug_emb =  self.base_model.get_aug_embed(
            emb=emb, encoder_hidden_states=modified_encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.base_model.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.base_model.time_embed_act is not None:
            emb = self.base_model.time_embed_act(emb)

        encoder_hidden_states = self.base_model.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        modified_encoder_hidden_states = self.base_model.process_encoder_hidden_states(
            encoder_hidden_states=modified_encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.base_model.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.base_model.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        down_block_res_samples = (sample,)
        for downsample_block in self.base_model.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=(modified_encoder_hidden_states if apply_dc[0] else encoder_hidden_states),
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                
            down_block_res_samples += res_samples

        # 4. mid
        if self.base_model.mid_block is not None:
            if hasattr(self.base_model.mid_block, "has_cross_attention") and self.base_model.mid_block.has_cross_attention:
                sample = self.base_model.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=(modified_encoder_hidden_states if apply_dc[1] else encoder_hidden_states),
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.base_model.mid_block(sample, emb)

        # 5. up
        for i, upsample_block in enumerate(self.base_model.up_blocks):
            is_final_block = i == len(self.base_model.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=(modified_encoder_hidden_states if apply_dc[2] else encoder_hidden_states),
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.base_model.conv_norm_out:
            sample = self.base_model.conv_norm_out(sample)
            sample = self.base_model.conv_act(sample)
        sample = self.base_model.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)