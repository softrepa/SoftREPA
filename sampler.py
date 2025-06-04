from typing import List, Tuple, Optional
import torch
from transformers import BitsAndBytesConfig, T5EncoderModel
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline, AutoencoderKL
from transformer import SD3Transformer2DModel_DC, UNet2DConditionModel_DC, UNet2DConditionModel_DC3
from diffusers import SD3Transformer2DModel, UNet2DConditionModel
from diffusers import BitsAndBytesConfig as StableDiffusion3Pipeline2 #StableDiffusionPipeline
from torch.nn.parallel import DistributedDataParallel
from diffusers import DDIMInverseScheduler, DDIMScheduler


class StableDiffusion3Base():
    def __init__(self, model_key:str='stabilityai/stable-diffusion-3-medium-diffusers', device='cuda', dtype=torch.float16, use_8bit=False):
        self.device = device

        self.dtype = dtype
        if use_8bit:
            print('load 8 bit text encoder 3')
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            text_encoder_3 = T5EncoderModel.from_pretrained(model_key, subfolder='text_encoder_3', 
                                                            quantization_config=quant_config, 
                                                            torch_dtype=self.dtype,
                                                            device_map={"":'cuda:0'})
            pipe = StableDiffusion3Pipeline2.from_pretrained(model_key, 
                                                            text_encoder_3=text_encoder_3,
                                                            torch_dtype=self.dtype)
        else:

            pipe = StableDiffusion3Pipeline.from_pretrained(model_key, 
                                                            torch_dtype=self.dtype,)

        self.scheduler = pipe.scheduler

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3

        if use_8bit:
            self.vae=pipe.vae
            self.text_enc_1 = pipe.text_encoder
            self.text_enc_2 = pipe.text_encoder_2
            self.text_enc_3 = pipe.text_encoder_3
            self.denoiser = pipe.transformer
        else:
            self.vae=pipe.vae.to(device)
            self.text_enc_1 = pipe.text_encoder.to(device)
            self.text_enc_2 = pipe.text_encoder_2.to(device)
            self.text_enc_3 = pipe.text_encoder_3.to(device)
            self.denoiser = pipe.transformer.to(device)
        self.denoiser.eval()
        self.denoiser.requires_grad_(False)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)-1) if hasattr(self, "vae") and self.vae is not None else 8
        )

        del pipe

    @torch.no_grad()
    def encode_prompt(self, prompt: List[str], batch_size:int=1) -> List[torch.Tensor]:
        '''
        We assume that
        1. number of tokens < max_length
        2. one prompt for one image
        '''
        # CLIP encode (used for modulation of adaLN-zero)
        # now, we have two CLIPs
        text_clip1_ids = self.tokenizer_1(prompt,
                                          padding="max_length",
                                          max_length=77,
                                          truncation=True,
                                          return_tensors='pt').input_ids
        text_clip1_emb = self.text_enc_1(text_clip1_ids.to(self.text_enc_1.device), output_hidden_states=True)
        pool_clip1_emb = text_clip1_emb[0].to(dtype=self.dtype, device=self.denoiser.device)
        text_clip1_emb = text_clip1_emb.hidden_states[-2].to(dtype=self.dtype, device=self.denoiser.device)

        text_clip2_ids = self.tokenizer_2(prompt,
                                          padding="max_length",
                                          max_length=77,
                                          truncation=True,
                                          return_tensors='pt').input_ids
        text_clip2_emb = self.text_enc_2(text_clip2_ids.to(self.text_enc_2.device), output_hidden_states=True)
        pool_clip2_emb = text_clip2_emb[0].to(dtype=self.dtype, device=self.denoiser.device)
        text_clip2_emb = text_clip2_emb.hidden_states[-2].to(dtype=self.dtype, device=self.denoiser.device)
        
        # T5 encode (used for text condition)
        text_t5_ids = self.tokenizer_3(prompt,
                                       padding="max_length",
                                       max_length=512,
                                       truncation=True,
                                       add_special_tokens=True,
                                       return_tensors='pt').input_ids
        text_t5_emb = self.text_enc_3(text_t5_ids.to(self.text_enc_3.device))[0]
        text_t5_emb = text_t5_emb.to(dtype=self.dtype, device=self.denoiser.device)

        # Merge
        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        clip_prompt_emb = torch.nn.functional.pad(
            clip_prompt_emb, (0, text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1])
        )
        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)

        return prompt_emb, pooled_prompt_emb


    def initialize_latent(self, img_size:Tuple[int], batch_size:int=1, **kwargs):

        H, W = img_size
        lH, lW = H//self.vae_scale_factor, W//self.vae_scale_factor
        if isinstance(self.denoiser, DistributedDataParallel):
            try:
                lC = getattr(self.denoiser.module, "module", self.denoiser.module).config.in_channels
            except:
                lC = getattr(self.denoiser.module.base_model, "module", self.denoiser.module.base_model).config.in_channels
        else:
            try:
                lC = getattr(self.denoiser, "module", self.denoiser).config.in_channels
            except:
                lC = getattr(self.denoiser.base_model, "module", self.denoiser.base_model).config.in_channels
        latent_shape = (batch_size, lC, lH, lW)

        z = torch.randn(latent_shape, device=self.denoiser.device, dtype=self.dtype)

        return z

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image).latent_dist.sample()
        z = (z-self.vae.config.shift_factor) * self.vae.config.scaling_factor
        z = z.to(self.denoiser.device)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z/self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]
    
    def predict_vector(self, z, t, prompt_emb, pooled_emb):
        v = self.denoiser(hidden_states=z,
                             timestep=t,
                             pooled_projections=pooled_emb,
                             encoder_hidden_states=prompt_emb,
                             return_dict=False)[0]
        return v 


class SD3Euler(StableDiffusion3Base):
    def __init__(self, model_key:str='stabilityai/stable-diffusion-3-medium-diffusers', device='cuda', use_8bit=True):
        super().__init__(model_key=model_key, device=device, use_8bit=use_8bit)

    def inversion(self, src_img, prompts: List[str], NFE:int, cfg_scale: float=1.0, batch_size: int=1):

        # encode text prompts
        prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
        null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)

        # initialize latent
        src_img = src_img.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            z = self.encode(src_img)
            z0 = z.clone()

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        timesteps = torch.cat([timesteps, torch.zeros(1, device=self.device)])
        timesteps = reversed(timesteps)
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps[:-1], total=NFE, desc='SD3 Euler Inversion')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
            else:
                pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1]

            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))

        return z


    def sample(self, prompts: List[str], NFE:int, img_shape: Optional[Tuple[int]]=None, cfg_scale: float=1.0, batch_size: int = 1, latent:Optional[torch.Tensor]=None):
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3 Euler') #1000 -> 0 
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb)
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb)
            else:
                pred_null_v = 0.0

            _cfg_scale = cfg_scale

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            z = z + (sigma_next - sigma) * (pred_null_v + _cfg_scale * (pred_v - pred_null_v))

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img 
     

class SD3EulerDC(SD3Euler):
    def __init__(self, model_key:str='stabilityai/stable-diffusion-3-medium-diffusers', device='cuda', n_dc_tokens:int=4, use_dc_t=True, n_dc_layers:int=6, use_8bit=True):
        super().__init__(model_key=model_key, device=device, use_8bit=use_8bit)

        custom_transformer = SD3Transformer2DModel_DC(self.denoiser, n_dc_tokens=n_dc_tokens, use_dc_t=use_dc_t,  n_dc_layers=n_dc_layers)

        self.denoiser = custom_transformer.to(device)
        self.denoiser.requires_grad_(False)
        self.denoiser.eval()

    def initialize_dc(self, dc_tokens, dc_t_tokens=None):
        self.denoiser.initialize_dc(dc_tokens=dc_tokens, dc_t_tokens=dc_t_tokens)

    @torch.no_grad()
    def sample(self, prompts: List[str], NFE:int, prompt_emb=None, null_prompt_emb=None, img_shape: Optional[Tuple[int]]=None, cfg_scale: float=1.0, batch_size: int = 1, latent:Optional[torch.Tensor]=None, use_dc:bool=False):
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        if prompt_emb is None:
            with torch.no_grad():
                prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
        else:
            prompt_emb, pooled_emb = prompt_emb
        
        if null_prompt_emb is None:
            with torch.no_grad():
                null_prompt_emb, null_pooled_emb = self.encode_prompt([""]*len(prompts), batch_size)
        else:
            null_prompt_emb, null_pooled_emb = null_prompt_emb

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.denoiser.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        # print(timesteps)
        # Solve ODE
        for i, t in enumerate(timesteps):
            timestep = t.expand(z.shape[0]).to(self.denoiser.device)
            pred_v = self.predict_vector(z, timestep, prompt_emb, pooled_emb, use_dc=use_dc)
            if cfg_scale != 1.0:
                pred_null_v = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_emb, use_dc=use_dc)
            else:
                pred_null_v = 0.0

            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            # cfg
            z = z + (sigma_next - sigma) * (pred_null_v + cfg_scale * (pred_v - pred_null_v))
            
        # decode
        with torch.no_grad():
            img = self.decode(z.to(self.vae.device))
        return img 
    
    def predict_vector(self, z, t, prompt_emb, pooled_emb, use_dc=False):
        v = self.denoiser(hidden_states=z,
                             timestep=t,
                             pooled_projections=pooled_emb,
                             encoder_hidden_states=prompt_emb,
                             return_dict=False,
                             dc_tokens=use_dc)[0]
        return v 


    def set_noise(self, img_shape:Tuple[int], batch_size:int=1):
        self.all_noise = self.initialize_latent(img_shape, batch_size)
        

    def error(self, latent, nidxs, pidxs, prompts_embs, t, noise=None, use_dc=False):
        nidxs = nidxs.to(self.denoiser.device)
        pidxs = pidxs.to(prompts_embs[0].device)
        t = t.to(self.denoiser.device)
        if noise is None:
            noise = self.all_noise[nidxs]
        prompt_emb, pooled_emb =prompts_embs[0][pidxs], prompts_embs[1][pidxs]
        timestep = t.expand(noise.shape[0]).to(self.denoiser.device)
        sigma = t / self.scheduler.config.num_train_timesteps

        zt = (1-sigma) * latent + sigma * noise
        pred_v = self.predict_vector(zt, timestep, prompt_emb, pooled_emb, use_dc=use_dc)
        v = noise-latent

        return v, pred_v


class SD3EulerFE(SD3Euler):
    def __init__(self, model_key:str='stabilityai/stable-diffusion-3-medium-diffusers', device='cuda', use_8bit=False):
        super().__init__(model_key=model_key, device=device, use_8bit=use_8bit)

    def ch_transformer(self, n_dc_layers=8, n_dc_tokens=4, use_dc_t=True, device='cuda'):
        custom_transformer = SD3Transformer2DModel_DC(self.denoiser, n_dc_tokens=n_dc_tokens, use_dc_t=use_dc_t, n_dc_layers=n_dc_layers)
        self.denoiser = custom_transformer.to(device)
        self.denoiser.requires_grad_(False)
        self.denoiser.eval()
    
    def predict_vector(self, z, t, prompt_emb, pooled_emb, use_dc=False):
        v = self.denoiser(hidden_states=z,
                             timestep=t,
                             pooled_projections=pooled_emb,
                             encoder_hidden_states=prompt_emb,
                             return_dict=False,
                             dc_tokens=use_dc)[0]
        return v 

    def sample(self, src_img:torch.Tensor, prompts:List[str], target_prompts:List[str], NFE:int, img_shape:Optional[Tuple[int]], src_cfg_scale:float=3.5, tar_cfg_scale:float=13.5, n_start:int=33, use_dc:bool=False, batch_size:int=1):
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        with torch.no_grad():
            prompt_emb, pooled_emb = self.encode_prompt(prompts, batch_size)
            tprompt_emb, tpooled_emb = self.encode_prompt(target_prompts, batch_size)
            null_prompt_emb, null_pooled_emb = self.encode_prompt([""], batch_size)

            prompt_emb = prompt_emb.to(self.denoiser.device)
            pooled_emb = pooled_emb.to(self.denoiser.device)
            tprompt_emb = tprompt_emb.to(self.denoiser.device)
            tpooled_emb = tpooled_emb.to(self.denoiser.device)
            null_prompt_emb = null_prompt_emb.to(self.denoiser.device)
            null_pooled_emb = null_pooled_emb.to(self.denoiser.device)

        # initial
        with torch.no_grad():
            zsrc = self.encode(src_img.to(self.vae.device).half())
            zsrc = zsrc.to(self.denoiser.device)
        z = zsrc.clone()

        # timesteps (default option. You can make your custom here.)
        self.scheduler.set_timesteps(NFE, device=self.denoiser.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD3 FlowEdit')
        for i, t in enumerate(pbar):

            n = n_start
            if i < NFE-n:  # skip. lower n will keep the source.
                continue

            timestep = t.expand(batch_size)
            sigma = sigmas[i]
            sigma_next = sigmas[i+1] if i+1 < NFE else 0.0

            noise = torch.randn_like(zsrc)
            ztsrc = (1-sigma) * zsrc + sigma * noise # forward
            zt = z + (ztsrc - zsrc)

            with torch.no_grad():
                # v for current estimate
                pred_v = self.predict_vector(zt, timestep, tprompt_emb, tpooled_emb, use_dc=use_dc)
                pred_vn = self.predict_vector(zt, timestep, null_prompt_emb, null_pooled_emb, use_dc=use_dc)
                pred_v = pred_v + tar_cfg_scale * (pred_v - pred_vn)
                
                # v for src estimate
                pred_vy = self.predict_vector(ztsrc, timestep, prompt_emb, pooled_emb, use_dc=use_dc)
                pred_vny = self.predict_vector(ztsrc, timestep, null_prompt_emb, null_pooled_emb, use_dc=use_dc)
                pred_vy = pred_vny + src_cfg_scale * (pred_vy - pred_vny)

                dv = pred_v - pred_vy

            # next step
            z = z + (sigma_next - sigma) * dv

        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img 



class StableDiffusion1Base():
    def __init__(self, model_key:str='stable-diffusion-v1-5/stable-diffusion-v1-5', device='cuda', dtype=torch.float16, use_8bit=False):
        self.device = device

        self.dtype = dtype
        pipe = StableDiffusionPipeline.from_pretrained(model_key, 
                                                        torch_dtype=self.dtype,)
        self.scheduler = pipe.scheduler
        self.tokenizer = pipe.tokenizer

        self.vae=pipe.vae.to(device)
        self.text_enc = pipe.text_encoder.to(device)
        self.denoiser = pipe.unet.to(device)
        self.denoiser.eval()
        self.denoiser.requires_grad_(False)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        del pipe

    @torch.no_grad()
    def encode_prompt(self, prompt: List[str], batch_size:int=1, clip_skip=None) -> List[torch.Tensor]:
        '''
        We assume that
        1. number of tokens < max_length
        2. one prompt for one image
        '''
        # CLIP encode (used for modulation of adaLN-zero)
        text_inputs = self.tokenizer(prompt,
                                padding="max_length",
                                max_length=77,
                                truncation=True,
                                return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )

        if hasattr(self.text_enc.config, "use_attention_mask") and self.text_enc.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.text_enc.device)
        else:
            attention_mask = None
        
        if clip_skip is None:
            prompt_embeds = self.text_enc(text_input_ids.to(self.text_enc.device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = self.text_enc(
                text_input_ids.to(self.text_enc.device), attention_mask=attention_mask, output_hidden_states=True
            )
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = self.text_enc.text_model.final_layer_norm(prompt_embeds)

        if self.text_enc is not None:
            prompt_embeds_dtype = self.text_enc.dtype
        elif self.denoiser is not None:
            prompt_embeds_dtype = self.denoiser.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype
        
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.denoiser.device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        return prompt_embeds,

    def initialize_latent(self, img_size:Tuple[int], batch_size:int=1, **kwargs):

        H, W = img_size
        lH, lW = H//self.vae_scale_factor, W//self.vae_scale_factor
        if isinstance(self.denoiser, DistributedDataParallel):
            if isinstance(self.denoiser.module, UNet2DConditionModel):
                lC = self.denoiser.module.config.in_channels
            else:
                lC = self.denoiser.module.base_model.config.in_channels
        else:
            if isinstance(self.denoiser, UNet2DConditionModel):
                lC = self.denoiser.config.in_channels
            else:
                lC = self.denoiser.base_model.config.in_channels
        latent_shape = (batch_size, lC, lH, lW)

        z = torch.randn(latent_shape, device=self.denoiser.device, dtype=self.dtype)

        return z

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(image).latent_dist.sample()
        z = (z) * self.vae.config.scaling_factor
        z = z.to(self.denoiser.device)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z/self.vae.config.scaling_factor)
        return self.vae.decode(z, return_dict=False)[0]
    
    def predict_vector(self, z, t, prompt_emb):
        v = self.denoiser(z,
                    timestep=t,
                    encoder_hidden_states=prompt_emb,
                    return_dict=False)[0]
        return v 


class SD1Euler(StableDiffusion1Base):
    def __init__(self, model_key:str='stable-diffusion-v1-5/stable-diffusion-v1-5', device='cuda', dtype=torch.float16, use_8bit=True):
        super().__init__(model_key=model_key, device=device, use_8bit=use_8bit, dtype=dtype)

    def inversion(self, src_img, prompts: List[str], NFE:int, cfg_scale: float=1.0, batch_size: int=1):

        # encode text prompts
        prompt_emb,_ = self.encode_prompt(prompts, batch_size)
        null_prompt_emb,_ = self.encode_prompt([""], batch_size)

        # initialize latent
        src_img = src_img.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            z = self.encode(src_img)
            z0 = z.clone()

        # timesteps (default option. You can make your custom here.)
        ddim_scheduler = DDIMInverseScheduler(
            beta_end=self.scheduler.beta_end,
            beta_schedule=self.scheduler.beta_schedule,
            beta_start=self.scheduler.beta_start,
            num_train_timesteps=self.scheduler.num_train_timesteps,
            set_alpha_to_one=self.scheduler.set_alpha_to_one,
            steps_offset=self.scheduler.steps_offset,
            trained_betas=self.scheduler.trained_betas,
            clip_sample=self.scheduler.clip_sample
        )
        ddim_scheduler.set_timesteps(NFE, device=self.device)
        timesteps = ddim_scheduler.timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SD1 DDIM Inversion')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_epsilon = self.predict_vector(z, timestep, prompt_emb)
            if cfg_scale != 1.0:
                pred_null_epsilon = self.predict_vector(z, timestep, null_prompt_emb)
            else:
                pred_null_epsilon = 0

            model_output = (pred_null_epsilon + cfg_scale * (pred_epsilon - pred_null_epsilon))

            z = ddim_scheduler.step(model_output, t, z, return_dict=False)[0]

        return z


    @torch.no_grad()
    def sample(self, prompts: List[str], NFE:int, prompt_emb=None, null_prompt_emb=None, img_shape: Optional[Tuple[int]]=None, cfg_scale: float=1.0, batch_size: int = 1, latent:Optional[torch.Tensor]=None):
        imgH, imgW = img_shape if img_shape is not None else (512, 512)

        # encode text prompts
        if prompt_emb is None:
            prompt_emb = self.encode_prompt(prompts, batch_size)[0]
        else:
            prompt_emb = prompt_emb[0]
        if null_prompt_emb is None:            
            null_prompt_emb = self.encode_prompt([""]*len(prompts), batch_size)[0]
        else:
            null_prompt_emb = null_prompt_emb[0]
        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = t.expand(z.shape[0]).to(self.denoiser.device)
            pred_epsilon = self.predict_vector(z, timestep, prompt_emb)
            if cfg_scale != 1.0:
                pred_null_epsilon = self.predict_vector(z, timestep, null_prompt_emb)
            else:
                pred_null_epsilon = 0.0

            model_output = (pred_null_epsilon + cfg_scale * (pred_epsilon - pred_null_epsilon))

            z = self.scheduler.step(model_output, t, z, return_dict=False)[0]
            
        # decode
        with torch.no_grad():
            img = self.decode(z.to(self.vae.device))
        return img 

    def set_noise(self, img_shape:Tuple[int], batch_size:int=1):
        self.all_noise = self.initialize_latent(img_shape, batch_size)
    
    def error(self, latent, nidxs, pidxs, prompts_embs, t, noise=None):
        nidxs = nidxs.to(self.denoiser.device)
        pidxs = pidxs.to(prompts_embs[0].device)
        t = t.to(self.denoiser.device)
        if noise is None:
            noise = self.all_noise[nidxs]
        prompt_emb =prompts_embs[0][pidxs]
        timestep = t.expand(noise.shape[0]).to(self.denoiser.device)

        zt = self.scheduler.add_noise(latent, noise, t)
        pred_epsilon = self.predict_vector(zt, timestep, prompt_emb)
        epsilon = noise

        return epsilon, pred_epsilon
    
    def predict_vector(self, z, t, prompt_emb):
        v = self.denoiser(z,
                    timestep=t,
                    encoder_hidden_states=prompt_emb,
                    return_dict=False,
                    )[0]
        return v 


class SD1EulerDC(SD1Euler):
    def __init__(self, model_key:str='stable-diffusion-v1-5/stable-diffusion-v1-5', device='cuda', n_dc_tokens:int=4, use_8bit=True, use_dc_t=True, apply_dc=[True, True, False]):
        super().__init__(model_key=model_key, device=device, use_8bit=use_8bit)

        custom_unet = UNet2DConditionModel_DC(self.denoiser, n_dc_tokens=n_dc_tokens, use_dc_t=use_dc_t, apply_dc=apply_dc)

        self.denoiser = custom_unet.to(device)
        self.denoiser.requires_grad_(False)
        self.denoiser.eval()

        self.scheduler = DDIMScheduler(
            beta_end=self.scheduler.beta_end,
            beta_schedule=self.scheduler.beta_schedule,
            beta_start=self.scheduler.beta_start,
            num_train_timesteps=self.scheduler.num_train_timesteps,
            set_alpha_to_one=self.scheduler.set_alpha_to_one,
            steps_offset=self.scheduler.steps_offset,
            trained_betas=self.scheduler.trained_betas,
            clip_sample=self.scheduler.clip_sample
        )
    
    def initialize_dc(self, dc_tokens, dc_t_tokens=None):
        self.denoiser.initialize_dc(dc_tokens=dc_tokens, dc_t_tokens=dc_t_tokens)

    @torch.no_grad()
    def sample(self, prompts: List[str], NFE:int, prompt_emb=None, null_prompt_emb=None, img_shape: Optional[Tuple[int]]=None, cfg_scale: float=1.0, batch_size: int = 1, latent:Optional[torch.Tensor]=None, use_dc:bool=False):
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        # encode text prompts
        if prompt_emb is None:
            prompt_emb = self.encode_prompt(prompts, batch_size)[0]
        else:
            prompt_emb = prompt_emb[0]
        if null_prompt_emb is None:            
            null_prompt_emb = self.encode_prompt([""]*len(prompts), batch_size)[0]
        else:
            null_prompt_emb = null_prompt_emb[0]
        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = t.expand(z.shape[0]).to(self.denoiser.device)
            pred_epsilon = self.predict_vector(z, timestep, prompt_emb, use_dc=use_dc)
            if cfg_scale != 1.0:
                pred_null_epsilon = self.predict_vector(z, timestep, null_prompt_emb, use_dc=use_dc)
            else:
                pred_null_epsilon = 0.0

            model_output = (pred_null_epsilon + cfg_scale * (pred_epsilon - pred_null_epsilon))

            z = self.scheduler.step(model_output, t, z, return_dict=False)[0]
            
        # decode
        with torch.no_grad():
            img = self.decode(z.to(self.vae.device))
        return img 
    
    def predict_vector(self, z, t, prompt_emb, use_dc=False):
        v = self.denoiser(z,
                    timestep=t,
                    encoder_hidden_states=prompt_emb,
                    return_dict=False,
                    dc_tokens=use_dc)[0]
        return v 


    def set_noise(self, img_shape:Tuple[int], batch_size:int=1):
        self.all_noise = self.initialize_latent(img_shape, batch_size)
        

    def error(self, latent, nidxs, pidxs, prompts_embs, t, noise=None, use_dc=False):
        nidxs = nidxs.to(self.denoiser.device)
        pidxs = pidxs.to(prompts_embs[0].device)
        t = t.to(self.denoiser.device)
        if noise is None:
            noise = self.all_noise[nidxs]
        prompt_emb =prompts_embs[0][pidxs]

        timestep = t.expand(noise.shape[0]).to(self.denoiser.device)

        zt = self.scheduler.add_noise(latent, noise, t)
        pred_epsilon = self.predict_vector(zt, timestep, prompt_emb, use_dc=use_dc)
        epsilon = noise

        return epsilon, pred_epsilon
    

class StableDiffusionXLBase():
    def __init__(self, model_key:str='stabilityai/stable-diffusion-xl-base-1.0', device='cuda', dtype=torch.float16, use_8bit=False):
        self.device = device

        self.dtype = dtype
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=self.dtype)
        pipe = StableDiffusionXLPipeline.from_pretrained(model_key, vae=vae,
                                                        torch_dtype=self.dtype, variant='fp16', use_safetensors=True)
        self.scheduler = pipe.scheduler

        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.cross_attention_kwargs = None

        self.vae=pipe.vae.to(device)
        if self.vae.config.force_upcast:
            self.vae.to(dtype=torch.float32)

        self.text_enc = pipe.text_encoder.to(device)
        self.text_enc_2 = pipe.text_encoder_2.to(device)
        self.denoiser = pipe.unet.to(device)
        self.denoiser.eval()
        self.denoiser.requires_grad_(False)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.addition_time_embed_dim = self.denoiser.config.addition_time_embed_dim

        self.expected_add_embed_dim = self.denoiser.add_embedding.linear_1.in_features

        del pipe

    @torch.no_grad()
    def encode_prompt(self, prompt: List[str], batch_size:int=1, clip_skip=None) -> List[torch.Tensor]:
        '''
        We assume that
        1. number of tokens < max_length
        2. one prompt for one image
        '''
        device = self.text_enc.device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_enc, self.text_enc_2] if self.text_enc is not None else [self.text_enc]
        )
        prompt_2 = prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])

            prompt_embeds = text_encoder(text_input_ids.to(self.text_enc.device), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            if clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # if self.text_enc_2 is not None:
        #     prompt_embeds = prompt_embeds.to(dtype=self.text_enc_2.dtype, device=self.denoiser.device)
        # else:
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.denoiser.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.dtype, device=self.denoiser.device)
        
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method

        return prompt_embeds, pooled_prompt_embeds


    def initialize_latent(self, img_size:Tuple[int], batch_size:int=1, **kwargs):

        H, W = img_size
        lH, lW = H//self.vae_scale_factor, W//self.vae_scale_factor
        if isinstance(self.denoiser, DistributedDataParallel):
            if isinstance(self.denoiser.module, UNet2DConditionModel):
                lC = self.denoiser.module.config.in_channels
            else:
                lC = self.denoiser.module.base_model.config.in_channels
        else:
            if isinstance(self.denoiser, UNet2DConditionModel):
                lC = self.denoiser.config.in_channels
            else:
                lC = self.denoiser.base_model.config.in_channels
        latent_shape = (batch_size, lC, lH, lW)

        z = torch.randn(latent_shape, device=self.denoiser.device, dtype=self.dtype)

        return z

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if self.vae.config.force_upcast:
            image = image.float()
            if self.vae.dtype == torch.float16:
                self.vae.to(dtype=torch.float32)
        z = self.vae.encode(image).latent_dist.sample()
        z = (z) * self.vae.config.scaling_factor
        z = z.to(self.denoiser.device)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.vae.config.force_upcast:
            z = z.float()
            if self.vae.dtype == torch.float16:
                self.vae.to(dtype=torch.float32)
        z = (z/self.vae.config.scaling_factor)
        return self.vae.decode(z, return_dict=False)[0]
    
    def predict_vector(self, z, t, prompt_emb, pooled_prompt_emb, add_time_emb):
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        v = self.denoiser(z,
                    timestep=t,
                    encoder_hidden_states=prompt_emb,
                    timestep_cond=None,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    return_dict=False)[0]
        return v 


class SDXLEuler(StableDiffusionXLBase):
    def __init__(self, model_key:str='stabilityai/stable-diffusion-xl-base-1.0', device='cuda', use_8bit=True):
        super().__init__(model_key=model_key, device=device, use_8bit=use_8bit)

    @torch.no_grad()
    def inversion(self, src_img, prompts: List[str], NFE:int, cfg_scale: float=1.0, batch_size: int=1):

        # encode text prompts
        prompt_emb, pooled_prompt_emb = self.encode_prompt(prompts, batch_size)
        null_prompt_emb = torch.zeros_like(prompt_emb, device=prompt_emb.device)
        null_pooled_prompt_emb = torch.zeros_like(pooled_prompt_emb, device=pooled_prompt_emb.device)

        # initialize latent
        src_img = src_img.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            z = self.encode(src_img)
            z0 = z.clone()

        # timesteps (default option. You can make your custom here.)
        ddim_scheduler = DDIMInverseScheduler(
            beta_end=self.scheduler.beta_end,
            beta_schedule=self.scheduler.beta_schedule,
            beta_start=self.scheduler.beta_start,
            num_train_timesteps=self.scheduler.num_train_timesteps,
            set_alpha_to_one=self.scheduler.set_alpha_to_one,
            steps_offset=self.scheduler.steps_offset,
            trained_betas=self.scheduler.trained_betas,
            clip_sample=self.scheduler.clip_sample
        )
        ddim_scheduler.set_timesteps(NFE, device=self.device)
        timesteps = ddim_scheduler.timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SDXL DDIM Inversion')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_epsilon = self.predict_vector(z, timestep, prompt_emb)
            if cfg_scale != 1.0:
                pred_null_epsilon = self.predict_vector(z, timestep, null_prompt_emb)
            else:
                pred_null_epsilon = 0

            model_output = (pred_null_epsilon + cfg_scale * (pred_epsilon - pred_null_epsilon))

            z = ddim_scheduler.step(model_output, t, z, return_dict=False)[0]

        return z

    @torch.no_grad()
    def sample(self, prompts: List[str], NFE:int, img_shape: Optional[Tuple[int]]=None, cfg_scale: float=1.0, batch_size: int = 1, latent:Optional[torch.Tensor]=None):
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        original_size = (imgH, imgW)
        target_size = (imgH, imgW)

        add_time_ids = list(original_size + (0,0) + target_size)

        text_encoder_projection_dim = self.text_enc_2.config.projection_dim

        passed_add_embed_dim = (
            self.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.expected_add_embed_dim

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=latent.dtype)

        # encode text prompts
        prompt_emb, pooled_prompt_emb = self.encode_prompt(prompts, batch_size)
        null_prompt_emb = torch.zeros_like(prompt_emb, device=prompt_emb.device)
        null_pooled_prompt_emb = torch.zeros_like(pooled_prompt_emb, device=pooled_prompt_emb.device)

        add_time_ids = add_time_ids.to(prompt_emb.device).repeat(prompt_emb.shape[0], 1)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        # timesteps (default option. You can make your custom here.)
        ddim_scheduler = DDIMScheduler(
            beta_end=self.scheduler.beta_end,
            beta_schedule=self.scheduler.beta_schedule,
            beta_start=self.scheduler.beta_start,
            num_train_timesteps=self.scheduler.num_train_timesteps,
            set_alpha_to_one=self.scheduler.set_alpha_to_one,
            steps_offset=self.scheduler.steps_offset,
            trained_betas=self.scheduler.trained_betas,
            clip_sample=self.scheduler.clip_sample
        )
        ddim_scheduler.set_timesteps(NFE, device=self.device)
        timesteps = ddim_scheduler.timesteps

        # Solve ODE
        pbar = tqdm(timesteps, total=NFE, desc='SDXL DDIM Inversion')
        for i, t in enumerate(pbar):
            timestep = t.expand(z.shape[0]).to(self.device)
            pred_epsilon = self.predict_vector(z, timestep, prompt_emb, pooled_prompt_emb, add_time_ids)
            if cfg_scale != 1.0:
                pred_null_epsilon = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_prompt_emb, add_time_ids)
            else:
                pred_null_epsilon = 0

            model_output = (pred_null_epsilon + cfg_scale * (pred_epsilon - pred_null_epsilon))

            z = ddim_scheduler.step(model_output, t, z, return_dict=False)[0]
        # decode
        with torch.no_grad():
            img = self.decode(z)
        return img 


class SDXLEulerDC(SDXLEuler):
    def __init__(self, model_key:str='stabilityai/stable-diffusion-xl-base-1.0', device='cuda', n_dc_tokens:int=4, use_8bit=True, use_dc_t=True, apply_dc=[True, True, False]):
        super().__init__(model_key=model_key, device=device, use_8bit=use_8bit)

        custom_unet = UNet2DConditionModel_DC3(self.denoiser, n_dc_tokens=n_dc_tokens, use_dc_t=use_dc_t, apply_dc=apply_dc)

        self.denoiser = custom_unet.to(device)
        self.denoiser.requires_grad_(False)
        self.denoiser.eval()

        self.scheduler = DDIMScheduler(
            beta_end=self.scheduler.beta_end,
            beta_schedule=self.scheduler.beta_schedule,
            beta_start=self.scheduler.beta_start,
            num_train_timesteps=self.scheduler.num_train_timesteps,
            set_alpha_to_one=self.scheduler.set_alpha_to_one,
            steps_offset=self.scheduler.steps_offset,
            trained_betas=self.scheduler.trained_betas,
            clip_sample=self.scheduler.clip_sample
        )
    
    def initialize_dc(self, dc_tokens, dc_t_tokens=None):
        self.denoiser.initialize_dc(dc_tokens=dc_tokens, dc_t_tokens=dc_t_tokens)

    @torch.no_grad()
    def sample(self, prompts: List[str], NFE:int, prompt_emb=None, null_prompt_emb=None, img_shape: Optional[Tuple[int]]=None, cfg_scale: float=1.0, batch_size: int = 1, latent:Optional[torch.Tensor]=None, use_dc:bool=False):
        imgH, imgW = img_shape if img_shape is not None else (1024, 1024)

        original_size = (imgH, imgW)
        target_size = (imgH, imgW)

        add_time_ids = list(original_size + (0,0) + target_size)

        # encode text prompts
        if prompt_emb is None:
            prompt_emb, pooled_prompt_emb = self.encode_prompt(prompts, batch_size)
        else:
            prompt_emb, pooled_prompt_emb = prompt_emb

        null_prompt_emb = torch.zeros_like(prompt_emb, device=prompt_emb.device)
        null_pooled_prompt_emb = torch.zeros_like(pooled_prompt_emb, device=pooled_prompt_emb.device)

        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_emb.dtype)
        add_time_ids = add_time_ids.to(prompt_emb.device).repeat(prompt_emb.shape[0], 1)

        # initialize latent
        if latent is None:
            z = self.initialize_latent((imgH, imgW), batch_size)
        else:
            z = latent

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = t.expand(z.shape[0]).to(self.denoiser.device)
            pred_epsilon = self.predict_vector(z, timestep, prompt_emb, pooled_prompt_emb, add_time_ids, use_dc=use_dc)
            if cfg_scale != 1.0:
                pred_null_epsilon = self.predict_vector(z, timestep, null_prompt_emb, null_pooled_prompt_emb, add_time_ids, use_dc=use_dc)
            else:
                pred_null_epsilon = 0.0

            model_output = (pred_null_epsilon + cfg_scale * (pred_epsilon - pred_null_epsilon))

            z = self.scheduler.step(model_output, t, z, return_dict=False)[0]

        # decode
        with torch.no_grad():
            img = self.decode(z.to(self.vae.device))
        return img 
    
    def predict_vector(self, z, t, prompt_emb, pooled_prompt_emb, add_time_ids, use_dc=False):
        added_cond_kwargs = {"text_embeds": pooled_prompt_emb, "time_ids": add_time_ids}
        v = self.denoiser(z,
                    timestep=t,
                    encoder_hidden_states=prompt_emb,
                    timestep_cond=None,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    return_dict=False,
                    dc_tokens=use_dc)[0]
        return v 

    def set_noise(self, img_shape:Tuple[int], batch_size:int=1):
        self.all_noise = self.initialize_latent(img_shape, batch_size)

    def error(self, latent, nidxs, pidxs, prompts_embs, t, noise=None, use_dc=False):

        nidxs = nidxs.to(self.denoiser.device)
        pidxs = pidxs.to(prompts_embs[0].device)
        t = t.to(self.denoiser.device)
        if noise is None:
            noise = self.all_noise[nidxs]
        prompt_emb, pooled_prompt_emb =prompts_embs[0][pidxs], prompts_embs[1][pidxs]
        timestep = t.expand(noise.shape[0]).to(self.denoiser.device)

        latentH, latentW = latent.shape[-2:]

        imgH = latentH * self.vae_scale_factor
        imgW = latentW * self.vae_scale_factor

        original_size = (imgH, imgW)
        target_size = (imgH, imgW)

        add_time_ids = list(original_size + (0,0) + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=latent.dtype)
        add_time_ids = add_time_ids.to(prompt_emb.device).repeat(prompt_emb.shape[0], 1)

        zt = self.scheduler.add_noise(latent, noise, t)
        pred_epsilon = self.predict_vector(zt, timestep, prompt_emb, pooled_prompt_emb, add_time_ids, use_dc=use_dc)
        epsilon = noise

        return epsilon, pred_epsilon