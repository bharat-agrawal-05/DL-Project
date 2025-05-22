# generate_dataset.py
import os
# Try to suppress TensorFlow CUDA/GPU initialization if it's causing issues
# and not directly needed by this PyTorch-centric script.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 = all messages, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

import torch
import torchvision
from tqdm import tqdm # Still used for the main message loop
import torch.nn.functional as F
from pathlib import Path
import re
import numpy as np
import math
import argparse # For command-line arguments
import random

# Diffusers and Transformers imports
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

# Accelerate imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

check_min_version("0.10.0.dev0") # diffusers check
logger = get_logger(__name__) # Use Accelerate's logger

# ========= Config Class Definition ==========
class Config:
    def __init__(self,
                 prompt: str = "A serene landscape with a hidden message",
                 output_dir: str = "./TokenPatternsDataset",
                 num_images_per_prompt: int = 1,
                 binary_message: str = "10101010101010101010101010101010", # This is a string
                 inference_steps: int = 50,
                 embedding_steps: list[int] = [40, 20],
                 save_intermediate_every_n_steps: int = 0,
                 embedding_num_tokens_dim: int = 8,
                 embedding_strength: float = 0.08,
                 STABLE_DIFFUSION_MODEL_NAME: str = "runwayml/stable-diffusion-v1-5"
                ):
        self.prompt = prompt
        self.output_dir = output_dir
        self.num_images_per_prompt = num_images_per_prompt

        self.STABLE_DIFFUSION_MODEL_NAME = STABLE_DIFFUSION_MODEL_NAME
        self.SD_REVISION = None
        self.sd_vae_scale_factor = 0.18215
        self.image_size = 512

        self.batch_size_generation = 1
        self.inference_steps = inference_steps
        self.guidance_scale = 7.5

        self.save_intermediate_every_n_steps = save_intermediate_every_n_steps

        self.mixed_precision = "fp16"
        self.gradient_accumulation_steps = 1

        self.binary_message = binary_message # Expects a string
        self.embedding_steps = embedding_steps
        self.embedding_num_tokens_dim = embedding_num_tokens_dim
        self.embedding_strength = embedding_strength

        if len(self.binary_message) != 32:
            pass
        if not self.embedding_steps:
            logger.warning("embedding_steps list is empty. No message embedding will occur.")
        for step in self.embedding_steps:
            if not (0 < step <= self.inference_steps):
                raise ValueError(f"Embedding step {step} must be between 1 and {self.inference_steps}.")

# Helper Functions (assumed correct and unchanged from previous version)
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, HUGGINGFACE_HUB_TOKEN:str=None):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        token=HUGGINGFACE_HUB_TOKEN
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel": return CLIPTextModel
    raise ValueError(f"Unsupported model class: {model_class}")

def load_models_and_tokenizer(accelerator: Accelerator, model_name: str, revision: str, HUGGINGFACE_HUB_TOKEN:str=None):
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else \
                   torch.bfloat16 if accelerator.mixed_precision == "bf16" else \
                   torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer", revision=revision, use_fast=False, token=HUGGINGFACE_HUB_TOKEN)
    text_encoder_cls = import_model_class_from_model_name_or_path(model_name, revision, HUGGINGFACE_HUB_TOKEN=HUGGINGFACE_HUB_TOKEN)
    text_encoder = text_encoder_cls.from_pretrained(model_name, subfolder="text_encoder", revision=revision, torch_dtype=weight_dtype, token=HUGGINGFACE_HUB_TOKEN)
    sd_vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", revision=revision, torch_dtype=weight_dtype, token=HUGGINGFACE_HUB_TOKEN)
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", revision=revision, torch_dtype=weight_dtype, token=HUGGINGFACE_HUB_TOKEN)
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler", token=HUGGINGFACE_HUB_TOKEN)

    sd_vae.requires_grad_(False); text_encoder.requires_grad_(False); unet.requires_grad_(False)
    try:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        logger.info("Using xformers if available and enabled.")
    except ImportError:
        logger.info("xformers not available or not enabled.")
    except Exception as e:
        logger.warning(f"Could not enable xformers: {e}")

    unet, text_encoder, sd_vae, noise_scheduler = accelerator.prepare(unet, text_encoder, sd_vae, noise_scheduler)
    return tokenizer, text_encoder, noise_scheduler, sd_vae, unet, weight_dtype

def setup_accelerator_for_dataset_gen(mixed_precision: str):
    accelerator = Accelerator(mixed_precision=mixed_precision)
    logger.info(f"Accelerator initialized with mixed_precision: {mixed_precision}, device: {accelerator.device}")
    return accelerator

def _encode_prompt(prompt_batch, tokenizer, text_encoder, device, num_images_per_prompt, do_classifier_free_guidance):
    text_inputs = tokenizer(prompt_batch, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device) if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask else None
    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)[0]
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    if do_classifier_free_guidance:
        uncond_tokens = [""] * len(prompt_batch)
        max_length = text_input_ids.shape[-1]
        uncond_input = tokenizer(uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        uncond_input_ids = uncond_input.input_ids.to(device)
        uncond_attention_mask = uncond_input.attention_mask.to(device) if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask else None
        negative_prompt_embeds = text_encoder(uncond_input_ids, attention_mask=uncond_attention_mask)[0]
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    return prompt_embeds

def embed_message_token_patterns(image_tensor: torch.Tensor, binary_message_str: str,
                                 positions_for_msg: list,
                                 num_tokens_dim: int,
                                 h_img: int, w_img: int,
                                 strength: float = 0.5):
    if image_tensor.dim() != 4: raise ValueError("Image tensor must be 4D (B, C, H, W)")
    device = image_tensor.device
    modified_image_batch = image_tensor.clone()
    batch_size, c, _, _ = image_tensor.shape
    msg_len = len(binary_message_str)
    if msg_len == 0: return image_tensor

    token_h_step = h_img / num_tokens_dim
    token_w_step = w_img / num_tokens_dim
    token_visual_size = max(1, int(min(token_h_step, token_w_step) / 4))

    if len(positions_for_msg) < msg_len:
        raise ValueError(f"Number of provided positions ({len(positions_for_msg)}) is less than message length ({msg_len}).")

    for b_idx in range(batch_size):
        img_to_modify = modified_image_batch[b_idx]
        for bit_idx, bit_val_char in enumerate(binary_message_str):
            if bit_idx >= len(positions_for_msg):
                logger.warning(f"Message bit index {bit_idx} exceeds available positions. Stopping embedding for this message.")
                break
            bit_val = int(bit_val_char)
            y_center, x_center = positions_for_msg[bit_idx]

            y_start = max(0, y_center - token_visual_size)
            y_end = min(h_img, y_center + token_visual_size + 1)
            x_start = max(0, x_center - token_visual_size)
            x_end = min(w_img, x_center + token_visual_size + 1)

            if y_start >= y_end or x_start >= x_end: continue

            y_grid_rel, x_grid_rel = torch.meshgrid(
                torch.arange(y_start, y_end, device=device).float() - y_center,
                torch.arange(x_start, x_end, device=device).float() - x_center,
                indexing='ij'
            )
            dist_norm = torch.sqrt(y_grid_rel**2 + x_grid_rel**2) / (token_visual_size + 1e-8)
            pattern_tensor = torch.zeros_like(dist_norm)
            if bit_val == 1:
                pattern_tensor = torch.cos(dist_norm * math.pi * 2) * torch.clamp(1 - dist_norm, 0, 1)
            else:
                angle = torch.atan2(y_grid_rel, x_grid_rel)
                pattern_tensor = torch.cos(angle * 4) * torch.clamp(1 - dist_norm, 0, 1)
            pattern_tensor = pattern_tensor * strength
            for ch_idx in range(c):
                region = img_to_modify[ch_idx, y_start:y_end, x_start:x_end]
                img_to_modify[ch_idx, y_start:y_end, x_start:x_end] = (region + pattern_tensor).clamp(0, 1)
    return modified_image_batch

def generate_single_image_with_embedding(
    accelerator: Accelerator, prompt_str: str, binary_msg_str: str,
    tokenizer, text_encoder, noise_scheduler, sd_vae, unet, weight_dtype,
    config_obj: Config, image_seed: int, output_path: Path,
    globally_fixed_positions_for_msg: list
):
    set_seed(image_seed)
    generator = torch.Generator(device=accelerator.device).manual_seed(image_seed)

    prompt_batch_for_encoder = [prompt_str]
    prompt_embeddings = _encode_prompt(
        prompt_batch_for_encoder, tokenizer, text_encoder, accelerator.device,
        1, config_obj.guidance_scale > 1.0
    )
    num_channels_latents = unet.config.in_channels
    height, width = config_obj.image_size, config_obj.image_size
    latents_shape = (1, num_channels_latents, height // 8, width // 8)
    initial_latents = torch.randn(latents_shape, generator=generator, device=accelerator.device, dtype=weight_dtype)
    latents = initial_latents * noise_scheduler.init_noise_sigma
    noise_scheduler.set_timesteps(config_obj.inference_steps, device=accelerator.device)
    timesteps = noise_scheduler.timesteps

    # Removed tqdm wrapper from here
    # pbar_desc = f"Img (seed {image_seed}, msg {binary_msg_str[:6]}...)"
    # pbar = tqdm(timesteps, desc=pbar_desc, leave=False, disable=not accelerator.is_local_main_process)
    # for i, t in enumerate(pbar):
    for i, t in enumerate(timesteps): # Iterate directly over timesteps
        current_denoising_step = config_obj.inference_steps - i
        latent_model_input = torch.cat([latents] * 2) if config_obj.guidance_scale > 1.0 else latents
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred_out = unet(latent_model_input, t, encoder_hidden_states=prompt_embeddings).sample

        if config_obj.guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
            noise_pred = noise_pred_uncond + config_obj.guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = noise_pred_out

        if config_obj.embedding_steps and current_denoising_step in config_obj.embedding_steps:
            # Optional: Log embedding action if desired, but without tqdm postfix
            if accelerator.is_local_main_process and i % 10 == 0 : # Log less frequently
                 logger.debug(f"Img (seed {image_seed}): Embedding at step {current_denoising_step}")
            with torch.no_grad():
                alpha_prod_t = noise_scheduler.alphas_cumprod[t] if t >=0 else noise_scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latents - beta_prod_t**(0.5) * noise_pred) / alpha_prod_t**(0.5)
                image_latents_for_decode = (pred_original_sample / config_obj.sd_vae_scale_factor).to(sd_vae.device, dtype=sd_vae.dtype)
                decoded_x0_image = sd_vae.decode(image_latents_for_decode).sample
                image_for_embedding = (decoded_x0_image / 2 + 0.5).clamp(0, 1)

                modified_image_for_embedding = embed_message_token_patterns(
                    image_for_embedding, binary_msg_str,
                    positions_for_msg=globally_fixed_positions_for_msg,
                    num_tokens_dim=config_obj.embedding_num_tokens_dim,
                    h_img=config_obj.image_size,
                    w_img=config_obj.image_size,
                    strength=config_obj.embedding_strength
                )
                modified_image_scaled_back = (modified_image_for_embedding * 2 - 1).to(sd_vae.device, dtype=sd_vae.dtype)
                modified_latents_dist = sd_vae.encode(modified_image_scaled_back).latent_dist
                modified_x0_latents = modified_latents_dist.sample() * config_obj.sd_vae_scale_factor

                current_noise = torch.randn_like(modified_x0_latents, device=latents.device, dtype=latents.dtype)

                t_b = t
                if t_b.ndim == 0: t_b = t_b.unsqueeze(0)
                if t_b.shape[0] != modified_x0_latents.shape[0] and modified_x0_latents.shape[0] == 1:
                    if t_b.numel() > 1 and modified_x0_latents.shape[0] == 1:
                        t_b = t_b[0].unsqueeze(0)
                elif t_b.shape[0] != modified_x0_latents.shape[0]:
                     t_b = t_b[0].repeat(modified_x0_latents.shape[0])

                latents = noise_scheduler.add_noise(
                    modified_x0_latents.to(latents.device, dtype=latents.dtype), current_noise, t_b
                ).to(latents.dtype)
        latents = noise_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

    with torch.no_grad():
        final_latents_for_decode = latents / config_obj.sd_vae_scale_factor
        image_final_decoded = sd_vae.decode(final_latents_for_decode.to(sd_vae.device, dtype=sd_vae.dtype)).sample
        image_final_save_scaled = (image_final_decoded / 2 + 0.5).clamp(0, 1)

    if accelerator.is_main_process:
        torchvision.utils.save_image(image_final_save_scaled[0].float().cpu(), str(output_path))
        logger.info(f"Saved image: {output_path} (Seed: {image_seed}, Msg: {binary_msg_str[:6]}...)") # Log on save

def main_dataset_generation(args):
    accelerator = setup_accelerator_for_dataset_gen(
        mixed_precision="fp16"
    )
    set_seed(args.global_seed)
    logger.info(f"Global seed for dataset generation set to: {args.global_seed}.")

    base_output_dir = Path(args.dataset_dir)
    binary_messages_path = base_output_dir / "tensorsBinary.pt"
    prompts_path = base_output_dir / "prompts.txt"

    if not binary_messages_path.exists():
        logger.error(f"Binary messages file not found: {binary_messages_path}. Run generate_binary_tensors.py first.")
        return
    if not prompts_path.exists():
        logger.error(f"Prompts file not found: {prompts_path}. Run generate_sample_prompts.py or create it.")
        return

    all_binary_messages_str = torch.load(binary_messages_path)
    if not isinstance(all_binary_messages_str, list) or \
       (all_binary_messages_str and not all(isinstance(msg, str) for msg in all_binary_messages_str)):
        logger.error(f"Expected {binary_messages_path} to contain a list of strings. "
                     f"Got type: {type(all_binary_messages_str)}")
        if isinstance(all_binary_messages_str, list) and all_binary_messages_str:
            logger.error(f"First element type: {type(all_binary_messages_str[0])}")
        return

    with open(prompts_path, "r") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    if not all_prompts:
        logger.error("No prompts loaded from prompts.txt. Exiting.")
        return
    if not all_binary_messages_str:
        logger.error("No valid binary messages loaded from tensorsBinary.pt. Exiting.")
        return

    logger.info(f"Loaded {len(all_binary_messages_str)} binary messages and {len(all_prompts)} prompts.")

    default_cfg_for_model_load = Config(
        inference_steps=args.inference_steps,
        embedding_steps=args.embedding_steps,
        embedding_num_tokens_dim=args.embedding_num_tokens_dim
    )

    logger.info("Loading diffusion models...")
    tokenizer, text_encoder, noise_scheduler, sd_vae, unet, weight_dtype = \
        load_models_and_tokenizer(accelerator, default_cfg_for_model_load.STABLE_DIFFUSION_MODEL_NAME,
                                  default_cfg_for_model_load.SD_REVISION,
                                  HUGGINGFACE_HUB_TOKEN=None)
    logger.info("Models loaded.")

    num_tokens_dim_arg = args.embedding_num_tokens_dim
    img_size_for_pos = default_cfg_for_model_load.image_size
    msg_len = 32

    token_h_step_pos = img_size_for_pos / num_tokens_dim_arg
    token_w_step_pos = img_size_for_pos / num_tokens_dim_arg
    base_positions = []
    for i_pos_calc in range(num_tokens_dim_arg):
        for j_pos_calc in range(num_tokens_dim_arg):
            center_y = int((i_pos_calc + 0.5) * token_h_step_pos)
            center_x = int((j_pos_calc + 0.5) * token_w_step_pos)
            base_positions.append((center_y, center_x))

    if len(base_positions) < msg_len:
        raise ValueError(f"Message length ({msg_len}) too long for available token positions ({len(base_positions)}) "
                         f"with num_tokens_dim={num_tokens_dim_arg}. Increase embedding_num_tokens_dim.")

    globally_fixed_positions_for_msg = base_positions[:msg_len]
    logger.info(f"Pre-calculated globally fixed positions for {msg_len} bits using {num_tokens_dim_arg}x{num_tokens_dim_arg} grid.")

    prompt_idx_counter = 0
    # Main progress bar for messages
    main_pbar = tqdm(all_binary_messages_str, desc="Total Messages Progress", 
                     disable=not accelerator.is_local_main_process, position=0) # position=0 for outer loop

    for msg_idx, binary_msg_str_current in enumerate(main_pbar):
        if len(binary_msg_str_current) != msg_len:
            logger.warning(f"Skipping message index {msg_idx} due to incorrect length ({len(binary_msg_str_current)} instead of {msg_len}). Message preview: {binary_msg_str_current[:10]}...")
            continue

        subdir_name = f"index{msg_idx}"
        subdir_path = base_output_dir / subdir_name
        if accelerator.is_main_process:
             subdir_path.mkdir(parents=True, exist_ok=True)
        
        # Optional: Progress bar for images within a message subdir
        # image_pbar_desc = f"Msg {msg_idx} Images"
        # image_pbar = tqdm(range(args.images_per_subdir), desc=image_pbar_desc, 
        #                   disable=not accelerator.is_local_main_process, position=1, leave=False)


        for i in range(args.images_per_subdir): # Use image_pbar if enabled above
            # if accelerator.is_local_main_process: # Update inner pbar description if used
            #    image_pbar.set_description(f"Msg {msg_idx} Img {i+1}/{args.images_per_subdir}")

            current_prompt = all_prompts[prompt_idx_counter % len(all_prompts)]
            prompt_idx_counter += 1
            image_specific_seed = args.global_seed + (msg_idx * args.images_per_subdir * 1000) + i + 1 # Increased multiplier for seed uniqueness

            cfg_iter = Config(
                prompt=current_prompt,
                binary_message=binary_msg_str_current,
                embedding_steps=args.embedding_steps,
                inference_steps=args.inference_steps,
                embedding_strength=args.embedding_strength,
                embedding_num_tokens_dim=num_tokens_dim_arg,
                STABLE_DIFFUSION_MODEL_NAME=default_cfg_for_model_load.STABLE_DIFFUSION_MODEL_NAME
            )

            image_filename = f"image_{i}_seed{image_specific_seed}_msgidx{msg_idx}.png"
            output_image_path = subdir_path / image_filename

            if output_image_path.exists() and not args.overwrite:
                if accelerator.is_local_main_process:
                    # Log less frequently if many images are skipped
                    if i % (args.images_per_subdir // 10 + 1) == 0 : # Log for ~10 skipped images
                         logger.debug(f"Skipping existing image: {output_image_path}")
                continue

            if accelerator.is_local_main_process:
                generate_single_image_with_embedding(
                    accelerator, current_prompt, binary_msg_str_current,
                    tokenizer, text_encoder, noise_scheduler, sd_vae, unet, weight_dtype,
                    cfg_iter, image_specific_seed, output_image_path,
                    globally_fixed_positions_for_msg
                )
        
        # if accelerator.is_local_main_process and 'image_pbar' in locals(): # Close inner pbar if used
        #    image_pbar.close()

        accelerator.wait_for_everyone()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # if accelerator.is_local_main_process and 'main_pbar' in locals(): # Close outer pbar
    #    main_pbar.close()


    logger.info("Dataset generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset of images with embedded binary messages using token patterns.")
    parser.add_argument("--dataset_dir", type=str, default="./TokenPatternsDataset", help="Base directory for the dataset output and where input files (prompts.txt, tensorsBinary.pt) are expected.")
    parser.add_argument("--images_per_subdir", type=int, default=2, help="Number of images per binary message.")
    parser.add_argument("--embedding_steps", type=int, nargs='+', default=[25, 5], help="List of denoising steps (remaining) at which to embed.")
    parser.add_argument("--inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--global_seed", type=int, default=12345, help="Global seed used to derive per-image seeds.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing images.")
    parser.add_argument("--embedding_strength", type=float, default=0.08, help="Strength of the token pattern embedding.")
    parser.add_argument("--embedding_num_tokens_dim", type=int, default=8, help="Grid dimension for token placement (e.g., 8 for 8x8).")

    args = parser.parse_args()

    if args.embedding_steps:
        for step in args.embedding_steps:
            if not (0 < step <= args.inference_steps):
                print(f"Error: Embedding step {step} is out of range for --inference_steps {args.inference_steps}.")
                exit(1)

    if args.embedding_num_tokens_dim ** 2 < 32:
         print(f"Error: embedding_num_tokens_dim ({args.embedding_num_tokens_dim}) is too small for a 32-bit message. "
               f"Need at least sqrt(32) approx 6 (for 6x6=36). Current: {args.embedding_num_tokens_dim**2} positions.")
         exit(1)

    main_dataset_generation(args)