import torch
import torchvision
from tqdm import tqdm
import os
import torch.nn.functional as F
from pathlib import Path
import re
import numpy as np # For message embedding
import math

# Diffusers and Transformers imports
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

# Accelerate imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

check_min_version("0.10.0.dev0") # diffusers check
logger = get_logger(__name__)

# ========= Config Class Definition ==========
class Config:
    def __init__(self,
                 prompt: str = "A serene landscape with a hidden message",
                 output_dir: str = "./generated_images_sd_embed",
                 num_images_per_prompt: int = 1,
                 binary_message: str = "10101010101010101010101010101010", # 32-bit default
                 inference_steps: int = 100,
                 embedding_steps: list[int] = [5,50,99], # Changed to list of ints
                 save_intermediate_every_n_steps: int = 20
                ):
        self.SEED = 42
        self.prompt = prompt
        self.output_dir = output_dir
        self.num_images_per_prompt = num_images_per_prompt

        # Stable Diffusion parameters
        self.STABLE_DIFFUSION_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
        self.SD_REVISION = None
        self.sd_vae_scale_factor = 0.18215
        self.image_size = 512

        # Generation parameters
        self.batch_size_generation = 1 # Note: if > 1, embed_message will apply to all in batch
        self.inference_steps = inference_steps
        self.guidance_scale = 7.5

        # Intermediate saving
        self.save_intermediate_every_n_steps = save_intermediate_every_n_steps

        # Technical
        self.mixed_precision = "fp16"
        self.gradient_accumulation_steps = 1

        # Message Embedding Parameters (for token-like pattern method)
        self.binary_message = binary_message
        self.embedding_steps = embedding_steps # Changed
        self.embedding_num_tokens_dim = 8 # e.g., 8x8 grid = 64 possible bit locations
        self.embedding_strength = 0.04

        if len(self.binary_message) != 32:
            raise ValueError("Binary message must be 32 bits long for this configuration.")
        if not self.embedding_steps: # Handle empty list case if necessary
            logger.warning("embedding_steps list is empty. No message embedding will occur.")
        for step in self.embedding_steps:
            if not (0 < step <= self.inference_steps):
                raise ValueError(f"Embedding step {step} must be between 1 and {self.inference_steps}.")


# ========= Helper Functions  ==========
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, HUGGINGFACE_HUB_TOKEN:str=None):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        token=HUGGINGFACE_HUB_TOKEN
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

def load_models_and_tokenizer(accelerator: Accelerator, model_name: str, revision: str, seed: int, HUGGINGFACE_HUB_TOKEN:str=None):
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    logger.info(f"Using weight_dtype: {weight_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, subfolder="tokenizer", revision=revision, use_fast=False, token=HUGGINGFACE_HUB_TOKEN
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(model_name, revision, HUGGINGFACE_HUB_TOKEN=HUGGINGFACE_HUB_TOKEN)
    text_encoder = text_encoder_cls.from_pretrained(
        model_name, subfolder="text_encoder", revision=revision, torch_dtype=weight_dtype, token=HUGGINGFACE_HUB_TOKEN
    )
    sd_vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", revision=revision, torch_dtype=weight_dtype, token=HUGGINGFACE_HUB_TOKEN
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", revision=revision, torch_dtype=weight_dtype, token=HUGGINGFACE_HUB_TOKEN
    )
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler", token=HUGGINGFACE_HUB_TOKEN)

    sd_vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    try:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        logger.info("Using xformers for UNet.")
    except ImportError:
        logger.info("xformers not available. UNet will use default attention.")
    except Exception as e:
        logger.warning(f"Could not enable xformers memory efficient attention: {e}. UNet will use default attention.")

    unet, text_encoder, sd_vae, noise_scheduler = accelerator.prepare(
        unet, text_encoder, sd_vae, noise_scheduler
    )

    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(seed)
    return tokenizer, text_encoder, noise_scheduler, sd_vae, unet, generator, weight_dtype

def setup_accelerator_and_logging(seed: int, gradient_accumulation_steps: int, mixed_precision: str, output_dir: str):
    logging_dir = Path(output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        try:
            accelerator.init_trackers(project_name="stable_diffusion_generation_embed")
        except ImportError:
            logger.warning("TensorBoard (or other configured tracker) not found. Logging to trackers will be disabled.")
        except Exception as e:
            logger.warning(f"Could not initialize trackers: {e}")

    set_seed(seed) # This sets torch, numpy, and random seeds
    return accelerator

def _encode_prompt(prompt_batch, tokenizer, text_encoder, device, num_images_per_prompt, do_classifier_free_guidance):
    text_inputs = tokenizer(
        prompt_batch, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    attention_mask = None
    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)

    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)[0]
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    if do_classifier_free_guidance:
        uncond_tokens = [""] * len(prompt_batch)
        max_length = text_input_ids.shape[-1]
        uncond_input = tokenizer(
            uncond_tokens, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(device)
        uncond_attention_mask = None
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            uncond_attention_mask = uncond_input.attention_mask.to(device)

        negative_prompt_embeds = text_encoder(uncond_input_ids, attention_mask=uncond_attention_mask)[0]
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    return prompt_embeds

# ========= Message Embedding Function (Token-like Patterns) ==========
def embed_message_token_patterns(image_tensor: torch.Tensor, binary_message: str,
                                 num_tokens_dim: int = 8, strength: float = 0.5):
    """
    Embeds a binary message using token-like patterns distributed across ALL images in the batch.
    The permutation of token positions is deterministic if np.random.seed is set beforehand.

    Args:
        image_tensor: (B, C, H, W), expected to be in [0, 1] range. Assumed to be on target device.
        binary_message: String of '0's and '1's.
        num_tokens_dim: Number of token positions in each dimension (e.g., 8 for an 8x8 grid).
        strength: How much to modify pixel values.

    Returns:
        Modified image tensor (on the same device as input).
    """
    if image_tensor.dim() != 4:
        raise ValueError("Image tensor must be 4D (B, C, H, W)")

    device = image_tensor.device
    modified_image_batch = image_tensor.clone()
    batch_size, c, h, w = image_tensor.shape

    msg_len = len(binary_message)
    if msg_len == 0:
        return image_tensor # No message, no change

    token_h_step = h / num_tokens_dim
    token_w_step = w / num_tokens_dim

    positions = []
    for i in range(num_tokens_dim):
        for j in range(num_tokens_dim):
            center_y = int((i + 0.5) * token_h_step)
            center_x = int((j + 0.5) * token_w_step)
            positions.append((center_y, center_x))

    if len(positions) < msg_len:
        error_msg = (f"Message too long ({msg_len} bits) for the number of token positions ({len(positions)}). "
                     f"Required: {msg_len}, Available: {len(positions)} with num_tokens_dim={num_tokens_dim}. "
                     f"Increase num_tokens_dim or shorten the message.")
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Permutation is deterministic due to global seed set by accelerate.set_seed()
    # This ensures the mapping of message bits to locations is consistent.
    permuted_indices = np.random.permutation(len(positions))
    # Select the first msg_len permuted positions. The order of bits in binary_message
    # will correspond to the order of these selected shuffled_positions.
    shuffled_positions = [positions[k] for k in permuted_indices[:msg_len]]

    token_visual_size = max(1, int(min(token_h_step, token_w_step) / 4))

    for b_idx in range(batch_size):
        img_to_modify = modified_image_batch[b_idx] # Shape (C,H,W)

        for bit_idx, bit_val_char in enumerate(binary_message): # No truncation, already checked
            bit_val = int(bit_val_char)
            y_center, x_center = shuffled_positions[bit_idx]

            y_start = max(0, y_center - token_visual_size)
            y_end = min(h, y_center + token_visual_size + 1)
            x_start = max(0, x_center - token_visual_size)
            x_end = min(w, x_center + token_visual_size + 1)

            if y_start >= y_end or x_start >= x_end:
                logger.warning(f"Skipping token for bit {bit_idx} at ({y_center},{x_center}) due to zero area after clipping.")
                continue

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
                try:
                    # Ensure pattern shape matches region if edge cases cause mismatch
                    # (though y_start/end logic for meshgrid should align them)
                    img_to_modify[ch_idx, y_start:y_end, x_start:x_end] = (region + pattern_tensor).clamp(0, 1)
                except RuntimeError as e:
                    logger.error(f"Error applying pattern. Batch {b_idx}, Bit {bit_idx}, Pos ({y_center},{x_center})")
                    logger.error(f"Region shape: {region.shape}, Pattern shape: {pattern_tensor.shape}")
                    logger.error(f"y_start: {y_start}, y_end: {y_end}, x_start: {x_start}, x_end: {x_end}")
                    raise e
        # modified_image_batch[b_idx] = img_to_modify # Not strictly necessary as img_to_modify is a view

    return modified_image_batch


# ========= Image Generation Function (Modified) ==========
def generate_images(
    accelerator: Accelerator, all_prompt_embeddings: torch.Tensor, noise_scheduler: DDPMScheduler,
    sd_vae: AutoencoderKL, unet: UNet2DConditionModel, generator: torch.Generator,
    weight_dtype: torch.dtype, config: Config, run_id: int = 0
):
    safe_prompt_tag = re.sub(r'[^\w_.)( -]', '', config.prompt[:50]).strip().replace(' ', '_')
    image_save_dir = os.path.join(config.output_dir, f"{safe_prompt_tag}_run{run_id}")
    os.makedirs(image_save_dir, exist_ok=True)

    num_channels_latents = unet.config.in_channels
    height, width = config.image_size, config.image_size
    total_images_to_generate_for_this_prompt = config.num_images_per_prompt
    unet_processing_batch_size = config.batch_size_generation
    num_unet_batches = (total_images_to_generate_for_this_prompt + unet_processing_batch_size - 1) // unet_processing_batch_size
    generated_image_count = 0
    do_classifier_free_guidance = config.guidance_scale > 1.0

    for batch_idx in range(num_unet_batches):
        current_unet_batch_size = min(
            unet_processing_batch_size,
            total_images_to_generate_for_this_prompt - (batch_idx * unet_processing_batch_size)
        )
        if current_unet_batch_size <= 0: continue

        latents_shape = (current_unet_batch_size, num_channels_latents, height // 8, width // 8)
        initial_latents_for_batch = torch.randn(
            latents_shape, generator=generator, device=accelerator.device, dtype=weight_dtype
        )
        latents = initial_latents_for_batch * noise_scheduler.init_noise_sigma

        noise_scheduler.set_timesteps(config.inference_steps, device=accelerator.device)
        timesteps = noise_scheduler.timesteps

        start_idx_for_this_batch_prompts = batch_idx * unet_processing_batch_size
        end_idx_for_this_batch_prompts = start_idx_for_this_batch_prompts + current_unet_batch_size

        if do_classifier_free_guidance:
            uncond_batch_embeds = all_prompt_embeddings[start_idx_for_this_batch_prompts : end_idx_for_this_batch_prompts]
            cond_offset = total_images_to_generate_for_this_prompt
            cond_batch_embeds = all_prompt_embeddings[
                cond_offset + start_idx_for_this_batch_prompts : \
                cond_offset + end_idx_for_this_batch_prompts
            ]
            batch_prompt_embeddings_for_unet = torch.cat([uncond_batch_embeds, cond_batch_embeds], dim=0)
        else:
            batch_prompt_embeddings_for_unet = all_prompt_embeddings[start_idx_for_this_batch_prompts : end_idx_for_this_batch_prompts]

        logger.info(f"\nBatch {batch_idx+1}/{num_unet_batches}: Generating {current_unet_batch_size} images.")
        logger.info(f"  UNet prompt embeddings shape: {batch_prompt_embeddings_for_unet.shape}")

        for i, t in enumerate(tqdm(timesteps, desc=f"Batch {batch_idx+1} Denoising")):
            # current_denoising_step is 1-indexed from total_steps down to 1
            # Timestep t is 0-indexed from scheduler.timesteps
            # The number of steps remaining, or current "logical" step number
            current_denoising_step = config.inference_steps - i


            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred_out = unet(latent_model_input, t, encoder_hidden_states=batch_prompt_embeddings_for_unet).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_out

            # Check if current_denoising_step is in the list of embedding_steps
            if config.embedding_steps and current_denoising_step in config.embedding_steps:
                logger.info(f"  Embedding message at denoising step {current_denoising_step} (timestep {t.item()}) using token patterns.")
                with torch.no_grad():
                    alpha_prod_t = noise_scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    pred_original_sample = (latents - beta_prod_t**(0.5) * noise_pred) / alpha_prod_t**(0.5)
                    image_latents_for_decode = (pred_original_sample / config.sd_vae_scale_factor).to(sd_vae.device, dtype=sd_vae.dtype)
                    decoded_x0_image = sd_vae.decode(image_latents_for_decode).sample
                    image_for_embedding = (decoded_x0_image / 2 + 0.5).clamp(0, 1)

                    if accelerator.is_main_process:
                        for k_img in range(image_for_embedding.shape[0]):
                            img_global_idx = generated_image_count + k_img
                            pre_embed_path = os.path.join(image_save_dir, f"pre_embed_s{current_denoising_step}_b{batch_idx}_img{img_global_idx}.png")
                            torchvision.utils.save_image(image_for_embedding[k_img].float().cpu(), pre_embed_path)

                    modified_image_for_embedding = embed_message_token_patterns(
                        image_for_embedding,
                        config.binary_message,
                        num_tokens_dim=config.embedding_num_tokens_dim,
                        strength=config.embedding_strength
                    )

                    if accelerator.is_main_process:
                        for k_img in range(modified_image_for_embedding.shape[0]):
                            img_global_idx = generated_image_count + k_img
                            post_embed_path = os.path.join(image_save_dir, f"post_embed_s{current_denoising_step}_b{batch_idx}_img{img_global_idx}.png")
                            torchvision.utils.save_image(modified_image_for_embedding[k_img].float().cpu(), post_embed_path)


                    modified_image_scaled_back = (modified_image_for_embedding * 2 - 1).to(sd_vae.device, dtype=sd_vae.dtype)
                    modified_latents_dist = sd_vae.encode(modified_image_scaled_back).latent_dist
                    modified_x0_latents = modified_latents_dist.sample()
                    modified_x0_latents = modified_x0_latents * config.sd_vae_scale_factor
                    
                    # Ensure noise is on the same device as latents
                    noise_for_current_step = torch.randn_like(modified_x0_latents, device=latents.device, dtype=latents.dtype)
                    
                    # Add noise back to the modified x0 to get latents for the current timestep t
                    # t needs to be compatible with add_noise, usually a tensor of timesteps
                    # If latents is (B,C,H,W), t_broadcast should be (B,)
                    t_broadcast = t.repeat(modified_x0_latents.shape[0]) if modified_x0_latents.shape[0] > 1 else t.unsqueeze(0)

                    latents = noise_scheduler.add_noise(
                        modified_x0_latents.to(latents.device, dtype=latents.dtype),
                        noise_for_current_step,
                        t_broadcast # Use broadcasted t
                    ).to(latents.dtype)
                    logger.info(f"    Message embedded. New latents shape: {latents.shape}")

            latents = noise_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

            if config.save_intermediate_every_n_steps > 0 and \
               (i + 1) % config.save_intermediate_every_n_steps == 0 and \
               (i + 1) < len(timesteps): # Don't save on last step as it's the final image
                with torch.no_grad():
                    # Estimate x0 for saving intermediate image
                    alpha_prod_t_inter = noise_scheduler.alphas_cumprod[t]
                    beta_prod_t_inter = 1 - alpha_prod_t_inter
                    # Use current latents (which are x_t) and noise_pred (estimate of noise in x_t)
                    pred_original_sample_inter = (latents - beta_prod_t_inter**(0.5) * noise_pred) / alpha_prod_t_inter**(0.5)
                    
                    image_latents_for_decode_inter = (pred_original_sample_inter / config.sd_vae_scale_factor).to(sd_vae.device, dtype=sd_vae.dtype)
                    image_decoded_inter = sd_vae.decode(image_latents_for_decode_inter).sample
                    image_to_save_inter = (image_decoded_inter / 2 + 0.5).clamp(0, 1)

                    if accelerator.is_main_process:
                        for img_idx_in_batch in range(image_to_save_inter.shape[0]):
                            global_img_idx = generated_image_count + img_idx_in_batch
                            intermediate_image_path = os.path.join(image_save_dir,
                                f"intermediate_s{current_denoising_step}_b{batch_idx}_img{global_img_idx}.png")
                            torchvision.utils.save_image(image_to_save_inter[img_idx_in_batch].float().cpu(), intermediate_image_path)

        with torch.no_grad():
            final_latents_for_decode = latents / config.sd_vae_scale_factor
            image_final_decoded = sd_vae.decode(final_latents_for_decode.to(sd_vae.device, dtype=sd_vae.dtype)).sample
            image_final_save_scaled = (image_final_decoded / 2 + 0.5).clamp(0, 1)

            if accelerator.is_main_process:
                for img_idx_in_batch in range(image_final_save_scaled.shape[0]):
                    global_img_idx = generated_image_count + img_idx_in_batch
                    # Modify filename to indicate embedding happened if steps were provided
                    embed_tag = "_embedded" if config.embedding_steps else ""
                    # Optionally include actual steps: f"_embedded_at_s{'and'.join(map(str, config.embedding_steps))}"
                    final_image_name = os.path.join(image_save_dir, f"final_img{global_img_idx}{embed_tag}_token_patterns.png")
                    torchvision.utils.save_image(image_final_save_scaled[img_idx_in_batch].float().cpu(), final_image_name)
                    logger.info(f"Saved final image: {final_image_name}")

        generated_image_count += current_unet_batch_size
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"All {generated_image_count} images generated and saved in {image_save_dir}")

# ========= Main Orchestration Function ==========
def main(config: Config):
    HUGGINGFACE_HUB_TOKEN = None # Set if you need to download from a private repo
    accelerator = setup_accelerator_and_logging(config.SEED, config.gradient_accumulation_steps, config.mixed_precision, config.output_dir)

    logger.info("Loading models and tokenizer...")
    tokenizer, text_encoder, noise_scheduler, sd_vae, unet, generator, weight_dtype = \
        load_models_and_tokenizer(accelerator, config.STABLE_DIFFUSION_MODEL_NAME,
                                  config.SD_REVISION, config.SEED, HUGGINGFACE_HUB_TOKEN)

    logger.info(f"Models loaded. UNet dtype: {unet.dtype}, SD VAE dtype: {sd_vae.dtype}, Text Encoder dtype: {text_encoder.dtype}")
    if config.embedding_steps:
        logger.info(f"Embedding a 32-bit message: '{config.binary_message}' at steps {config.embedding_steps} using token-like patterns.")
    else:
        logger.info("No embedding steps defined. Proceeding with standard image generation.")


    unet.eval()
    text_encoder.eval()
    sd_vae.eval()

    do_classifier_free_guidance = config.guidance_scale > 1.0
    prompt_batch_for_encoder = [config.prompt]

    all_prompt_embeddings = _encode_prompt(
        prompt_batch_for_encoder, tokenizer, text_encoder, accelerator.device,
        config.num_images_per_prompt,
        do_classifier_free_guidance
    )

    logger.info(f"Total prompt embeddings generated (shape): {all_prompt_embeddings.shape}")

    generate_images(
        accelerator, all_prompt_embeddings, noise_scheduler, sd_vae, unet,
        generator, weight_dtype, config, run_id=0 # You might want to make run_id configurable or increment it
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            if hasattr(accelerator, 'end_training'): # In recent versions, end_training is for training loops
                accelerator.end_training()
            # For simple scripts, trackers might be closed automatically or use a direct call if available
            # For example, if using wandb, wandb.finish()
        except Exception as e: logger.warning(f"Error during accelerator cleanup: {e}")
    logger.info("Image generation complete.")

# ========= Entry Point ==========
if __name__ == "__main__":
    config_run = Config(
        num_images_per_prompt=1,
        inference_steps=100,
        embedding_steps=[5,20], # Example: embed at step 80 and 99 (steps from end)
        save_intermediate_every_n_steps=10,
        # binary_message can be overridden here if needed
        # embedding_num_tokens_dim default is 8 (8x8=64 tokens), sufficient for 32 bits.
    )
    main(config_run)