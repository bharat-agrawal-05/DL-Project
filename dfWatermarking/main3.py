import torch
import torchvision
from tqdm import tqdm
import os
import torch.nn.functional as F
from pathlib import Path
import re
import numpy as np # For message embedding
# math import might be needed if you ever combine with other embedders
# import math

# Diffusers and Transformers imports
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

# Accelerate imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration # ProjectConfiguration might not be strictly needed

check_min_version("0.10.0.dev0") # diffusers check
logger = get_logger(__name__)

# ========= Config Class Definition ==========
class Config:
    def __init__(self,
                 prompt: str = "A serene landscape with a hidden message",
                 output_dir: str = "./generated_images_sd_embed_fft", # Changed output dir slightly
                 num_images_per_prompt: int = 1,
                 binary_message: str = "10101010101010101010101010101010", # 32-bit default
                 inference_steps: int = 100,
                 embedding_steps: list[int] = [80], # <<<< CHANGED to list
                 save_intermediate_every_n_steps: int = 20,
                 embedding_strength: float = 0.05,
                 embedding_patch_size: int = 4,
                 embedding_start_x: int = 20,
                 embedding_start_y: int = 20    
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
        self.batch_size_generation = 1 # If > 1, embedding applies to all in batch
        self.inference_steps = inference_steps
        self.guidance_scale = 7.5

        # Intermediate saving
        self.save_intermediate_every_n_steps = save_intermediate_every_n_steps

        # Technical
        self.mixed_precision = "fp16" # "no" or "fp16" or "bf16"
        self.gradient_accumulation_steps = 1

        # Message Embedding Parameters (FFT method)
        self.binary_message = binary_message
        self.embedding_steps = embedding_steps # <<<< CHANGED to list
        self.embedding_strength = embedding_strength # How much to modify frequency coefficients (e.g., 0.0001 to 0.01)
        self.embedding_patch_size = 4    # Size of the square patch for each bit in the frequency domain.
        self.embedding_start_x = 10      # Start X for FFT patch embedding
        self.embedding_start_y = 10      # Start Y for FFT patch embedding

        if len(self.binary_message) != 32:
            raise ValueError("Binary message must be 32 bits long for this configuration.")
        if not self.embedding_steps:
            logger.warning("embedding_steps list is empty. No message embedding will occur.")
        for step in self.embedding_steps:
            if not (0 < step <= self.inference_steps):
                raise ValueError(f"Embedding step {step} must be between 1 and {self.inference_steps}.")
        if self.embedding_patch_size <= 0:
            raise ValueError("embedding_patch_size must be positive.")


# ========= Helper Functions ==========
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
    else: # "no" mixed precision
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

    # Prepare models with Accelerate
    # Note: generator is not a nn.Module, so not prepared typically
    unet, text_encoder, sd_vae, noise_scheduler = accelerator.prepare(
        unet, text_encoder, sd_vae, noise_scheduler
    )

    # Generator should be initialized after prepare if using accelerator.device
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(seed) # Seed the generator on the correct device
    return tokenizer, text_encoder, noise_scheduler, sd_vae, unet, generator, weight_dtype

def setup_accelerator_and_logging(seed: int, gradient_accumulation_steps: int, mixed_precision: str, output_dir: str):
    logging_dir = Path(output_dir, "logs")
    # Using project_dir directly is often simpler unless you use `accelerate config` extensively
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=str(logging_dir)
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        try:
            accelerator.init_trackers(project_name="sd_fft_watermarking")
        except ImportError:
            logger.warning("TensorBoard (or other configured tracker) not found. Logging to trackers will be disabled.")
        except Exception as e:
            logger.warning(f"Could not initialize trackers: {e}")

    set_seed(seed) # Sets torch, numpy, and random seeds
    return accelerator

def _encode_prompt(prompt_batch, tokenizer, text_encoder, device, num_images_per_prompt, do_classifier_free_guidance):
    text_inputs = tokenizer(
        prompt_batch,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    attention_mask = None
    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )[0]

    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    if do_classifier_free_guidance:
        uncond_tokens = [""] * len(prompt_batch)
        max_length = text_input_ids.shape[-1]
        uncond_input = tokenizer(
            uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(device)

        uncond_attention_mask = None
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            uncond_attention_mask = uncond_input.attention_mask.to(device)

        negative_prompt_embeds = text_encoder(
            uncond_input_ids,
            attention_mask=uncond_attention_mask,
        )[0]
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    return prompt_embeds

# ========= Message Embedding Function (FFT based) ==========
def embed_message_in_image_tensor(image_tensor: torch.Tensor, binary_message: str,
                                  patch_size: int, strength: float,
                                  start_x: int = 10, start_y: int = 10):
    """
    Embeds a binary message into an image tensor using FFT.
    Applies to all images in the batch.

    Args:
        image_tensor: (B, C, H, W), expected to be in [0, 1] range.
        binary_message: String of '0's and '1's.
        patch_size: Size of the square patch for each bit in the frequency domain.
        strength: How much to modify frequency coefficients. (e.g., 0.01 for 1% change)
        start_x, start_y: Top-left coordinates to start embedding patches in the frequency domain
                          (relative to the top-left of the *shifted* frequency map).

    Returns modified image_tensor.
    """
    import torch.fft as fft # Keep import local if only used here

    if image_tensor.dim() != 4:
        raise ValueError("Image tensor must be 4D (B, C, H, W)")

    modified_image_batch = image_tensor.clone()
    batch_size, num_channels, img_h_spatial, img_w_spatial = image_tensor.shape
    msg_len = len(binary_message)

    if msg_len == 0:
        return image_tensor # No message, no change

    # Check if there's enough space in the frequency domain for all bits.
    # The frequency domain has the same H, W as the spatial domain.
    # Patches are laid out column by column.
    patches_per_col = (img_h_spatial - start_y) // patch_size
    if patches_per_col <= 0:
        raise ValueError(f"Image height ({img_h_spatial}) or start_y ({start_y}) "
                         f"prevents fitting any patches of size {patch_size}.")

    num_cols_needed = (msg_len + patches_per_col - 1) // patches_per_col
    required_width_in_freq = start_x + (num_cols_needed * patch_size)

    if required_width_in_freq > img_w_spatial:
        available_patches = ((img_w_spatial - start_x) // patch_size) * patches_per_col
        raise ValueError(
            f"Not enough space in frequency domain to embed {msg_len} bits. "
            f"Required width: {required_width_in_freq} > Available width: {img_w_spatial}. "
            f"Calculated available patches: {available_patches}. "
            f"Consider smaller patch_size, smaller start_x/start_y, or larger image."
        )

    for b_idx in range(batch_size):
        for c_idx in range(num_channels):
            channel = modified_image_batch[b_idx, c_idx]

            # Ensure channel is float32 for robust FFT, then convert to complex
            # ComplexHalf (if channel is fp16) can be experimental
            channel_float32 = channel.float()
            channel_complex = torch.complex(channel_float32, torch.zeros_like(channel_float32))
            freq_domain = fft.fft2(channel_complex)
            freq_domain_shifted = fft.fftshift(freq_domain)

            current_x_freq, current_y_freq = start_x, start_y
            freq_h, freq_w = freq_domain_shifted.shape # Should be img_h_spatial, img_w_spatial

            for bit_msg_idx, bit_char in enumerate(binary_message):
                # This check should not be strictly necessary due to upfront validation,
                # but kept as a safeguard during placement loop.
                if current_y_freq + patch_size > freq_h:
                    current_y_freq = start_y
                    current_x_freq += patch_size
                # This should also not be hit due to upfront validation.
                if current_x_freq + patch_size > freq_w:
                    # This should ideally be caught by the initial check
                    logger.error(f"Unexpectedly ran out of space during embedding loop. "
                                   f"Bit {bit_msg_idx}, Pos ({current_x_freq},{current_y_freq}). This indicates an issue with pre-check.")
                    break

                patch_freq = freq_domain_shifted[
                    current_y_freq : current_y_freq + patch_size,
                    current_x_freq : current_x_freq + patch_size
                ]

                if bit_char == '1':
                    modified_patch_freq = patch_freq * (1 + strength)
                else: # bit == '0'
                    modified_patch_freq = patch_freq * (1 - strength)

                freq_domain_shifted[
                    current_y_freq : current_y_freq + patch_size,
                    current_x_freq : current_x_freq + patch_size
                ] = modified_patch_freq

                current_y_freq += patch_size

            # Inverse shift and IFFT
            freq_domain_unshifed = fft.ifftshift(freq_domain_shifted)
            spatial_domain_complex = fft.ifft2(freq_domain_unshifed)
            spatial_domain_real = spatial_domain_complex.real

            # Clamp and assign back, ensuring original dtype is respected if possible
            # (though VAE output is usually float)
            modified_image_batch[b_idx, c_idx] = spatial_domain_real.clamp(0, 1).to(channel.dtype)
            
        # Log after each image in the batch is processed if verbose
        # logger.debug(f"Finished FFT embedding for image {b_idx} in batch.")

    return modified_image_batch


# ========= Image Generation Function (Modified) ==========
def generate_images(
    accelerator: Accelerator,
    all_prompt_embeddings: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    sd_vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    generator: torch.Generator,
    weight_dtype: torch.dtype,
    config: Config,
    run_id: int = 0
):
    safe_prompt_tag = re.sub(r'[^\w_.)( -]', '', config.prompt[:50]).strip().replace(' ', '_')
    image_save_dir = os.path.join(config.output_dir, f"{safe_prompt_tag}_run{run_id}")
    os.makedirs(image_save_dir, exist_ok=True)

    num_channels_latents = unet.config.in_channels
    height = config.image_size
    width = config.image_size

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
            cond_batch_embeds = all_prompt_embeddings[
                total_images_to_generate_for_this_prompt + start_idx_for_this_batch_prompts : \
                total_images_to_generate_for_this_prompt + end_idx_for_this_batch_prompts]
            batch_prompt_embeddings_for_unet = torch.cat([uncond_batch_embeds, cond_batch_embeds], dim=0)
        else:
            batch_prompt_embeddings_for_unet = all_prompt_embeddings[start_idx_for_this_batch_prompts : end_idx_for_this_batch_prompts]


        logger.info(f"\nBatch {batch_idx+1}/{num_unet_batches}: Generating {current_unet_batch_size} images.")
        logger.info(f"  UNet prompt embeddings shape: {batch_prompt_embeddings_for_unet.shape}")

        for i, t in enumerate(tqdm(timesteps, desc=f"Batch {batch_idx+1} Denoising")):
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

            if config.embedding_steps and current_denoising_step in config.embedding_steps:
                logger.info(f"  Embedding message at denoising step {current_denoising_step} (timestep {t.item()}) using FFT method.")
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

                    modified_image_for_embedding = embed_message_in_image_tensor(
                        image_for_embedding,
                        config.binary_message,
                        config.embedding_patch_size,
                        config.embedding_strength,
                        config.embedding_start_x,
                        config.embedding_start_y
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
                    
                    noise_for_current_step = torch.randn_like(modified_x0_latents, device=latents.device, dtype=latents.dtype)
                    
                    # Ensure t_broadcast is correctly shaped for add_noise
                    t_broadcast = t.repeat(modified_x0_latents.shape[0]) if modified_x0_latents.shape[0] > 1 and t.ndim == 0 else t
                    if t_broadcast.ndim == 0: t_broadcast = t_broadcast.unsqueeze(0) # Ensure t is at least 1D

                    latents = noise_scheduler.add_noise(
                        modified_x0_latents.to(latents.device, dtype=latents.dtype),
                        noise_for_current_step,
                        t_broadcast
                    ).to(latents.dtype)
                    logger.info(f"    Message embedded. New latents shape: {latents.shape}")

            latents = noise_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

            if config.save_intermediate_every_n_steps > 0 and \
               (i + 1) % config.save_intermediate_every_n_steps == 0 and \
               (i + 1) < len(timesteps):
                with torch.no_grad():
                    alpha_prod_t_inter = noise_scheduler.alphas_cumprod[t]
                    beta_prod_t_inter = 1 - alpha_prod_t_inter
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
                    embed_tag = "_fft_embedded" if config.embedding_steps else ""
                    steps_str = "_s" + "_and_".join(map(str,sorted(config.embedding_steps))) if config.embedding_steps else ""
                    final_image_name = os.path.join(image_save_dir, f"final_img{global_img_idx}{embed_tag}{steps_str}.png")
                    torchvision.utils.save_image(image_final_save_scaled[img_idx_in_batch].float().cpu(), final_image_name)
                    logger.info(f"Saved final image: {final_image_name}")

        generated_image_count += current_unet_batch_size
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"All {generated_image_count} images generated and saved in {image_save_dir}")


# ========= Main Orchestration Function ==========
def main(config: Config):
    HUGGINGFACE_HUB_TOKEN = None
    accelerator = setup_accelerator_and_logging(config.SEED, config.gradient_accumulation_steps, config.mixed_precision, config.output_dir)

    logger.info("Loading models and tokenizer...")
    tokenizer, text_encoder, noise_scheduler, sd_vae, unet, generator, weight_dtype = \
        load_models_and_tokenizer(accelerator, config.STABLE_DIFFUSION_MODEL_NAME,
                                  config.SD_REVISION, config.SEED, HUGGINGFACE_HUB_TOKEN)

    logger.info(f"Models loaded. UNet dtype: {unet.dtype}, SD VAE dtype: {sd_vae.dtype}, Text Encoder dtype: {text_encoder.dtype}")
    if config.embedding_steps:
        logger.info(f"Embedding a {len(config.binary_message)}-bit message: '{config.binary_message}' at steps {config.embedding_steps} using FFT method.")
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
        generator, weight_dtype, config, run_id=0
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            # Proper way to close trackers
            if accelerator.trackers: # Check if trackers were initialized
                # For multiple trackers, you might need to iterate or use a specific method
                # For tensorboard, usually no explicit close is strictly needed, but good practice for others
                # accelerator.end_training() is more for training loops with optimizers.
                for tracker in accelerator.trackers:
                    if hasattr(tracker, 'finish'): # e.g. for wandb
                        tracker.finish()
                    elif hasattr(tracker, 'close'): # e.g. for tensorboard SummaryWriter
                        tracker.close()
        except Exception as e: logger.warning(f"Error during accelerator cleanup: {e}")
    logger.info("Image generation complete.")

# ========= Entry Point ==========
if __name__ == "__main__":
    config_run = Config(
        num_images_per_prompt=1,
        inference_steps=100,
        embedding_steps=[95, 80], # Example: embed at step 95 and 80 (denoising steps from end)
        embedding_strength=5, # Kept very small for FFT
        embedding_patch_size=4,    # For FFT
        embedding_start_x=20,      # For FFT, try moving away from very center
        embedding_start_y=20,      # For FFT
        save_intermediate_every_n_steps=10
    )
    main(config_run)