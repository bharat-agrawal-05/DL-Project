import torch
import torchvision
from tqdm import tqdm
import os
import torch.nn.functional as F
from pathlib import Path
import re
import numpy as np # For message embedding

# Diffusers and Transformers imports
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

# Accelerate importsrene landscape with a hidden message
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration

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
                 embedding_step: int = 80, 
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
        self.batch_size_generation = 1
        self.inference_steps = inference_steps
        self.guidance_scale = 7.5

        # Intermediate saving
        self.save_intermediate_every_n_steps = save_intermediate_every_n_steps

        # Technical
        self.mixed_precision = "fp16"
        self.gradient_accumulation_steps = 1

        # Message Embedding Parameters (for patch-based method)
        self.binary_message = binary_message
        self.embedding_step = embedding_step 
        self.embedding_strength = 0.2 # For patch-based or global methods
        self.embedding_patch_size = 4 # For patch-based method

        # Parameters for Channel Bias method (if you choose to use it)
        self.channel_bias_strength_per_bit = 0.02
        self.channel_bias_num_bits_per_channel = 10 

        # Parameters for DCT/FFT method (if you choose to use it)
        self.dct_strength = 0.005
        self.dct_num_coeffs_to_mod = 32
        self.dct_start_coeff_idx = 10


        if len(self.binary_message) != 32:
            raise ValueError("Binary message must be 32 bits long.")


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
        pt_version_major_minor = ".".join(torch.__version__.split(".")[:2])
        if pt_version_major_minor >= "2.0":
             unet.enable_xformers_memory_efficient_attention()
             logger.info("Using xformers for UNet.")
        else:
            logger.info("PyTorch version might not be fully compatible with xformers build. Skipping xformers enabling.")
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
    # project_config to avoid the warning if tensorboard is not installed
    # If using Accelerate > 0.18.0, ProjectConfiguration is good.
    # For older versions, this might not be necessary or could cause issues if not recognized.
    # Let's remove it for broader compatibility if it was causing issues.
    # project_config = ProjectConfiguration(project_dir=logging_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        # project_configuration=project_config # Commented out if it was an issue
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        try:
            accelerator.init_trackers("stable_diffusion_generation")
        except ImportError:
            logger.warning("TensorBoard (or other configured tracker) not found. Logging to trackers will be disabled.")
        except Exception as e: # Catch other potential errors during init_trackers
            logger.warning(f"Could not initialize trackers: {e}")

    set_seed(seed)
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

# ========= Message Embedding Functions ==========

# This is the patch-based method you were likely using from the previous full script



# This is the function that was causing the TypeError if called with wrong args
def embed_message_channel_bias(image_tensor: torch.Tensor, binary_message: str, 
                               strength_per_bit: float, num_bits_per_channel: int): # Removed defaults to ensure they are passed
    """
    Embeds a binary message by slightly shifting the mean of color channels.
    image_tensor: (B, C, H, W) in [0, 1]
    binary_message: String of '0's and '1's (e.g., 32 bits)
    strength_per_bit: How much to shift the channel mean for a '1' bit.
    num_bits_per_channel: How many bits from the message influence each channel.
    """
    if image_tensor.dim() != 4 or image_tensor.shape[1] != 3:
        raise ValueError("Image tensor for channel_bias must be 4D (B, 3, H, W)")
    
    img_to_modify = image_tensor[0].clone() 
    num_channels = img_to_modify.shape[0]
    total_bits = len(binary_message)
    
    bit_idx = 0
    for c in range(num_channels):
        channel_shift = 0.0
        # Ensure num_bits_per_channel is an integer for range()
        for _ in range(int(num_bits_per_channel)): # <<<<<<<<<<< CAST TO INT HERE
            if bit_idx < total_bits:
                if binary_message[bit_idx] == '1':
                    channel_shift += strength_per_bit
                bit_idx += 1
            else: 
                bit_idx = 0 
                if bit_idx < total_bits and binary_message[bit_idx] == '1': # Check again after reset
                    channel_shift += strength_per_bit
                bit_idx += 1 # Increment even if not used, to advance for next loop

        img_to_modify[c, :, :] += channel_shift
        
    modified_batch_image = image_tensor.clone()
    modified_batch_image[0] = img_to_modify.clamp(0, 1)
    return modified_batch_image




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
            cond_offset = total_images_to_generate_for_this_prompt # Key change for correct slicing
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

            if current_denoising_step == config.embedding_step:
                logger.info(f"  Embedding message at denoising step {current_denoising_step} (timestep {t.item()})")
                with torch.no_grad():
                    alpha_prod_t = noise_scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    pred_original_sample = (latents - beta_prod_t**(0.5) * noise_pred) / alpha_prod_t**(0.5)
                    image_latents_for_decode = (pred_original_sample / config.sd_vae_scale_factor).to(sd_vae.device, dtype=sd_vae.dtype)
                    decoded_x0_image = sd_vae.decode(image_latents_for_decode).sample
                    image_for_embedding = (decoded_x0_image / 2 + 0.5).clamp(0, 1)
                    
                    if accelerator.is_main_process:
                        pre_embed_path = os.path.join(image_save_dir, f"pre_embed_s{current_denoising_step}_b{batch_idx}_img{generated_image_count}.png")
                        torchvision.utils.save_image(image_for_embedding[0].float().cpu(), pre_embed_path)

                    # CHOOSE YOUR EMBEDDING METHOD HERE:
                    # Option 1: Patch-based (as per original config)
                    # modified_image_for_embedding = embed_message_patch_based(
                    #     image_for_embedding, 
                    #     config.binary_message,
                    #     config.embedding_patch_size, # From Config
                    #     config.embedding_strength    # From Config
                    # )

                    # Option 2: Channel Bias
                    modified_image_for_embedding = embed_message_channel_bias(
                         image_for_embedding,
                         config.binary_message,
                         strength_per_bit=config.channel_bias_strength_per_bit, # From Config
                         num_bits_per_channel=config.channel_bias_num_bits_per_channel # From Config
                     )

                    # Option 3: FFT Coeffs (as per your comment "Image->FFT->Embed->IFFT")
                    # modified_image_for_embedding = embed_message_fft_coeffs(
                    #     image_for_embedding,
                    #     config.binary_message,
                    #     strength=config.dct_strength, # From Config
                    #     num_coeffs_to_mod=config.dct_num_coeffs_to_mod, # From Config
                    #     start_coeff_idx=config.dct_start_coeff_idx # From Config
                    # )
                    
                    if accelerator.is_main_process:
                        post_embed_path = os.path.join(image_save_dir, f"post_embed_s{current_denoising_step}_b{batch_idx}_img{generated_image_count}.png")
                        torchvision.utils.save_image(modified_image_for_embedding[0].float().cpu(), post_embed_path)

                    modified_image_scaled_back = (modified_image_for_embedding * 2 - 1).to(sd_vae.device, dtype=sd_vae.dtype)
                    modified_latents_dist = sd_vae.encode(modified_image_scaled_back).latent_dist
                    modified_x0_latents = modified_latents_dist.sample() 
                    modified_x0_latents = modified_x0_latents * config.sd_vae_scale_factor
                    noise_for_current_step = torch.randn_like(modified_x0_latents)
                    latents = noise_scheduler.add_noise(
                        modified_x0_latents.to(latents.device, dtype=latents.dtype), 
                        noise_for_current_step.to(latents.device, dtype=latents.dtype), 
                        t.unsqueeze(0) 
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
                    image_latents_for_decode_inter = (pred_original_sample_inter / config.sd_vae_scale_factor).to(sd_vae.dtype)
                    image_decoded_inter = sd_vae.decode(image_latents_for_decode_inter).sample
                    image_to_save_inter = (image_decoded_inter / 2 + 0.5).clamp(0, 1)
                    
                    for img_idx_in_batch in range(image_to_save_inter.shape[0]):
                        global_img_idx = generated_image_count + img_idx_in_batch
                        intermediate_image_path = os.path.join(image_save_dir,
                            f"intermediate_s{current_denoising_step}_b{batch_idx}_img{global_img_idx}.png")
                        torchvision.utils.save_image(image_to_save_inter[img_idx_in_batch].float().cpu(), intermediate_image_path)
        
        with torch.no_grad():
            final_latents_for_decode = latents / config.sd_vae_scale_factor
            image_final_decoded = sd_vae.decode(final_latents_for_decode.to(sd_vae.dtype)).sample
            image_final_save_scaled = (image_final_decoded / 2 + 0.5).clamp(0, 1)

            for img_idx_in_batch in range(image_final_save_scaled.shape[0]):
                global_img_idx = generated_image_count + img_idx_in_batch
                final_image_name = os.path.join(image_save_dir, f"final_img{global_img_idx}_embedded_s{config.embedding_step}.png")
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
    logger.info(f"Embedding a 32-bit message: '{config.binary_message}' at step {config.embedding_step}")

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
        try: accelerator.end_training()
        except Exception as e: logger.warning(f"Error during accelerator.end_training(): {e}")
    logger.info("Image generation complete.")

# ========= Entry Point ==========
if __name__ == "__main__":
    config_run = Config(
        num_images_per_prompt=1, 
        inference_steps=100, 
        embedding_step=80,   
        save_intermediate_every_n_steps=10 
    )
    main(config_run)