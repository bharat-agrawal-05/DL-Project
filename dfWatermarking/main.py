import torch
import torchvision
from tqdm import tqdm
import os
import torch.nn.functional as F
from pathlib import Path
import re

# Diffusers and Transformers imports
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
# CLIPFeatureExtractor and StableDiffusionSafetyChecker are not strictly needed for this simplified script
# from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
# from transformers import CLIPFeatureExtractor

# Accelerate imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration

check_min_version("0.10.0.dev0") # diffusers check
logger = get_logger(__name__)

# ========= Config Class Definition ==========
class Config:
    def __init__(self, prompt: str = "A majestic lion in a lush jungle, cinematic lighting",
                 output_dir: str = "./generated_images_sd",
                 num_images_per_prompt: int = 1): # Defaulted to 1 for quicker test, was 10
        self.SEED = 42
        self.prompt = prompt
        self.output_dir = output_dir
        self.num_images_per_prompt = num_images_per_prompt # Number of images to generate for the prompt

        # Stable Diffusion parameters
        self.STABLE_DIFFUSION_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
        self.SD_REVISION = None # Or "fp16" for the fp16 branch
        self.sd_vae_scale_factor = 0.18215
        self.image_size = 512 # Output image size

        # Generation parameters
        self.batch_size_generation = 1 # How many images to generate in one go (latent batch size)
                                       # If num_images_per_prompt > batch_size_generation, multiple batches will run.
        self.inference_steps = 30     # Number of denoising steps
        self.guidance_scale = 7.5     # Classifier-Free Guidance scale

        # Intermediate saving
        self.save_intermediate_every_n_steps = 5 # Set to 0 to disable intermediate saving

        # Technical
        self.mixed_precision = "fp16" # "no", "fp16", "bf16" (bf16 needs Ampere or newer)
        self.gradient_accumulation_steps = 1 # Not training, so 1 is fine


# ========= Helper Functions (Inspired by original models.py) ==========

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, HUGGINGFACE_HUB_TOKEN:str=None):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        token=HUGGINGFACE_HUB_TOKEN # Updated from use_auth_token
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
        model_name,
        subfolder="tokenizer",
        revision=revision,
        use_fast=False,
        token=HUGGINGFACE_HUB_TOKEN
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(model_name, revision, HUGGINGFACE_HUB_TOKEN=HUGGINGFACE_HUB_TOKEN)
    text_encoder = text_encoder_cls.from_pretrained(
        model_name,
        subfolder="text_encoder",
        revision=revision,
        torch_dtype=weight_dtype,
        token=HUGGINGFACE_HUB_TOKEN
    )
    sd_vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae",
        revision=revision,
        torch_dtype=weight_dtype,
        token=HUGGINGFACE_HUB_TOKEN
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name,
        subfolder="unet",
        revision=revision,
        torch_dtype=weight_dtype,
        token=HUGGINGFACE_HUB_TOKEN
    )
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler", token=HUGGINGFACE_HUB_TOKEN)

    sd_vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    try:
        import xformers # Check if xformers is available
        # Before enabling, check if PyTorch version and xformers build are compatible
        # This simple check might not be foolproof for all edge cases of custom builds
        # but addresses the warning you saw.
        pt_version_major_minor = ".".join(torch.__version__.split(".")[:2])
        # xformers 0.0.23+ needs PyTorch 2.1.0. For older xformers or PyTorch, this might need adjustment.
        # This is a heuristic. A more robust check might involve parsing xformers._build_info
        if pt_version_major_minor >= "2.0": # General check for PyTorch 2.0+
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
    project_config = ProjectConfiguration(project_dir=logging_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard", # Will warn if tensorboard is not installed, but is fine
        #project_configuration=project_config
    )
    
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        try:
            accelerator.init_trackers("stable_diffusion_generation")
        except ImportError: # Handle case where tensorboard (or other tracker) is not installed
            logger.warning("TensorBoard (or other configured tracker) not found. Logging to trackers will be disabled.")


    set_seed(seed)
    return accelerator


def _encode_prompt(prompt_batch, tokenizer, text_encoder, device, num_images_per_prompt, do_classifier_free_guidance):
    text_inputs = tokenizer(
        prompt_batch,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device) # Move to device
    
    attention_mask = None
    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device) # Move to device

    prompt_embeds = text_encoder(
        text_input_ids, # Already on device
        attention_mask=attention_mask, # Already on device or None
    )[0] # Take the last hidden state, shape: (batch_size_prompts, seq_len, embed_dim)

    # `prompt_embeds` is already 3D: (num_prompt_in_batch, sequence_length, embed_features)
    # Here, num_prompt_in_batch is len(prompt_batch), which is 1 in our case.

    # Duplicate prompt embeddings for each generation per prompt
    # bs_embed is len(prompt_batch), which is 1.
    # After repeat, prompt_embeds shape: (1, num_images_per_prompt * seq_len, embed_dim) <- This was an error in logic
    # It should be repeated along the batch dimension for the UNet.
    # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1) <- This repeats sequence length
    # Correct: repeat along a new batch dimension for num_images_per_prompt
    
    # The UNet expects (batch_size_unet, seq_len, embed_dim)
    # Our 'batch_size_unet' for text embeddings will be num_images_per_prompt.
    # So, we need to replicate the single prompt's embedding num_images_per_prompt times.
    # Current prompt_embeds shape: (1, seq_len, embed_dim) because len(prompt_batch) is 1
    
    # Replicate the prompt embedding for each image to be generated from this prompt
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    # Now prompt_embeds shape: (num_images_per_prompt, seq_len, embed_dim)

    if do_classifier_free_guidance:
        uncond_tokens = [""] * len(prompt_batch) # Create batch of empty strings, len is 1
        max_length = text_input_ids.shape[-1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(device) # Move to device

        uncond_attention_mask = None
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            uncond_attention_mask = uncond_input.attention_mask.to(device) # Move to device
        
        negative_prompt_embeds = text_encoder(
            uncond_input_ids, # Already on device
            attention_mask=uncond_attention_mask, # Already on device or None
        )[0] # Shape: (1, seq_len, embed_dim)

        # Replicate the negative prompt embedding for each image
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        # Now negative_prompt_embeds shape: (num_images_per_prompt, seq_len, embed_dim)

        # For classifier-free guidance, concatenate unconditional and text embeddings
        # This creates a tensor of shape (2 * num_images_per_prompt, seq_len, embed_dim)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) 
    
    return prompt_embeds # Final shape for CFG: (2*num_images_per_prompt, seq_len, embed_dim)
                         # Final shape w/o CFG: (num_images_per_prompt, seq_len, embed_dim)


# ========= Image Generation Function ==========
def generate_images(
    accelerator: Accelerator,
    # prompt_text_embeddings: torch.Tensor, # This is what _encode_prompt returns
    all_prompt_embeddings: torch.Tensor, # Renamed for clarity
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
    unet_processing_batch_size = config.batch_size_generation # How many latents UNet processes at once
    
    num_unet_batches = (total_images_to_generate_for_this_prompt + unet_processing_batch_size - 1) // unet_processing_batch_size

    generated_image_count = 0
    do_classifier_free_guidance = config.guidance_scale > 1.0

    for batch_idx in range(num_unet_batches):
        # Determine how many images to generate in this specific UNet batch
        # This is for the latents batch size
        current_unet_batch_size = min(
            unet_processing_batch_size, 
            total_images_to_generate_for_this_prompt - (batch_idx * unet_processing_batch_size)
        )
        if current_unet_batch_size <= 0:
            continue

        latents_shape = (current_unet_batch_size, num_channels_latents, height // 8, width // 8)
        latents = torch.randn(
            latents_shape, generator=generator, device=accelerator.device, dtype=weight_dtype
        )
        latents = latents * noise_scheduler.init_noise_sigma

        noise_scheduler.set_timesteps(config.inference_steps, device=accelerator.device)
        timesteps = noise_scheduler.timesteps
        
        # Correctly slice the pre-generated prompt embeddings for this UNet batch
        # `all_prompt_embeddings` contains embeddings for all `total_images_to_generate_for_this_prompt`
        # If CFG: shape is (2 * total_images_to_generate_for_this_prompt, seq_len, embed_dim)
        # Else:   shape is (total_images_to_generate_for_this_prompt, seq_len, embed_dim)

        start_idx_for_this_batch_prompts = batch_idx * unet_processing_batch_size
        end_idx_for_this_batch_prompts = start_idx_for_this_batch_prompts + current_unet_batch_size

        if do_classifier_free_guidance:
            # Unconditional embeddings for the current_unet_batch_size
            # These are in the first half of all_prompt_embeddings
            uncond_batch_embeds = all_prompt_embeddings[
                start_idx_for_this_batch_prompts : end_idx_for_this_batch_prompts
            ]
            # Conditional embeddings for the current_unet_batch_size
            # These are in the second half, offset by total_images_to_generate_for_this_prompt
            cond_batch_embeds = all_prompt_embeddings[
                total_images_to_generate_for_this_prompt + start_idx_for_this_batch_prompts : \
                total_images_to_generate_for_this_prompt + end_idx_for_this_batch_prompts
            ]
            batch_prompt_embeddings_for_unet = torch.cat([uncond_batch_embeds, cond_batch_embeds], dim=0)
            # Expected shape: (2 * current_unet_batch_size, seq_len, embed_dim)
        else:
            batch_prompt_embeddings_for_unet = all_prompt_embeddings[
                start_idx_for_this_batch_prompts : end_idx_for_this_batch_prompts
            ]
            # Expected shape: (current_unet_batch_size, seq_len, embed_dim)

        if len(batch_prompt_embeddings_for_unet.shape) != 3:
            raise ValueError(
                f"batch_prompt_embeddings_for_unet must have 3 dimensions, "
                f"but got shape {batch_prompt_embeddings_for_unet.shape}"
            )
        expected_first_dim = 2 * current_unet_batch_size if do_classifier_free_guidance else current_unet_batch_size
        if batch_prompt_embeddings_for_unet.shape[0] != expected_first_dim:
            raise ValueError(
                f"batch_prompt_embeddings_for_unet has incorrect first dimension. "
                f"Got {batch_prompt_embeddings_for_unet.shape[0]}, expected {expected_first_dim} "
                f"for current_unet_batch_size {current_unet_batch_size} (CFG: {do_classifier_free_guidance})"
            )


        logger.info(f"\nBatch {batch_idx+1}/{num_unet_batches}: Generating {current_unet_batch_size} images for prompt: '{config.prompt}'")
        logger.info(f"  Latents shape: {latents.shape}, dtype: {latents.dtype}")
        logger.info(f"  UNet prompt embeddings shape: {batch_prompt_embeddings_for_unet.shape}")

        for i, t in enumerate(tqdm(timesteps, desc=f"Batch {batch_idx+1} Denoising")):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=batch_prompt_embeddings_for_unet # Use the correctly batched embeddings
                ).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
            
            if config.save_intermediate_every_n_steps > 0 and \
               (i + 1) % config.save_intermediate_every_n_steps == 0 and \
               (i + 1) < len(timesteps):
                with torch.no_grad():
                    alpha_prod_t = noise_scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                    image_latents_for_decode = pred_original_sample / config.sd_vae_scale_factor
                    image_decoded = sd_vae.decode(image_latents_for_decode.to(sd_vae.dtype)).sample
                    image_to_save = (image_decoded / 2 + 0.5).clamp(0, 1)
                    
                    for img_idx_in_batch in range(image_to_save.shape[0]): # image_to_save is for current_unet_batch_size
                        global_img_idx = generated_image_count + img_idx_in_batch
                        intermediate_image_path = os.path.join(image_save_dir,
                            f"intermediate_b{batch_idx}_img{global_img_idx}_step{i+1:03d}_t{t.item():04d}.png")
                        torchvision.utils.save_image(image_to_save[img_idx_in_batch].float(), intermediate_image_path)
        
        with torch.no_grad():
            final_latents_for_decode = latents / config.sd_vae_scale_factor
            image_final_decoded = sd_vae.decode(final_latents_for_decode.to(sd_vae.dtype)).sample
            image_final_save_scaled = (image_final_decoded / 2 + 0.5).clamp(0, 1)

            for img_idx_in_batch in range(image_final_save_scaled.shape[0]):
                global_img_idx = generated_image_count + img_idx_in_batch
                final_image_name = os.path.join(image_save_dir, f"final_img{global_img_idx}.png")
                torchvision.utils.save_image(image_final_save_scaled[img_idx_in_batch].float(), final_image_name)
                logger.info(f"Saved final image: {final_image_name}")
        
        generated_image_count += current_unet_batch_size # Increment by the number of images processed in this UNet batch
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

    unet.eval()
    text_encoder.eval()
    sd_vae.eval()

    do_classifier_free_guidance = config.guidance_scale > 1.0
    prompt_batch_for_encoder = [config.prompt] 
    
    # _encode_prompt now returns embeddings for all num_images_per_prompt
    # If CFG: shape is (2 * num_images_per_prompt, seq_len, embed_dim)
    # Else:   shape is (num_images_per_prompt, seq_len, embed_dim)
    all_prompt_embeddings = _encode_prompt(
        prompt_batch_for_encoder, tokenizer, text_encoder, accelerator.device, 
        config.num_images_per_prompt, 
        do_classifier_free_guidance
    )
    
    logger.info(f"Total prompt embeddings generated for UNet processing (shape): {all_prompt_embeddings.shape}")
    if do_classifier_free_guidance:
        # For logging, show shapes of uncond and cond parts if CFG
        num_total_for_prompt_single_side = config.num_images_per_prompt
        logger.info(f"  Shape of unconditional part (first half): {all_prompt_embeddings[:num_total_for_prompt_single_side].shape}")
        logger.info(f"  Shape of conditional part (second half): {all_prompt_embeddings[num_total_for_prompt_single_side:].shape}")

    generate_images(
        accelerator,
        all_prompt_embeddings, # Pass the pre-computed embeddings for all images
        noise_scheduler,
        sd_vae,
        unet,
        generator,
        weight_dtype,
        config,
        run_id=0 
    )
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            accelerator.end_training()
        except Exception as e:
            logger.warning(f"Error during accelerator.end_training(): {e}")


    logger.info("Image generation complete.")

# ========= Entry Point ==========
if __name__ == "__main__":
    config_run = Config()
    # For testing with multiple images and batching:
    # config_run = Config(num_images_per_prompt=4, output_dir="./test_multi_img")
    # config_run.batch_size_generation = 2 # UNet processes 2 latents at a time
    
    main(config_run)