# custom_videoseal_loader.py
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial # For norm_layer in ImageEncoderViT

# --- From videoseal/modules/common.py (if LayerNorm is used directly and not torch.nn.LayerNorm) ---
# Assuming videoseal.modules.common.LayerNorm is a specific implementation
# If it's just torch.nn.LayerNorm, this can be simpler.
# For now, let's define a compatible LayerNorm if it's custom.
class LayerNorm(nn.LayerNorm): # Placeholder if it's just torch.nn.LayerNorm
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__(normalized_shape, eps=eps)
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return super().forward(x)
        elif self.data_format == "channels_first":
            # Permute to (N, H, W, C) for LayerNorm
            x = x.permute(0, 2, 3, 1)
            x = super().forward(x)
            # Permute back to (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            return x

# --- From videoseal/modules/common.py ---
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

# --- From videoseal/modules/vit.py ---
# (Pasting the full vit.py content here, then ImageEncoderViT, Block, Attention, PatchEmbed, etc.)
class PatchEmbed(nn.Module):
    def __init__(
        self, kernel_size: tuple[int, int] = (16, 16), stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0), in_chans: int = 3, embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x); x = x.permute(0, 2, 3, 1); return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear",)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
    attn: torch.Tensor, q: torch.Tensor, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
    q_size: tuple[int, int], k_size: tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size; k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h); Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape; r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh); rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)
    return attn

class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8, qkv_bias: bool = True, use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True, input_size: tuple[int, int] = None,
    ) -> None:
        super().__init__(); self.num_heads = num_heads; head_dim = dim // num_heads; self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias); self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos: attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x); return x

def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    B, H, W, C = x.shape; pad_h = (window_size - H % window_size) % window_size; pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0: x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)

def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]) -> torch.Tensor:
    Hp, Wp = pad_hw; H, W = hw; B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W: x = x[:, :H, :W, :].contiguous()
    return x

class Block(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm, act_layer: nn.Module = nn.GELU, use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True, window_size: int = 0, input_size: tuple[int, int] = None,
    ) -> None:
        super().__init__(); self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init, input_size=input_size if window_size == 0 else (window_size, window_size),)
        self.norm2 = norm_layer(dim); self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x; x = self.norm1(x)
        if self.window_size > 0: H, W = x.shape[1], x.shape[2]; x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0: x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x; x = x + self.mlp(self.norm2(x)); return x

class ImageEncoderViT(nn.Module):
    def __init__(
        self, img_size: int = 256, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 384,
        depth: int = 12, num_heads: int = 6, mlp_ratio: float = 4.0, out_chans: int = 256,
        qkv_bias: bool = True, norm_layer: nn.Module = nn.LayerNorm, act_layer: nn.Module = nn.GELU,
        use_abs_pos: bool = True, use_rel_pos: bool = False, rel_pos_zero_init: bool = True,
        window_size: int = 0, global_attn_indexes: tuple[int, ...] = (),
        temporal_attention: bool = False, max_temporal_length: int = 32, # Added temporal for completeness
    ) -> None:
        super().__init__(); self.img_size = img_size
        self.patch_embed = PatchEmbed(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), in_chans=in_chans, embed_dim=embed_dim,)
        self.pos_embed: nn.Parameter = None
        if use_abs_pos: self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        # Simplified: Removed temporal attention parts for this image-only use case
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                act_layer=act_layer, use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),)
            self.blocks.append(block)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False,),
            LayerNorm(out_chans, data_format="channels_first"), # Use our defined LayerNorm
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False,),
            LayerNorm(out_chans, data_format="channels_first"), # Use our defined LayerNorm
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None: x = x + self.pos_embed
        for blk in self.blocks: x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2).contiguous()); return x

# --- From videoseal/modules/pixel_decoder.py ---
class PixelDecoder(nn.Module):
    def __init__(self, embed_dim=256, nbits=0, num_select=1, decoder_type='mask'):
        super().__init__()
        self.nbits = nbits
        self.decoder_type = decoder_type
        self.mask_tokens = nn.Embedding(num_select, embed_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm(embed_dim // 4, data_format="channels_first"), # Use our defined LayerNorm
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, nbits + 1, kernel_size=1),  # predict (1+nbits) masks
        )
    def forward(self, image_embeddings): # image_embeddings: B, C, H, W
        # Currently, we only support single-point prompting
        # B, C, H, W -> B, C, H*W
        src = image_embeddings.reshape(image_embeddings.shape[0], image_embeddings.shape[1], -1)
        # B, N_mask, C @ B, C, H*W -> B, N_mask, H*W
        masks = self.mask_tokens.weight @ src
        # B, N_mask, H*W -> B, N_mask*H*W
        masks = masks.reshape(-1, masks.shape[-1])
        # B, N_mask*H*W -> B, N_mask, H, W
        masks = masks.reshape(image_embeddings.shape[0], -1, image_embeddings.shape[-2], image_embeddings.shape[-1])
        # B, N_mask, H, W -> B, N_mask, H*4, W*4
        masks = self.output_upscaling(masks)
        return masks

# --- From videoseal/models/extractor.py ---
class Extractor(nn.Module):
    def __init__(self) -> None:
        super(Extractor, self).__init__(); self.preprocess = lambda x: x * 2 - 1
    def forward(self, imgs: torch.Tensor) -> torch.Tensor: return ... # This will be implemented by subclasses

class SegmentationExtractor(Extractor):
    def __init__(self, image_encoder: ImageEncoderViT, pixel_decoder: PixelDecoder) -> None:
        super(SegmentationExtractor, self).__init__()
        self.image_encoder = image_encoder; self.pixel_decoder = pixel_decoder
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = self.preprocess(imgs); latents = self.image_encoder(imgs)
        masks = self.pixel_decoder(latents); return masks

# We won't implement all extractor types for now, just what's needed or a common one.
# The build_extractor function would typically use a config.
# For y_256b_img.pth, we'll assume it's a SegmentationExtractor with specific ViT params.

def build_custom_videoseal_extractor(nbits, device, img_size=256, vit_model_size='tiny'):
    """
    Builds a VideoSeal extractor (detector) manually based on assumed configuration.
    This is a simplified version of what videoseal.evals.full.setup_model_from_checkpoint
    and the internal build_extractor might do for a specific model type.
    """
    # Configuration for a 'tiny' ViT-based SegmentationExtractor
    # These are educated guesses and might need adjustment if the state_dict keys don't match.
    if vit_model_size == 'tiny':
        encoder_embed_dim = 192 # Common for ViT-Tiny
        encoder_depth = 12
        encoder_num_heads = 3
        out_chans_vit = 256 # Output channels of ViT neck, input to PixelDecoder
    elif vit_model_size == 'small':
        encoder_embed_dim = 384
        encoder_depth = 12
        encoder_num_heads = 6
        out_chans_vit = 512 # Might be 256 or 512, check state_dict
    # Add more sizes if needed (e.g., 'base')

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=16, # Standard
        in_chans=3,
        embed_dim=encoder_embed_dim,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=4.0,
        out_chans=out_chans_vit, # Neck output channels
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), # Standard LayerNorm
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False, # Often False for simpler ViTs unless specified
        window_size=0, # Global attention
        global_attn_indexes=(), # Or typical ones like [2,5,8,11] if it's a deeper ViT
    )

    pixel_decoder = PixelDecoder(
        embed_dim=out_chans_vit, # Must match out_chans of ImageEncoderViT's neck
        nbits=nbits,
        num_select=1, # Default for single point prompting in SAM-like models
        decoder_type='mask' # Default
    )

    extractor = SegmentationExtractor(image_encoder=image_encoder, pixel_decoder=pixel_decoder)
    return extractor.to(device)


# --- Wrapper class similar to Videoseal for the .detect() method ---
# This is needed because your generateImage.py calls `videoseal_model.detect()`
class CustomVideosealDetectorWrapper(nn.Module):
    def __init__(self, extractor_model: Extractor, img_size: int = 256):
        super().__init__()
        self.detector = extractor_model
        self.img_size = img_size # Processing image size for the detector

    @torch.no_grad()
    def detect(self, imgs: torch.Tensor, is_video: bool = False, # is_video arg for compatibility
               interpolation: dict = {"mode": "bilinear", "align_corners": False, "antialias": True}) -> dict:
        """
        Simplified detect method for image-only use case, mimicking Videoseal.detect().
        """
        if is_video:
            # This custom loader is simplified for image DGM.
            # For actual video, videoseal's chunking logic would be needed.
            # For now, assume batch of images if is_video=True but imgs is BxCxHxW
            if len(imgs.shape) == 5: # B F C H W
                imgs = imgs.flatten(0,1) # Treat frames as batch B*F C H W

        imgs_res = imgs.clone()
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs_res = F.interpolate(imgs, size=(self.img_size, self.img_size), **interpolation)
        
        # The actual detection by the extractor
        preds_masks = self.detector(imgs_res) # B, (1+nbits), H_proc, W_proc
        
        # Videoseal's .detect method for images pools the mask to get per-image logits.
        # The "preds" in videoseal_outputs["preds"] from the original setup_model_from_checkpoint
        # had shape [batch_size, num_bits + 1] which implies spatial pooling.
        # Let's assume adaptive average pooling.
        # The output of SegmentationExtractor is Bx(1+nbits)xHxW (mask-like).
        # We need to convert this to Bx(1+nbits)
        
        # Global average pooling over spatial dimensions
        pooled_preds = torch.mean(preds_masks, dim=[2, 3]) # B, (1+nbits)

        return {"preds": pooled_preds} # Matches expected output structure