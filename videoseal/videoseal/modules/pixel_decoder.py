# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import Upsample


class PixelDecoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        nbits: int = 0,
        activation: Type[nn.Module] = nn.GELU,
        upscale_stages: List[int] = [4, 2, 2],
        upscale_type: str = 'bilinear',
        #FLAG BELOW
        sigmoid_output: bool = False,
        pixelwise: bool = False,
    ) -> None:
        """
        Predicts masks given an image embedding, using a simple CNN.

        Arguments:
            embed_dim (int): the input channel dimension
            nbits (int): the number of bits to predict (0 for zero-bit)
            activation (nn.Module): the type of activation to use when
            upscaling masks
            upscale_stages (List[int]): the upscaling factors to use
            upscale_type (str): the type of upscaling to use
            sigmoid_output (bool): whether to apply sigmoid to the output
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.nbits = nbits

        self.output_upscaling = []
        for up_factor in upscale_stages:
                self.output_upscaling += [
                        Upsample(upscale_type, embed_dim, embed_dim // up_factor, up_factor, activation),
                ]
                embed_dim //= up_factor
        self.output_upscaling = nn.Sequential(*self.output_upscaling)

        self.pixelwise = pixelwise
        if self.pixelwise:
            self.linear = nn.Conv2d(embed_dim, self.nbits + 1, stride=1, kernel_size=1)
        else:
            self.linear = nn.Linear(embed_dim, self.nbits + 1)

        self.sigmoid_output = sigmoid_output
    
        
            
    def forward(
        self,
        image_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
            image_embeddings (torch.Tensor): the embeddings from the image encoder

        Returns:
            torch.Tensor: batched predicted masks (1+nbits)
        """
        # Upscale mask embeddings and predict masks using the mask tokens
        upscaled_embedding = self.output_upscaling(image_embeddings)    # b c h/f w/f -> b c/f h w
        if not self.pixelwise:  
            upscaled_embedding = upscaled_embedding.mean(dim=[-2, -1])  # b c
        preds = self.linear(upscaled_embedding)    # b c/f ... -> b 1+nbits ...
        # Apply sigmoid if needed and return
        #RECHECKFLAG
        #print("THIS IS SIGMOID OUTPUT: ",self.sigmoid_output)
        if self.sigmoid_output: 
            return F.sigmoid(preds)
        return preds

