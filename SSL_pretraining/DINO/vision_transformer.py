# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        activation="gelu",
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if activation == "gelu":
            act_func = nn.GELU()
        elif activation == "relu":
            act_func = nn.ReLU()

        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_func)
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(act_func)
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


def patches_to_grid(images, nrow=7):
    """
    Arrange patches into a grid of images with gradients enabled.

    Args:
        images (Tensor): Input tensor of patches to be arranged into a grid.
        nrow (int, optional): Number of patches displayed in each row of the grid. Default is 7.

    Returns:
        Tensor: A tensor containing the grid of images.
    """
    # Get the shape of the input tensor
    batch_size, n_patches, height, width = images.shape

    # Check if the number of patches is divisible by nrow
    assert n_patches % nrow == 0, "Number of patches must be divisible by nrow"

    # Calculate number of columns in the grid
    ncol = n_patches // nrow

    # Unfold the patches along height and width dimensions
    # (batch_size, n_patches, nrow, height, width)
    unfolded_height = images.unfold(2, height, height)
    # (batch_size, n_patches, nrow, ncol, height, width)
    unfolded_patches = unfolded_height.unfold(3, width, width)

    # Reshape and permute to form the grid
    grid = unfolded_patches.contiguous().view(batch_size, nrow, ncol, height, width)
    grid = (
        grid.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch_size, nrow * height, ncol * width)
    )

    return grid


class DINOHeadV2(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        conv1x1 = []
        conv1x1.append(nn.Conv2d(2, 1, kernel_size=1, stride=1))
        conv1x1.append(nn.GELU())
        self.conv1x1 = nn.Sequential(*conv1x1)

        layers = []
        layers.append(nn.Linear(224 * 224, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # batch, 2, 1024, 7, 7
        x = x.reshape(x.shape[0], 2, 1024, x.shape[2], x.shape[3])
        # batch, 2, 32, 32, 49
        x = x.reshape(x.shape[0], x.shape[1], 32, 32, x.shape[3] * x.shape[4])
        # 2 batch 49 32 32
        x = x.permute(1, 0, 4, 2, 3)

        x1 = x[0]
        x2 = x[1]
        image1 = patches_to_grid(x1)
        image2 = patches_to_grid(x2)
        # batch 2 244 244
        x = torch.cat((image1.unsqueeze(dim=1), image2.unsqueeze(dim=1)), dim=1)
        # batch 1 244 244
        x = self.conv1x1(x)
        # batch 244 244
        x = x.squeeze(dim=1)
        # batch 50176
        x = torch.flatten(x, start_dim=-2)
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOHeadV3(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        kernel_size=3,
    ):
        super().__init__()
        conv3x3 = []
        conv3x3.append(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )
        conv3x3.append(nn.GELU())
        self.conv3x3 = nn.Sequential(*conv3x3)

        layers = []
        layers.append(nn.Linear(224 * 224, bottleneck_dim))
        layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # batch, 2, 1024, 7, 7
        x = x.reshape(x.shape[0], 2, 1024, x.shape[2], x.shape[3])
        # batch, 2, 32, 32, 49
        x = x.reshape(x.shape[0], x.shape[1], 32, 32, x.shape[3] * x.shape[4])
        # 2 batch 49 32 32
        x = x.permute(1, 0, 4, 2, 3)

        x1 = x[0]
        x2 = x[1]
        image1 = patches_to_grid(x1)
        image2 = patches_to_grid(x2)
        # batch 2 244 244
        x = torch.cat((image1.unsqueeze(dim=1), image2.unsqueeze(dim=1)), dim=1)
        # batch 1 244 244
        x = self.conv3x3(x)
        # batch 244 244
        x = x.squeeze(dim=1)
        # batch 50176
        x = torch.flatten(x, start_dim=-2)
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOHeadV4(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        kernel_size=3,
    ):
        super().__init__()
        conv = []
        conv.append(
            nn.Conv2d(
                2, 32, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            )
        )
        conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Conv2d(
                32, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            )
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # batch, 2, 1024, 7, 7
        x = x.reshape(x.shape[0], 2, 1024, x.shape[2], x.shape[3])
        # batch, 2, 32, 32, 49
        x = x.reshape(x.shape[0], x.shape[1], 32, 32, x.shape[3] * x.shape[4])
        # 2 batch 49 32 32
        x = x.permute(1, 0, 4, 2, 3)

        x1 = x[0]
        x2 = x[1]
        image1 = patches_to_grid(x1)
        image2 = patches_to_grid(x2)
        # batch 2 244 244
        x = torch.cat((image1.unsqueeze(dim=1), image2.unsqueeze(dim=1)), dim=1)
        # batch 1 244 244
        x = self.conv(x)
        # batch 244 244
        x = self.last_layer(x)
        x = x.squeeze(dim=1)
        # batch 50176
        x = torch.flatten(x, start_dim=-2)
        return x


class DINOHeadV4_1(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        kernel_size=3,
    ):
        super().__init__()
        conv_prime = []
        conv_prime.append(
            nn.Conv2d(4, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )
        conv_prime.append(nn.GELU())
        self.conv_prime = nn.Sequential(*conv_prime)

        upsample = []
        upsample.append(
            nn.ConvTranspose2d(
                1,
                1,
                kernel_size=2,
                stride=2,
            )
        )
        upsample.append(nn.GELU())
        self.upsample = nn.Sequential(*upsample)

        conv = []
        conv.append(
            nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )
        conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # batch, 2, 1024, 7, 7
        x_in = x.reshape(x.shape[0], 2, 1024, 7, 7)
        x = x_in[:, 0, :, :, :]
        # batch, 1, 1024, 7, 7
        x_prime = x_in[:, 1, :, :, :]
        # batch, 4, 256, 7, 7
        x_prime = x_prime.reshape(x_prime.shape[0], 4, 256, 7, 7)
        # batch, 4, 16, 16, 49
        x_prime = x_prime.reshape(
            x_prime.shape[0],
            4,
            16,
            16,
            49,
        )
        # 4 batch 49 16 16
        x_prime = x_prime.permute(1, 0, 4, 2, 3)
        img_1 = patches_to_grid(x_prime[0])
        img_2 = patches_to_grid(x_prime[1])
        img_3 = patches_to_grid(x_prime[2])
        img_4 = patches_to_grid(x_prime[3])
        # batch 4 112 112
        x_prime = torch.cat(
            (
                img_1.unsqueeze(dim=1),
                img_2.unsqueeze(dim=1),
                img_3.unsqueeze(dim=1),
                img_4.unsqueeze(dim=1),
            ),
            dim=1,
        )
        x_prime = self.conv_prime(x_prime)
        x_prime = self.upsample(x_prime)

        # batch, 1, 32, 32, 49
        x = x.reshape(x.shape[0], 1, 32, 32, 49)
        # 1 batch 49 32 32
        x = x.permute(1, 0, 4, 2, 3)

        x1 = x[0]
        image1 = patches_to_grid(x1)
        image1 = image1.unsqueeze(dim=1)
        x = self.conv(image1)
        x = torch.cat((x, x_prime), dim=1)
        # x = x.reshape(x.shape[0], x.shape[1], 224 * 224)
        # x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.reshape(x.shape[0], x.shape[1], 224, 224)
        x = self.last_layer(x)
        x = x.squeeze(dim=1)
        # batch 50176
        x = torch.flatten(x, start_dim=-2)
        return x


class DINOHeadConvTranspose(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        hidden_dim=2048,
        bottleneck_dim=256,
        channel_reduction_factor=8,
    ):
        super().__init__()
        # Layer 1
        layers = [
            nn.ConvTranspose2d(
                in_dim,
                in_dim // (channel_reduction_factor**1),
                kernel_size=2,
                stride=2,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(in_dim // (channel_reduction_factor**1)))
        layers.append(nn.GELU())
        # Layer 2
        layers.append(
            nn.ConvTranspose2d(
                in_dim // (channel_reduction_factor**1),
                in_dim // (channel_reduction_factor**2),
                kernel_size=2,
                stride=2,
            )
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(in_dim // (channel_reduction_factor**2)))
        layers.append(nn.GELU())
        # Layer 3
        layers.append(
            nn.ConvTranspose2d(
                in_dim // (channel_reduction_factor**2),
                in_dim // (channel_reduction_factor**3),
                kernel_size=2,
                stride=2,
            )
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(in_dim // (channel_reduction_factor**3)))
        layers.append(nn.GELU())
        self.upsample = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # output layer
        self.last_layer = nn.utils.weight_norm(
            nn.ConvTranspose2d(
                in_dim // (channel_reduction_factor**3),
                1,
                kernel_size=4,
                stride=4,
                bias=False,
            )
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        self.flatten = nn.Flatten()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            trunc_normal_(m.weight, std=0.02)
            if (
                isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d)
            ) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.upsample(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        x = self.flatten(x)
        return x


class DINOHeadConvTransposeV2(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        hidden_dim=2048,
        bottleneck_dim=256,
        channel_reduction_factor=8,
    ):
        super().__init__()
        # Layer 1
        layers = []
        layers.append(
            nn.ConvTranspose2d(
                in_dim,
                in_dim,
                kernel_size=2,
                stride=2,
            )
        )
        layers.append(nn.GELU())
        layers.append(
            nn.Conv2d(
                in_dim, in_dim // (channel_reduction_factor**1), kernel_size=1, stride=1
            )
        )
        layers.append(nn.GELU())

        # Layer 2
        layers.append(
            nn.ConvTranspose2d(
                in_dim // (channel_reduction_factor**1),
                in_dim // (channel_reduction_factor**1),
                kernel_size=2,
                stride=2,
            )
        )
        layers.append(nn.GELU())
        layers.append(
            nn.Conv2d(
                in_dim // (channel_reduction_factor**1),
                in_dim // (channel_reduction_factor**2),
                kernel_size=1,
                stride=1,
            )
        )
        layers.append(nn.GELU())

        # Layer 3
        layers.append(
            nn.ConvTranspose2d(
                in_dim // (channel_reduction_factor**2),
                in_dim // (channel_reduction_factor**2),
                kernel_size=2,
                stride=2,
            )
        )
        layers.append(nn.GELU())
        layers.append(
            nn.Conv2d(
                in_dim // (channel_reduction_factor**2),
                in_dim // (channel_reduction_factor**3),
                kernel_size=1,
                stride=1,
            )
        )
        layers.append(nn.GELU())

        # layer 4
        layers.append(
            nn.ConvTranspose2d(
                in_dim // (channel_reduction_factor**3),
                in_dim // (channel_reduction_factor**3),
                kernel_size=4,
                stride=4,
            )
        )
        layers.append(nn.GELU())
        self.upsample = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # output layer
        self.last_layer = nn.utils.weight_norm(
            nn.Conv2d(
                in_dim // (channel_reduction_factor**3),
                1,
                kernel_size=1,
                stride=1,
                bias=False,
            )
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        self.flatten = nn.Flatten()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            trunc_normal_(m.weight, std=0.02)
            if (
                isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d)
            ) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.upsample(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        x = self.flatten(x)
        return x


class DINOHeadConvTransposeV3(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        hidden_dim=2048,
        bottleneck_dim=256,
        channel_reduction_factor=8,
    ):
        super().__init__()
        # Layer 1
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        layers.append(
            nn.Conv2d(
                in_dim, in_dim // (channel_reduction_factor**1), kernel_size=1, stride=1
            )
        )
        layers.append(nn.GELU())

        # Layer 2
        layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        layers.append(
            nn.Conv2d(
                in_dim // (channel_reduction_factor**1),
                in_dim // (channel_reduction_factor**2),
                kernel_size=1,
                stride=1,
            )
        )
        layers.append(nn.GELU())

        # Layer 3
        layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        layers.append(
            nn.Conv2d(
                in_dim // (channel_reduction_factor**2),
                in_dim // (channel_reduction_factor**3),
                kernel_size=1,
                stride=1,
            )
        )
        layers.append(nn.GELU())
        layers.append(nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False))
        self.upsample = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # output layer
        self.last_layer = nn.utils.weight_norm(
            nn.Conv2d(
                in_dim // (channel_reduction_factor**3),
                1,
                kernel_size=1,
                stride=1,
                bias=False,
            )
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        self.flatten = nn.Flatten()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            trunc_normal_(m.weight, std=0.02)
            if (
                isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d)
            ) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.upsample(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        x = self.flatten(x)
        return x
