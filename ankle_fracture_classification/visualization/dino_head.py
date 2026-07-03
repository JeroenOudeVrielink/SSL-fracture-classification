import torch
from torch import nn
from dino_utils import trunc_normal_


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
        in_dim=2056,
        out_dim=256**2,
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
        super_feature_map = x
        # batch 244 244
        x = x.squeeze(dim=1)
        # batch 50176
        x = torch.flatten(x, start_dim=-2)
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x, super_feature_map


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
    ):
        super().__init__()
        conv3x3 = []
        conv3x3.append(nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1))
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
        super_fmap = self.conv3x3(x)
        # batch 244 244
        x = super_fmap.squeeze(dim=1)
        # batch 50176
        x = torch.flatten(x, start_dim=-2)
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x, super_fmap
