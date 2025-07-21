import torch
import torch.nn.functional as F
import numpy as np


def scatter_nd(indices, updates, shape):
    # Get on the same device as indices
    device = indices.device

    # Initialize empty array
    size = np.prod(shape)
    out = torch.zeros(size).to(device)

    # Get flattened index (Calculation needs to be done in long to preserve indexing precision)
    temp = np.array([int(np.prod(shape[i + 1:])) for i in range(len(shape))])
    flattened_indices = torch.mul(indices.long(), torch.as_tensor(temp, dtype=torch.long).to(device)).sum(dim=1)

    # Scatter_add
    out = out.scatter_add(0, flattened_indices, updates)

    # Reshape
    return out.view(shape)


class Voxel(torch.nn.Module):
    def __init__(self, num_restype, dim):
        super(Voxel, self).__init__()
        self.num_restype = num_restype
        self.dim = dim
        self.add_module("retype", torch.nn.Conv3d(self.num_restype, self.dim, 1, padding=0,
                                                  bias=False))  # in_channels, out_channels, kernel_size,
        self.add_module("conv3d_1", torch.nn.Conv3d(20, 20, 3, padding=0, bias=True))
        self.add_module("conv3d_2", torch.nn.Conv3d(20, 30, 4, padding=0, bias=True))
        self.add_module("conv3d_3", torch.nn.Conv3d(30, 10, 4, padding=0, bias=True))
        self.add_module("pool3d_1", torch.nn.AvgPool3d(kernel_size=4, stride=4, padding=0))

    def forward(self, idx, val, nres):
        x = scatter_nd(idx, val, (nres, 24, 24, 24, self.num_restype))
        x = x.permute(0, 4, 1, 2, 3)
        out_retype = self._modules["retype"](x)
        out_conv3d_1 = F.gelu(self._modules["conv3d_1"](out_retype))
        out_conv3d_2 = F.gelu(self._modules["conv3d_2"](out_conv3d_1))
        out_conv3d_3 = F.gelu(self._modules["conv3d_3"](out_conv3d_2))
        out_pool3d_1 = self._modules["pool3d_1"](out_conv3d_3)
        voxel_emb = torch.flatten(out_pool3d_1.permute(0, 2, 3, 4, 1), start_dim=1, end_dim=-1)
        return voxel_emb

