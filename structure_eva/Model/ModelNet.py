import torch
from torch.nn import functional as F
import numpy as np
import copy
from torch.nn import Conv1d, Conv2d, InstanceNorm2d, Sigmoid, BatchNorm1d, Linear, ModuleList
from structure_eva.Model.ResNet import ResNet, ResNetBlock
from structure_eva.Model.Attention import Pair2Pair
from structure_eva.Model.Vox import Voxel
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import SAGPooling

np.set_printoptions(threshold=np.inf)


class Score_m(torch.nn.Module):
    def __init__(self,
                 twobody_size=32,
                 num_update=1, abag=False,
                 d_pair=128,
                 n_head_pair=4, r_ff=2,
                 p_drop=0.1,
                 ):

        super(Score_m, self).__init__()
        self.num_update = num_update
        # self.abag = abag
        self.GATv2Conv = GATv2Conv(77, 16, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25).jittable()
        self.GATv2Conv1 = GATv2Conv(16 * 8, 16, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25).jittable()
        self.GATv2Conv2 = GATv2Conv(16 * 8, 16, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25).jittable()
        self.GATv2Conv3 = GATv2Conv(16 * 8, 8, heads=8, edge_dim=1, add_self_loops=False, dropout=0.25).jittable()
        self.lin1 = Linear(8 * 8, 1)

        self.add_module("conv1d_1", Conv1d(640, d_pair // 2, 1, padding=0, bias=True))
        self.add_module("conv2d_1", Conv2d(twobody_size, d_pair, 1, padding=0, bias=True))
        self.add_module("conv2d_2", Conv2d(d_pair * 2, d_pair, 1, padding=0, bias=True))
        self.add_module("rep", Conv1d(512, d_pair // 2, 1, padding=0, bias=True))
        self.add_module("inorm_1", InstanceNorm2d(d_pair * 2, eps=1e-06, affine=True))
        self.add_module("inorm_2", InstanceNorm2d(d_pair, eps=1e-06, affine=True))
        self.add_module("sample", Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True))
        self.pair_updata = _get_clones(Pair2Pair(n_layer=1,n_att_head=n_head_pair,n_feat=d_pair,r_ff=r_ff,p_drop=p_drop), num_update)
        self.mask_conv2d = Conv2d(d_pair, 1, 1, padding=0, bias=True)
        self.sigmoid = Sigmoid()
        self.bn1 = BatchNorm1d(128)
        self.fc1 = Linear(128, 128)
        self.voxel = Voxel(20, 20)
        self.sag1 = SAGPooling(77, 0.5)

    def forward(self, data):

        ### GAT ####
        mie_mee = data['mie_mee']
        mie_feat = data['mie_feat'].permute(0, 2, 1)
        mie_mee_rep = torch.cat((mie_mee, mie_feat), dim=-2).squeeze(0).permute(1, 0).to(torch.float32)
        mae_adj = data["mae_adj"].squeeze().permute(1, 0)
        mae_adj_rep = torch.LongTensor(mae_adj.to(torch.int64).to(torch.device('cpu'))).to(mie_mee_rep.device
                                                                                        )
        mie_mee_mae_rep = self.sag1(mie_mee_rep, mae_adj_rep)
        gat_node = mie_mee_mae_rep[0]
        gat_edge = mie_mee_mae_rep[1]
        gat_node = F.gelu(self.GATv2Conv(gat_node, gat_edge))
        gat_node = F.gelu(self.GATv2Conv1(gat_node, gat_edge))
        gat_node = F.gelu(self.GATv2Conv2(gat_node, gat_edge))
        updated_mie_mee_mae_rep = self.GATv2Conv3(gat_node, gat_edge)
        mie_mee_mae_1d_rep = self.lin1(updated_mie_mee_mae_rep)

        ### 3D CNN ####
        nres = mie_mee.shape[2]
        mee_vidx = data["mee_vidx"].squeeze(0)
        mee_val = data["mee_val"].squeeze(0)
        mee_rep = self.voxel(mee_vidx, mee_val, nres).permute(1, 0).unsqueeze(0)
        mee_rep = F.gelu(self._modules["conv1d_1"](mee_rep))
        mee_rep_2d_1 = tile(mee_rep.unsqueeze(3), 3, nres)
        mee_rep_2d_2 = tile(mee_rep.unsqueeze(2), 2, nres)

        ### sample & transform ###
        mae_rep = data['mae'].permute(0, 3, 1, 2)
        mae_rep = self._modules["conv2d_1"](mae_rep)  ## sample
        mee_mae_rep = torch.cat([mee_rep_2d_1, mee_rep_2d_2, mae_rep], dim=1)
        mee_mae_rep = F.gelu(self._modules["inorm_1"](mee_mae_rep))
        mee_mae_rep = self._modules["conv2d_2"](mee_mae_rep)
        mee_mae_rep = F.gelu(self._modules["inorm_2"](mee_mae_rep))
        mee_mae_rep_sample = self._modules["sample"](mee_mae_rep)
        mee_mae_rep_sample = mee_mae_rep_sample.permute(0, 2, 3, 1)
        for i_m in range(self.num_update):   ## transform
            mee_mae_rep_sample = self.pair_updata[i_m](mee_mae_rep_sample)
        mee_mae_rep_sample = mee_mae_rep_sample.permute(0, 3, 1, 2)
        mee_mae_rep_sample = self.mask_conv2d(mee_mae_rep_sample)

        mie_mee_mae_reps = mie_mee_mae_1d_rep * mee_mae_rep_sample
        mie_mee_mae_reps = torch.mean(mie_mee_mae_reps, dim=(1, 2, 3)).reshape(1, 1)

        # option
        score = self.sigmoid(mie_mee_mae_reps)
        # score=mie_mee_mae_reps  ## infer and train not sigmoid
        return score


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def tile(a, dim, n_tile):
    # Get on the same device as indices
    device = a.device
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        device)
    return torch.index_select(a, dim, order_index)






