import torch
import torch.nn as nn
from conv import SpiralConv

# Self-Attention in GMR
class SA(nn.Module):
    def __init__(self, in_dim):
        super(SA, self).__init__()
        self.query_conv = nn.Linear(in_dim, in_dim)
        self.key_conv = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)
        self.dim = torch.tensor(in_dim, dtype=torch.float)

        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x).permute(0, 2, 1)
        proj_value = self.value_conv(x)
        energy = torch.bmm(proj_query, proj_key) / torch.sqrt(self.dim.to(device=x.device))
        attention = self.softmax(energy)
        out = torch.bmm(attention, proj_value)

        out = self.gamma * out + x

        return out
# GMR
class GlobalMeshRefiner(nn.Module):
    def __init__(self, conv_in, coord_dim, indices):
        super(GlobalMeshRefiner, self).__init__()
        self.sa = Ref_SA_Scale(conv_in + coord_dim)
        self.head = SpiralConv(conv_in + coord_dim, 3, indices)

    def forward(self, feature_in, coord2d_in, coord3d_in):
        att_in = torch.cat((feature_in, coord2d_in), 2)
        att = self.sa(att_in)
        fine_3d = self.head(att)
        fine_3d *= 0.5
        return fine_3d + coord3d_in

