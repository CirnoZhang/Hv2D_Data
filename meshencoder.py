import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from conv import SpiralConv

# MSEB
class MSEB(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MSEB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        y = torch.transpose(y, 1, 2)
        x = torch.transpose(x, 1, 2)
        return x * y.expand_as(x)

# Our Mesh Encoder with MSEB+ISM
class MeshEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(MeshEncoder, self).__init__()
        indices_2d3 = indices[:, :indices.size(1) // 3 * 2]
        indices_d3 = indices[:, :indices.size(1) // 3]
        indices_1 = indices[:, 0:1]
        self.conv_2d3 = SpiralConv(in_channels, out_channels // 4, indices_2d3)
        self.conv_d3 = SpiralConv(in_channels, out_channels // 4, indices_d3)
        self.conv = SpiralConv(in_channels, out_channels // 2, indices)
        self.conv1 = SpiralConv(in_channels, out_channels, indices_1)
        self.refconv225 = SpiralConv(in_channels + 225, in_channels, indices)
        self.mseb = MSEB(in_channels, reduction=16)

    def forward(self, x, up_transform, roi_in):
        out = Pool(x, up_transform)
        b, n, c = out.size()

        """
        We resize the pose feature(15,15)(from HJE) to (1,225) and *n (n,225) to concate to mesh features (n,c)
        """
        pose = roi_in.view(b, 1, -1)
        pn_xy = torch.repeat_interleave(pose, repeats=n, dim=1)
        out = torch.cat((out, pn_xy), 2)
        out = self.refconv225(out)
        out = self.mseb(out)

        """
        we use ISM(from CMR) to encodes features
        """
        short_cut = self.conv1(out)
        p_d3 = self.conv_d3(out)
        p_2d3 = self.conv_2d3(out)
        p = self.conv(out)
        f = torch.cat((p, p_2d3, p_d3), 2)
        out = F.relu(short_cut + f)

        return out