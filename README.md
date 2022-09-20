# Hv2D_Data

This repository contains a PyTorch implementation of the paper (coming soon):
3D Hand Pose and Shape Estimation from Monocular RGB via Efficient 2D Cues
## Implementation
Our implementation is based on:
https://github.com/SeanChenxy/HandMesh

HJE:
After getting pred4 in cmr_pg, we use the following command to get the pose feature(HJE is defined in ./hje.py):
```python
joint_position, handposemap = Get_HandPoseMap(pred4[:, :self.uv_channel])
box_list = Get_BoxList(joint_position)
posefeature = GetPoseFeatures(handposemap, box_list, (15, 15))
```
MESB: In our implementation, we use our MeshEncoder (Encoder and MSEB is defined in ./meshencoder.py) to replace ParallelDeblock in cmr_pg.

GMR:
Similar to head in cmr, we use a 2D head to regress the 2D Hand Mesh , and feed the 2D Hand Mesh, 3D Hand Mesh, and features into GMR(GMR is defined in ./gmr.py):
```python
# instantiation in class CMR_PG
self.heads = nn.ModuleList()
self.heads2d = nn.ModuleList()
for oc, sp_idx in zip(self.out_channels[::-1], self.spiral_indices[::-1]):
    self.heads.append(SpiralConv(oc, self.in_channels, sp_idx))
    self.heads2d.append(SpiralConv(oc, 2, sp_idx))
self.gmr = nn.ModuleList()
"""
GMR
"""
for oc, sp_idx in zip(self.out_channels[::-1], self.spiral_indices[::-1]):
    self.gmr.append(GlobalMeshRefiner(oc, 2, sp_idx))

...

# call in CMR_PG.decoder
# 3D Mesh
pred = self.heads[i - 1](x)
# 2D Mesh
pred2d = self.heads2d[i - 1](x)

pred = self.gmr[i - 1](x, pred2d, pred)

```
