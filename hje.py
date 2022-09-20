import torch
from torchvision.ops import roi_align

def Get_HandPoseMap(pred4_uv):
    #input: joints heatmaps from 2D stage
    posehtmp = pred4_uv
    #max point in each joint heatmap remain, others points set 0
    posehtmp_max = posehtmp.view(posehtmp.size()[0], posehtmp.size()[1], -1).max(dim=2,keepdim=True)
    jointposemap = torch.zeros((posehtmp.size())).to(posehtmp.device)
    for b in range(posehtmp.size()[0]):
        for c in range(posehtmp.size()[1]):
            jointposemap.view((posehtmp.size()[0], posehtmp.size()[1], -1))[b][c][
                posehtmp_max[1][b][c]] = posehtmp_max[0][b][c]
    # get coord of each joint to set RoIBox
    joint_position = jointposemap.nonzero().view(jointposemap.size()[0], jointposemap.size()[1], -1).T[2::].T
    #sum the 21 jointposemaps into one handposemap
    handposemap = torch.sum(jointposemap, dim=1, keepdim=True)
    return joint_position, handposemap

def Get_BoxList(joint_position):
    box_list = []
    for b in range(joint_position.size()[0]):
        x1 = joint_position[b].T[0].min()
        x2 = joint_position[b].T[0].max()
        y1 = joint_position[b].T[1].min()
        y2 = joint_position[b].T[1].max()
        box = torch.tensor([[x1, y1, x2, y2]]).to(joint_position.device)
        box_list.append(box.float())
    return box_list

def GetPoseFeatures(handposemap, box_list, size=(15, 15)):
    posefeature = roi_align(handposemap, box_list, size, aligned=True)
    return posefeature