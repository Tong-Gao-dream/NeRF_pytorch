import torch
import numpy as np


def get_rays(H, W, K, c2w):

    # 在像素坐标下进行网格点采样，得到特定分辨率图像的各个像素点坐标。

    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    # 进行像素坐标到相机坐标的变换，将二维坐标转换为三维坐标，投影变换的逆变换

    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # 将相机坐标转换到世界坐标

    # 相机坐标与世界左边中各个轴的朝向不同，通过旋转对齐，rays_d
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 相机坐标与世界坐标原点不一致，需要通过相机坐标进行平移变换对齐。rays_o
    rays_o = c2w[:3,-1].expand(rays_d.shape)

    return rays_o, rays_d
