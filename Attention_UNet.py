# -*- coding: utf-8 -*-
'''
将上采样写成反卷积的样式;这是加Attention的Unet

这是加Attention的Unet,两种方式：1. 加在Unet开始和结尾，2. 加在Unet特征提取阶段
'''

import torch
import torch.nn as nn

# 通道attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes,  out_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d( out_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# 特征attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

# 混合attention
class CNNAttetion(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CNNAttetion, self).__init__()

        self.channel_att = ChannelAttention(in_planes, out_planes)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


#定义卷积层
class CNNLayer(nn.Module):
    def __init__(self, C_in, C_out):
        super(CNNLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C_in, C_out, 3, 1, 1),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(C_out, C_out, 3, 1, 1),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


#下采样用大步长卷积
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C, C, 3, 2, 1),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)

#上采样
class UpSampling(nn.Module):
    def __init__(self,C):
        super().__init__()
        self.up_layer=nn.ConvTranspose2d(C,C//2,3,2,1,1)

    def forward(self, x,r):
        x=self.up_layer(x)

        return torch.cat((x, r), 1)

# attetion加在了Unet的开始和结尾
# class MainNet(torch.nn.Module):
#     def __init__(self, nChannels):
#         super(MainNet, self).__init__()
#         self.attetion1 = CNNAttetion(1, nChannels//2)
#         #这里用的灰度图，所以是1
#         self.C1 = CNNLayer(1, nChannels)
#         self.D1 = DownSampling(nChannels)
#         self.C2 = CNNLayer(nChannels, nChannels*2)
#         self.D2 = DownSampling(nChannels*2)
#         self.C3 = CNNLayer(nChannels*2, nChannels*4)
#         self.D3 = DownSampling(nChannels*4)
#         self.C4 = CNNLayer(nChannels*4, nChannels*8)
#         self.D4 = DownSampling(nChannels*8)
#         self.C5 = CNNLayer(nChannels*8, nChannels*16)
#         self.U1 = UpSampling(nChannels*16)
#         self.C6 = CNNLayer(nChannels*16, nChannels*8)
#         self.U2 = UpSampling(nChannels*8)
#         self.C7 = CNNLayer(nChannels*8, nChannels*4)
#         self.U3 = UpSampling(nChannels*4)
#         self.C8 = CNNLayer(nChannels*4, nChannels*2)
#         self.U4 = UpSampling(nChannels*2)
#         self.C9 = CNNLayer(nChannels*2, nChannels)
#         self.Th = torch.nn.Sigmoid()
#         # 这里用的灰度图，输出和原图要对于，所以是1
#         self.pre = torch.nn.Conv2d(nChannels, 1, 3, 1, 1)
#         self.attetion2 = CNNAttetion(1, nChannels//2)
#
#     def forward(self, x):
#         x = self.attetion1(x)
#         R1 = self.C1(x)
#         R2 = self.C2(self.D1(R1))
#         R3 = self.C3(self.D2(R2))
#         R4 = self.C4(self.D3(R3))
#         Y1 = self.C5(self.D4(R4))
#         O1 = self.C6(self.U1(Y1, R4))
#         O2 = self.C7(self.U2(O1, R3))
#         O3 = self.C8(self.U3(O2, R2))
#         O4 = self.C9(self.U4(O3, R1))
#         Th = self.Th(self.pre(O4))
#         return self.attetion2(Th)


# attetion加在了Unet的特征提取阶段
class MainNet(torch.nn.Module):
    def __init__(self, nChannels):
        super(MainNet, self).__init__()
        self.attetion1 = CNNAttetion(1, nChannels//2)
        self.C1 = CNNLayer(1, nChannels)
        self.D1 = DownSampling(nChannels)

        self.attetion2 = CNNAttetion(nChannels, nChannels)
        self.C2 = CNNLayer(nChannels, nChannels*2)
        self.D2 = DownSampling(nChannels*2)

        self.attetion3 = CNNAttetion(nChannels*2, nChannels*2)
        self.C3 = CNNLayer(nChannels*2, nChannels*4)
        self.D3 = DownSampling(nChannels*4)

        self.attetion4 = CNNAttetion(nChannels*4, nChannels*4)
        self.C4 = CNNLayer(nChannels*4, nChannels*8)
        self.D4 = DownSampling(nChannels*8)

        self.attetion5 = CNNAttetion(nChannels*8, nChannels*8)
        self.C5 = CNNLayer(nChannels*8, nChannels*16)
        self.U1 = UpSampling(nChannels*16)
        self.C6 = CNNLayer(nChannels*16, nChannels*8)
        self.U2 = UpSampling(nChannels*8)
        self.C7 = CNNLayer(nChannels*8, nChannels*4)
        self.U3 = UpSampling(nChannels*4)
        self.C8 = CNNLayer(nChannels*4, nChannels*2)
        self.U4 = UpSampling(nChannels*2)
        self.C9 = CNNLayer(nChannels*2, nChannels)
        self.Th = torch.nn.Sigmoid()
        self.pre = torch.nn.Conv2d(nChannels, 1, 3, 1, 1)

    def forward(self, x):
        R1 = self.C1(self.attetion1(x))
        R2 = self.C2(self.attetion2(self.D1(R1)))
        R3 = self.C3(self.attetion3(self.D2(R2)))
        R4 = self.C4(self.attetion4(self.D3(R3)))
        Y1 = self.C5(self.attetion5(self.D4(R4)))
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))
        Th = self.Th(self.pre(O4))
        return Th


if __name__ == '__main__':
    '''加载普通unet训练好的模型来训练加Attention的unet'''
    # 以下是加载更改后模型的代码
    net = MainNet(16)
    net_dict = net.state_dict() # 更改后模型的初始参数
    pretrained_dict = torch.load("module.pkl") # 加载预训练模型参数
    # # 将pretrained_dict的参数与net_dict参数进行比较，剔除不同的
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
    net_dict.update(pretrained_dict) # 更新现有的net_dict
    net.load_state_dict(net_dict) # 加载更改后的模型参数



