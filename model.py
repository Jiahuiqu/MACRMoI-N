import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_D_and_A_m
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.linear_1 = nn.Linear(30, 60)
        self.linear_2 = nn.Linear(30, 60)
        # self.linear_1 = nn.Linear(256, 256)
        # self.linear_2 = nn.Linear(256, 256)
    def forward(self, H, D):
        H1 = self.linear_1(H)
        D1 = self.linear_2(D.T).T
        H2 = self.linear_1(H).T
        D2 = self.linear_2(D.T)
        return H1, D1, H2, D2


class Model(nn.Module):
    def __init__(self, bands: int, FM: int, Classes:int):
        super(Model,self).__init__()
        self.attention_net = attention()
        self.out = nn.Sequential(
            nn.Linear(120, 60),
            nn.Linear(60, Classes),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(63, 45),
            nn.Linear(45, 30),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1,15),
            nn.Linear(15, 30),
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, H_HSI, H_LiDAR,HSI_D, LiDAR_D,Patch_HSI,Patch_LiDAR,D_Patch_HSI, D_Patch_LiDAR, K, m, batchsize, D_num):
        data_HSI_S = Patch_HSI
        data_LiDAR_S = Patch_LiDAR
        D_HSI_S = D_Patch_HSI
        D_LiDAR_S = D_Patch_LiDAR

        # hsi_H
        data_HSI = H_HSI.clone()
        data_HSI = data_HSI.view(data_HSI.size(0), -1, 1)

        # lidar_H
        data_LiDAR = H_LiDAR.clone()
        data_LiDAR = data_LiDAR.view(data_LiDAR.size(0), -1, 1)

        # hsi_D
        HSI_D = HSI_D.view(HSI_D.size(0), -1, 1)

        D_HSI_S = D_HSI_S.view(D_HSI_S.size(0), D_HSI_S.size(1), -1)
        # lidar_D
        LiDAR_D = LiDAR_D.view(LiDAR_D.size(0), -1, 1)
        D_LiDAR_S = D_LiDAR_S.view(D_LiDAR_S.size(0), D_LiDAR_S.size(1), -1)

        # 字典的HSI_P
        D_HSI_P = torch.pairwise_distance(HSI_D, D_HSI_S)
        D_HSI_P = torch.exp(-D_HSI_P)

        D_HSI_P = F.softmax(D_HSI_P, dim=1)
        D_HSI_P = D_HSI_P.view(D_HSI_P.size(0), -1, 1)
        D_HSI_P = D_HSI_P.to(dtype=torch.float32)

        # 字典的LiDAR_P
        D_LiDAR_P = torch.pairwise_distance(LiDAR_D, D_LiDAR_S)
        D_LiDAR_P = torch.exp(-D_LiDAR_P)
        D_LiDAR_P = F.softmax(D_LiDAR_P, dim=1)
        D_LiDAR_P = D_LiDAR_P.view(D_LiDAR_P.size(0), -1, 1)
        D_LiDAR_P = D_LiDAR_P.to(dtype=torch.float32)
        # train的HSI_S
        data_HSI_S = data_HSI_S.view(data_HSI_S.size(0), data_HSI_S.size(1), -1)

        # train的LiDAR_S
        data_LiDAR_S = data_LiDAR_S.view(data_LiDAR_S.size(0), data_LiDAR_S.size(1), -1)

        # train的HSI_P
        data_HSI_P = torch.pairwise_distance(data_HSI, data_HSI_S)
        data_HSI_P = torch.exp(-data_HSI_P)

        data_HSI_P = F.softmax(data_HSI_P, dim=1)
        data_HSI_P = data_HSI_P.view(data_HSI_P.size(0), -1, 1)
        data_HSI_P = data_HSI_P.to(dtype=torch.float32)

        # train的LiDAR_P
        data_LiDAR_P = torch.pairwise_distance(data_LiDAR, data_LiDAR_S)
        data_LiDAR_P = torch.exp(-data_LiDAR_P)
        data_LiDAR_P = F.softmax(data_LiDAR_P, dim=1)
        data_LiDAR_P = data_LiDAR_P.view(data_LiDAR_P.size(0), -1, 1)
        data_LiDAR_P = data_LiDAR_P.to(dtype=torch.float32)

        SP_HSI = torch.matmul(data_HSI_S, data_HSI_P)
        SP_HSI = SP_HSI.view(SP_HSI.size(0), -1)
        SP_LiDAR = torch.matmul(data_LiDAR_S, data_LiDAR_P)
        SP_LiDAR = SP_LiDAR.view(SP_LiDAR.size(0), -1)

        D_SP_HSI = torch.matmul(D_HSI_S, D_HSI_P)
        D_SP_HSI = D_SP_HSI.view(D_SP_HSI.size(0), -1)

        data_HSI = SP_HSI
        data_LiDAR = SP_LiDAR
        data_HSI = self.linear1(data_HSI)
        data_LiDAR = self.linear2(data_LiDAR)

        D_SP_LiDAR = torch.matmul(D_LiDAR_S, D_LiDAR_P)
        D_SP_LiDAR = D_SP_LiDAR.view(D_SP_LiDAR.size(0), -1)

        HSI_D = D_SP_HSI
        LiDAR_D = D_SP_LiDAR
        HSI_D = self.linear1(HSI_D).T
        LiDAR_D = self.linear2(LiDAR_D).T

        output_HSI, HSI_A, HSI_D, output_LiDAR, LiDAR_A, LiDAR_D = get_D_and_A_m(HSI_D, LiDAR_D, data_HSI, data_LiDAR,
                                                                                 m, K, batchsize, D_num,
                                                                                 self.attention_net, self.attention_net)
        HSI_A = HSI_A.T
        LiDAR_A = LiDAR_A.T


        A = torch.cat((LiDAR_A, HSI_A),dim = 1)
        output = self.out(A)
        return  data_HSI, data_LiDAR, HSI_A, HSI_D, LiDAR_A, LiDAR_D, output