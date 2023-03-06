import torch
import torch.nn.functional as F
import torch.nn as nn


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
temp_dim = 30
# temp_dim = 256
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        with torch.no_grad():
            delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
            loss = delta.dot(delta.T)
        torch.cuda.empty_cache()
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return loss

def mmd_data_standard(data):
    d_min = data.min()
    if d_min < 0:
        data = data + torch.abs(d_min)
        d_min = data.min()
    d_max = data.max()
    dst = d_max - d_min
    norm_data = (data - d_min).true_divide(dst)
    return norm_data

def mmd_loss(source, target):
    mmd = MMD_loss(kernel_type='linear')
    source = mmd_data_standard(source)
    target = mmd_data_standard(target)
    loss = mmd(source, target)
    loss.requires_grad_(True)
    return loss

#更新res
def RES(res, x, batchsize):
    res_out = torch.zeros((temp_dim,1))
    res_out = res_out.to(device)
    for i in range(0, batchsize):
        indices = torch.tensor([i])
        indices = indices.to(device)
        res_new = torch.index_select(res, 1, indices)
        x1 = torch.index_select(x, 0, indices)
        x1 = x1.view(x1.size(1), x1.size(2))
        D = torch.matmul(x1.T, x1)
        D_inv = torch.pinverse(D)
        D_mul = torch.matmul(x1, D_inv)
        theta = torch.matmul(torch.matmul(D_mul, x1.T), res_new)  # 利用最小二乘估计 计算一次
        res_new = res_new - theta   # 更新残差
        res_out = torch.cat((res_out, res_new),dim=1)
        if i == 0:
            res_out = res_out[:,1:]
    return res_out.T

def XISHU_A(A,batchsize, indexs,K,D_num):
    A_new = torch.zeros((D_num,batchsize))
    A_new = A_new.to(device)
    indexs = torch.tensor(indexs)
    x = indexs.reshape(K,batchsize)
    a = torch.LongTensor(x)
    a = a.to(device)
    A_new.scatter_(0, a, A)
    A_new = A_new.to(device)
    return A_new

#求解稀疏系数A
def getA(H, x, batchsize, k):
    H = H.T
    A_new = torch.zeros((k,1))
    A_new = A_new.to(device)
    for i in range(0, batchsize):
        indices = torch.tensor([i])
        indices = indices.to(device)
        H1 = torch.index_select(H, 1, indices)
        x1 = torch.index_select(x, 0, indices)
        x1 = x1.view(x1.size(1), x1.size(2))
        A = torch.matmul(x1.T, x1)
        A_inv = torch.pinverse(A)#.detach()
        A = torch.matmul(torch.matmul(A_inv, x1.T), H1)

        A_new = torch.cat((A_new,A),dim=1)
        if i == 0:
            A_new = A_new[:,1:]


    return A_new

#求解矩阵D
def getD(H, A):
    H = H.T
    b = torch.matmul(A, A.T)
    A_inv = torch.pinverse(b)#.detach()
    D = torch.matmul(torch.matmul(H, A.T), A_inv)

    return D

def euclidean_distances(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.mm(x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

#求解res和D的相关度
def Relevancy(res, D, a, batchsize, D_new, indexs, attention,attention1):
    res_1, D_1, res_2, D_2= attention(res, D)
    B1 = torch.matmul(res_1, D_1)
    B2 = torch.matmul(D_2, res_2)
    B = B1 + B2.T
    B = torch.abs(B)
    B = F.softmax(B,dim=1)
    relevancy = (B == B.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
    for i in range(0, batchsize):
        D_copy = D
        indices = torch.tensor([i])
        indices = indices.to(device)
        relevancy1 = torch.index_select(relevancy, 0, indices)
        index = torch.argmax(relevancy1)
        indexs.append(index)
        x = D_copy[:, index]  # 每K次稀疏得到的x
        x = x.view(1, temp_dim, 1)
        a = torch.cat((a, x), dim=0)
        D_copy[:, index] = 0
        D_copy = D_copy.view(1, D.size(0), D.size(1))
        D_new = torch.cat((D_new, D_copy),dim=0)
        if i == 0:
            a = a[1:, :, :]
            D_new = D_new[1:, :, :]
    return a, D_new,indexs

#求解res和D的相关度
def Relevancy1(res, D, a, batchsize, D_new, x, indexs, attention, attention1):
    for i in range(0, batchsize):
        indices = torch.tensor([i])
        indices = indices.to(device)
        res1 = torch.index_select(res, 0, indices)
        D1 = torch.index_select(D, 0, indices)
        D1 = D1.view(D1.size(1), D1.size(2))
        res2, D2, res3, D3 = attention(res1, D1)
        B2 = torch.matmul(res2, D2)
        B3 = torch.matmul(D3, res3)
        B = B2 + B3.T
        B = torch.abs(B)
        B = F.softmax(B, dim=1)
        relevancy = (B == B.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
        index = torch.argmax(relevancy)
        indexs.append(index)
        b = D1[:, index]  # 每K次稀疏得到的x
        b = b.view(1, temp_dim, 1)
        a = torch.cat((a, b), dim=0)
        D1[:, index] = 0
        D1 = D1.view(1, D1.size(0), D1.size(1))
        D_new = torch.cat((D_new, D1),dim=0)
        if i == 0:
            a = a[1:, :, :]
            D_new = D_new[1:, :, :]
    x = torch.cat((x, a),dim=2)
    return x, D_new, indexs

def get_D_and_A(HSI_D, LiDAR_D, output_HSI, output_LiDAR, K, batchsize, D_num, attention,attention1):
    HSI_A, HSI_D = OMP(output_HSI, HSI_D, K, batchsize, D_num, attention,attention1)
    LiDAR_A, LiDAR_D = OMP1(output_LiDAR, LiDAR_D, K, batchsize, D_num, attention,attention1)
    return output_HSI, HSI_A, HSI_D, output_LiDAR, LiDAR_A, LiDAR_D

def get_D_and_A_m(HSI_D, LiDAR_D, output_HSI, output_LiDAR, m, K, batchsize,D_num, attention,attention1):
    for i in range(0,m):
        output_HSI, HSI_A, HSI_D,\
            output_LiDAR, LiDAR_A, LiDAR_D= get_D_and_A(HSI_D, LiDAR_D, output_HSI, output_LiDAR,
                                                                                  K, batchsize, D_num, attention,attention1)
    return output_HSI, HSI_A, HSI_D, output_LiDAR, LiDAR_A, LiDAR_D

#求解A和D
def OMP(H, D, K, batchsize, D_num, attention,attention1):
    res = H
    a = torch.zeros((1, temp_dim, 1))
    a = a.to(device)
    D_new = torch.zeros((1, temp_dim, D_num))
    D_new = D_new.to(device)
    indexs = []
    for j in range(0, K):
        if j ==0:
            x, D, indexs = Relevancy(res, D, a, batchsize, D_new, indexs, attention,attention1)
        else:
            x, D, indexs= Relevancy1(res, D, a, batchsize, D_new, x, indexs, attention,attention1)
        res = res.T
        res = RES(res, x, batchsize)
    A = getA(H, x, batchsize, K)
    A = XISHU_A(A, batchsize, indexs, K, D_num)
    D = getD(H, A)
    return A, D


def OMP1(H, D, K, batchsize, D_num, attention,attention1):
    res = H
    a = torch.zeros((1, temp_dim, 1))
    a = a.to(device)
    D_new = torch.zeros((1, temp_dim, D_num))
    D_new = D_new.to(device)
    indexs = []
    for j in range(0, K):
        if j ==0:
            x, D, indexs = Relevancy(res, D, a, batchsize, D_new, indexs, attention,attention1)
        else:
            x, D, indexs= Relevancy1(res, D, a, batchsize, D_new, x, indexs, attention,attention1)
        res = res.T
        res = RES(res, x, batchsize)
    A = getA(H, x, batchsize, K)
    A = XISHU_A(A, batchsize, indexs, K, D_num)
    D = getD(H, A)
    return A, D