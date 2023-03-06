import numpy as np
import torch
import random
from thop import profile
import time
import scipy.io as sio
from dataset import get_dataset, data_standard, print_data, data_partition, gen_model_data, data_HSI_LIDATR
from model import Model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#超参数

print('\n')
Seed_List = [1]   # 随机种子点

# ###################### 超参预设 ######################
curr_train_ratio = 50   # 每类训练集占这类总样本的比例，或每类训练样本的个数
# 的超参
patchsize_HSI = 11
patchsize_LiDAR = 11
batchsize = 300
LR = 0.01
FM = 30     # 输出的维度
BestAcc = 0     # 最优精度
m = 1
K = 10
D_num = 60  #字典个数


# ###################### 加载数据集 ######################
samples_type = ['ratio', 'same_num', 'fixed'][1]    # 训练集按照 0-按比例取训练集 1-按每类个数取训练集
# 选择数据集
datasets = 2
# dataset=1, 2012Houston
# dataset=2, trento
# 加载数据
[data_HSI, data_LiDAR, gt, train_gt, val_ratio, class_count, learning_rate,
 max_epoch, dataset_name, trainloss_result, LiDAR_bands] = get_dataset(datasets)

# 源域和目标域数据信息
height, width, bands = data_HSI.shape
# 数据标准化
[data_HSI, data_LiDAR] = data_standard(data_HSI, data_LiDAR, LiDAR_bands)
# 给LiDAR降一个维度
data_LiDAR = data_LiDAR[:, :, 0]

# 打印每类样本个数
print('#####源域样本个数#####')
print_data(gt, class_count)

# ###################### 参数初始化 ######################
train_samples_per_class = curr_train_ratio  # 当定义为每类样本个数时,则该参数更改为训练样本数
train_ratio = curr_train_ratio  # 训练比例

# ###################### 划分训练测试验证集 ######################
for curr_seed in Seed_List:
    random.seed(curr_seed)  # 当seed()没有参数时，每次生成的随机数是不一样的，而当seed()有参数时，每次生成的随机数是一样的

    # 对源域样本进行划分，得到训练、测试、验证集, 初始化D
    [train_label,test_label,D_label] = data_partition(samples_type, class_count,  gt, train_gt,train_ratio, val_ratio, height, width)

    # ###################### 搭建网络 ######################
    # 搭建两个网络分别对HSI和LiDAR进行特征提取

    [TrainPatch_HSI, TrainPatch_LiDAR,TrainLabel_HSI,TestPatch_HSI, TestPatch_LIDAR,TestLabel_HSI, D_Patch_HSI, D_Label_HSI,D_Patch_LiDAR, D_Label_LiDAR] = \
        gen_model_data(data_HSI, data_LiDAR, patchsize_HSI, patchsize_LiDAR,
                                           train_label, test_label, D_label, batchsize)

    [train_data_HSI, D_data_HSI, train_data_LIDAR, D_data_LIDAR, test_data_HSI, test_data_LIDAR] = data_HSI_LIDATR(train_label, data_HSI, D_label, data_LiDAR, test_label)
    # 构建网络
    model = Model(bands, FM, class_count)
    model.to(device)
    # ###################### 训练 ######################
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load('net_params.pkl'))
    model.eval()
    torch.cuda.synchronize()
    start1 = time.time()
    pred_y = np.empty((len(TestLabel_HSI)), dtype='float32')
    pred_y = torch.from_numpy(pred_y).to(device)
    number = len(TestLabel_HSI) // 300  # //是除以某个数取整，进行分批操作
    for j in range(number):
        print(j)
        temp1 = TestPatch_HSI[j * 300:(j + 1) * 300, :, : , :]
        temp2 = TestPatch_LIDAR[j * 300:(j + 1) * 300, :, : , :]
        temp3 = test_data_HSI[j * 300:(j + 1) * 300, :]
        temp4 = test_data_LIDAR[j * 300:(j + 1) * 300,: ]
        output_HSI_test, output_LiDAR_test, HSI_A_test, HSI_D_test, LiDAR_A_test, LiDAR_D_test, output_test = \
            model(temp3, temp4,D_data_HSI,D_data_LIDAR,temp1, temp2, D_Patch_HSI, D_Patch_LiDAR, K, m,batchsize, D_num,j,D_Label_HSI)
        pred = torch.max(output_test, 1)[1].squeeze()
        pred_y[j * 300:(j + 1) * 300] = pred
        del temp1, temp2, pred, temp3, temp4

    if (j + 1) * 300 < len(TestLabel_HSI):
        l = len(TestLabel_HSI) - TestLabel_HSI[(j + 1) * 300]
        l = l.int()
        temp1 = TestPatch_HSI[(j + 1) * 300:len(TestLabel_HSI), :,:,:]
        temp2 = TestPatch_LIDAR[(j + 1) * 300:len(TestLabel_HSI), :,:,:]
        temp3 = test_data_HSI[(j + 1) * 300:len(TestLabel_HSI), :]
        temp4 = test_data_LIDAR[(j + 1) * 300:len(TestLabel_HSI), :]
        output_HSI_test, output_LiDAR_test,HSI_A_test, HSI_D_test, LiDAR_A_test, LiDAR_D_test, output_test = \
            model(temp3, temp4,D_data_HSI,D_data_LIDAR,temp1, temp2, D_Patch_HSI, D_Patch_LiDAR, K, m,214, D_num,j,D_Label_HSI)
        pred = torch.max(output_test, 1)[1].squeeze()
        pred_y[(j + 1) * 300:len(TestLabel_HSI)] = pred
        del temp1, temp2, pred, temp3, temp4

        accuracy = torch.sum(pred_y == TestLabel_HSI).type(torch.FloatTensor) / TestLabel_HSI.size(0)
        print('| test accuracy: %.6f' % accuracy)
    pred_time = time.time() - start1
    print('testing: {}'.format(pred_time))
    # 将结果按照testlabel的index转为全图尺寸

    pred_y_entropy = np.empty((len(TestLabel_HSI), class_count), dtype='float32')
    pred_y_entropy = torch.from_numpy(pred_y_entropy).to(device)

    temp = np.reshape(test_label, (height*width))
    index = np.where(temp)[0]

    temp_zero = np.zeros(len(temp))
    pred_y = pred_y + 1
    temp_zero[index] = pred_y.cpu().numpy()
    result = np.reshape(temp_zero, [height, width])

    temp_zero = np.zeros((len(temp), class_count))
    temp_zero[index] = pred_y_entropy.cpu().numpy()
    result_entropy = np.reshape(temp_zero, [height, width, class_count])

    sio.savemat("output.mat", {'output': result})
    sio.savemat("result_entropy.mat", {'result_entropy': result_entropy})


    # 精度计算
    pred_y = pred_y - 1
    OA_temp = torch.sum(pred_y == TestLabel_HSI).type(torch.FloatTensor) / TestLabel_HSI.size(0)
    OA = torch.sum(pred_y == TestLabel_HSI).type(torch.FloatTensor) / TestLabel_HSI.size(0)

    Classes = class_count
    EachAcc = np.empty(Classes)

    for i in range(Classes):
        cla = i
        right = 0
        sum = 0

        for j in range(len(TestLabel_HSI)):
            if TestLabel_HSI[j] == cla:
                sum += 1
            if TestLabel_HSI[j] == cla and pred_y[j] == cla:
                right += 1

        EachAcc[i] = right.__float__() / sum.__float__()

    print(OA)
    print(EachAcc)

    torch.cuda.synchronize()

    Final_OA = OA

    print('The OA is: ', Final_OA)
