import sio
import random
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data as dataf
import scipy.io as sio
import scipy.io as sio
import os
from tqdm import tqdm
from sklearn import preprocessing

# device = torch.device("cpu" if torch.cuda.is_available() else "cuda:0")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def print_data(gt, class_count):
    gt_reshape = np.reshape(gt, [-1])
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        print('第' + str(i + 1) + '类的个数为' + str(samplesCount))

def get_dataset(dataset):

    data_HSI = []
    data_LiDAR = []
    gt = []
    val_ratio = 0
    class_count = 0
    learning_rate = 0
    max_epoch = 0
    dataset_name = ''
    trainloss_result = []
    LiDAR_bands = 0

    if dataset == 1:
        data_HSI_mat = sio.loadmat('data/2012houston/HSI_data.mat')
        data_HSI = data_HSI_mat['HSI_data']
        data_LiDAR_mat = sio.loadmat('data/2012houston/LiDAR_data.mat')
        data_LiDAR = data_LiDAR_mat['LiDAR_data']

        gt_mat = sio.loadmat('data/2012houston/All_Label.mat')
        gt = gt_mat['All_Label']
        train_gt_mat = sio.loadmat('data/2012houston/Train_Label.mat')
        train_gt = train_gt_mat['Train_Label']

        # 参数预设
        val_ratio = 0.01  # 验证机比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 15  # 样本类别数
        learning_rate = 2e-4  # 学习率
        max_epoch = 500  # 迭代次数
        dataset_name = "Huston2012"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        LiDAR_bands = 1
        pass
    if dataset == 2:
        data_HSI_mat = sio.loadmat('data/Trento/HSI_data.mat')
        data_HSI = data_HSI_mat['HSI_data']
        data_LiDAR_mat = sio.loadmat('data/Trento/LiDAR_data.mat')
        data_LiDAR = data_LiDAR_mat['LiDAR_data']
        gt_mat = sio.loadmat('data/Trento/All_Label.mat')
        gt = gt_mat['All_Label']
        train_gt_mat = sio.loadmat('data/Trento/Train_Label.mat')
        train_gt = train_gt_mat['Train_Label']

        # 参数预设
        val_ratio = 0.01  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
        class_count = 6  # 样本类别数
        learning_rate = 2e-4  # 学习率
        max_epoch = 1000 # 迭代次数
        dataset_name = "Trento"  # 数据集名称
        trainloss_result = np.zeros([max_epoch + 1, 1])
        LiDAR_bands = 1
        pass
    return [data_HSI, data_LiDAR, gt, train_gt,val_ratio, class_count,
            learning_rate, max_epoch, dataset_name, trainloss_result, LiDAR_bands]

def data_standard(data_HSI, data_LiDAR, LiDAR_bands):

    height, width, bands = data_HSI.shape  # 原始高光谱数据的三个维度

    data_HSI = np.reshape(data_HSI, [height * width, bands])  # 将数据转为HW * B
    minMax = preprocessing.MinMaxScaler()
    data_HSI = minMax.fit_transform(data_HSI)  # 这两行用来归一化数据，归一化时需要进行数据转换
    data_HSI = np.reshape(data_HSI, [height, width, bands])  # 将数据转回去 H * W * B

    data_LiDAR = np.reshape(data_LiDAR, [height * width, LiDAR_bands])  # 将数据转为HW * B
    minMax = preprocessing.MinMaxScaler()
    data_LiDAR = minMax.fit_transform(data_LiDAR)  # 这两行用来归一化数据，归一化时需要进行数据转换
    data_LiDAR = np.reshape(data_LiDAR, [height, width, LiDAR_bands])  # 将数据转回去 H * W * B
    return [data_HSI, data_LiDAR]

def gen_model_data(data_HSI, data_LiDAR, patchsize_HSI, patchsize_LiDAR, train_label, test_label, D_label, batchsize):
    height, width, bands = data_HSI.shape
    # ##### 给HSI和LiDAR打padding #####
    # 先给第一个维度打padding，确定打完padding的矩阵的大小后，建立一个[H,W,C]的空矩阵，再用循环给所有维度打padding
    temp = data_HSI[:, :, 0]
    pad_width = np.floor(patchsize_HSI / 2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [h_pad, w_pad] = temp2.shape
    data_HSI_pad = np.empty((h_pad, w_pad, bands), dtype='float32')

    for i in range(bands):
        temp = data_HSI[:, :, i]
        pad_width = np.floor(patchsize_HSI / 2)
        pad_width = np.int(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        data_HSI_pad[:, :, i] = temp2

    data_LiDAR_pad = data_LiDAR
    pad_width2 = np.floor(patchsize_LiDAR / 2)
    pad_width2 = np.int(pad_width2)
    temp = np.pad(data_LiDAR_pad, pad_width2, 'symmetric')
    data_LiDAR_pad = temp

    # #### 构建高光谱的训练集和测试集 #####
    [ind1, ind2] = np.where(train_label != 0)  # ind1和ind2是符合where中条件的点的横纵坐标
    TrainNum = len(ind1)
    TrainPatch_HSI = np.empty((TrainNum, bands, patchsize_HSI, patchsize_HSI), dtype='float32')
    TrainLabel_HSI = np.empty(TrainNum)
    ind3 = ind1 + pad_width  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        # x是打了padding的高光谱，下文的x2是打了padding的LiDAR
        # 取第i个训练patch，取一个立方体
        patch = data_HSI_pad[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                             (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
        TrainPatch_HSI[i, :, :, :] = patch
        patchlabel_HSI = train_label[ind1[i], ind2[i]]
        TrainLabel_HSI[i] = patchlabel_HSI


    [ind1, ind2] = np.where(D_label != 0)  # ind1和ind2是符合where中条件的点的横纵坐标
    DNum = len(ind1)
    D_Patch_HSI = np.empty((DNum, bands, patchsize_HSI, patchsize_HSI), dtype='float32')
    D_Label_HSI = np.empty(DNum)
    ind3 = ind1 + pad_width  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        # x是打了padding的高光谱，下文的x2是打了padding的LiDAR
        # 取第i个训练patch，取一个立方体
        patch = data_HSI_pad[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                             (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
        D_Patch_HSI[i, :, :, :] = patch
        patchlabel_HSI = D_label[ind1[i], ind2[i]]
        D_Label_HSI[i] = patchlabel_HSI


    [ind1, ind2] = np.where(test_label != 0)
    TestNum = len(ind1)
    TestPatch_HSI = np.empty((TestNum, bands, patchsize_HSI, patchsize_HSI), dtype='float32')
    TestLabel_HSI = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = data_HSI_pad[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                             (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
        TestPatch_HSI[i, :, :, :] = patch
        patchlabel_HSI = test_label[ind1[i], ind2[i]]
        TestLabel_HSI[i] = patchlabel_HSI

    print('Training size and testing size of HSI are:', TrainPatch_HSI.shape, 'and', TestPatch_HSI.shape)

    # #### 构建LiDAR的训练集和测试集 #####
    [ind1, ind2] = np.where(train_label != 0)
    TrainNum = len(ind1)
    TrainPatch_LiDAR = np.empty((TrainNum, 1, patchsize_LiDAR, patchsize_LiDAR), dtype='float32')
    TrainLabel_LiDAR = np.empty(TrainNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = data_LiDAR_pad[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1),
                               (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize_LiDAR * patchsize_LiDAR, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize_LiDAR, patchsize_LiDAR))
        TrainPatch_LiDAR[i, :, :, :] = patch
        patchlabel_LiDAR = train_label[ind1[i], ind2[i]]
        TrainLabel_LiDAR[i] = patchlabel_LiDAR


    [ind1, ind2] = np.where(D_label != 0)  # ind1和ind2是符合where中条件的点的横纵坐标
    DNum = len(ind1)
    D_Patch_LiDAR = np.empty((DNum, 1, patchsize_LiDAR, patchsize_LiDAR), dtype='float32')
    D_Label_LiDAR = np.empty(DNum)
    ind3 = ind1 + pad_width2  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        # x是打了padding的高光谱，下文的x2是打了padding的LiDAR
        # 取第i个训练patch，取一个立方体
        patch = data_LiDAR_pad[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1),
                             (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch = np.reshape(patch, (patchsize_LiDAR * patchsize_LiDAR,1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize_LiDAR, patchsize_LiDAR))
        D_Patch_LiDAR[i, :, :, :] = patch
        patchlabel_LiDAR = D_label[ind1[i], ind2[i]]
        D_Label_LiDAR[i] = patchlabel_LiDAR

    [ind1, ind2] = np.where(test_label != 0)
    TestNum = len(ind1)
    TestPatch_LIDAR = np.empty((TestNum, 1, patchsize_LiDAR, patchsize_LiDAR), dtype='float32')
    TestLabel_LiDAR = np.empty(TestNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = data_LiDAR_pad[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize_LiDAR * patchsize_LiDAR, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize_LiDAR, patchsize_LiDAR))
        TestPatch_LIDAR[i, :, :, :] = patch
        patchlabel_LiDAR = test_label[ind1[i], ind2[i]]
        TestLabel_LiDAR[i] = patchlabel_LiDAR

    print('Training size and testing size of LiDAR are:', TrainPatch_LiDAR.shape, 'and', TestPatch_LIDAR.shape)

    # #### 数据转换以及把数据搬到GPU #####
    TrainPatch_HSI = torch.from_numpy(TrainPatch_HSI).to(device)
    TrainLabel_HSI = torch.from_numpy(TrainLabel_HSI) - 1
    TrainLabel_HSI = TrainLabel_HSI.long().to(device)

    D_Patch_HSI = torch.from_numpy(D_Patch_HSI).to(device)
    D_Label_HSI = torch.from_numpy(D_Label_HSI) - 1
    D_Label_HSI = D_Label_HSI.long().to(device)

    TestPatch_HSI = torch.from_numpy(TestPatch_HSI).to(device)
    TestLabel_HSI = torch.from_numpy(TestLabel_HSI) - 1
    TestLabel_HSI = TestLabel_HSI.long().to(device)

    TrainPatch_LiDAR = torch.from_numpy(TrainPatch_LiDAR).to(device)
    TrainLabel_LiDAR = torch.from_numpy(TrainLabel_LiDAR) - 1
    TrainLabel_LiDAR = TrainLabel_LiDAR.long().to(device)

    D_Patch_LiDAR = torch.from_numpy(D_Patch_LiDAR).to(device)
    D_Label_LiDAR = torch.from_numpy(D_Label_LiDAR) - 1
    D_Label_LiDAR = D_Label_LiDAR.long().to(device)

    TestPatch_LIDAR = torch.from_numpy(TestPatch_LIDAR).to(device)
    TestLabel_LiDAR = torch.from_numpy(TestLabel_LiDAR) - 1
    TestLabel_LiDAR = TestLabel_LiDAR.long().to(device)

    return TrainPatch_HSI, TrainPatch_LiDAR, TrainLabel_HSI, TestPatch_HSI, TestPatch_LIDAR, TestLabel_HSI, D_Patch_HSI, D_Label_HSI,D_Patch_LiDAR, D_Label_LiDAR

def data_partition(samples_type, class_count, gt, train_gt, train_ratio, val_ratio, height, width):

    train_rand_idx = []
    train_data_index = []
    test_data_index = []
    D_rand_idx = []
    D_data_index = []
    D_samples = train_ratio   # 当选取每类固定数量样本测试时，选择的字典每类数目
    real_D_samples_per_class = train_ratio
    gt_reshape = np.reshape(gt, [-1])

    if samples_type == 'ratio':     # 每个类别取一定的比例训练
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            rand_idx = random.sample(rand_list,
                                     np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)

        # 将测试集（所有样本，包括训练集）也转为特定形式
        train_data_index = set(train_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - background_idx


        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        # val_data_index = list(val_data_index)
        pass

    if samples_type == 'same_num':  # 取固定数量训练
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = train_ratio
            real_D_samples_per_class = 10
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            if real_train_samples_per_class > samplesCount:
                real_train_samples_per_class = samplesCount
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)

            rand_idx_D = random.sample(rand_list,
                                     real_D_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_D = idx[rand_idx_D[0:real_D_samples_per_class]]
            D_rand_idx.append(rand_real_idx_per_class_D)
        D_rand_idx_1 = np.array(D_rand_idx)
        D_data_index = []
        for c in range(D_rand_idx_1.shape[0]):
            a = D_rand_idx_1[c]
            for j in range(a.shape[0]):
                D_data_index.append(a[j])
        D_data_index = np.array(D_data_index)
        D_data_index = set(D_data_index)

        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)
        train_data_index = set(train_data_index)

        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - background_idx - train_data_index

        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        D_data_index = list(D_data_index)
        pass

    if samples_type == 'fixed':  # 固定的训练集
        gt_train_reshape = np.reshape(train_gt, [-1])
        for i in range(class_count):  # i从0跑到 class_count-1
            idx = np.where(gt_train_reshape == i + 1)[-1]
            train_rand_idx.append(idx)
        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)
        train_data_index = set(train_data_index)


        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            rand_idx_D = random.sample(rand_list,
                                     real_D_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_D = idx[rand_idx_D[0:real_D_samples_per_class]]
            D_rand_idx.append(rand_real_idx_per_class_D)
        D_rand_idx_1 = np.array(D_rand_idx)
        D_data_index = []
        for c in range(D_rand_idx_1.shape[0]):
            a = D_rand_idx_1[c]
            for j in range(a.shape[0]):
                D_data_index.append(a[j])
        D_data_index = np.array(D_data_index)
        D_data_index = set(D_data_index)


        ##将测试集（所有样本，包括训练样本）也转化为特定形式

        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - background_idx

        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        D_data_index = list(D_data_index)


    # 获取训练样本的标签图
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass
    train_label = np.reshape(train_samples_gt, [height, width])

    # 获取测试样本的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass
    test_label = np.reshape(test_samples_gt, [height, width])  # 测试样本图

    # 获取字典样本的标签图
    D_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(D_data_index)):
        D_samples_gt[D_data_index[i]] = gt_reshape[D_data_index[i]]
        pass
    D_label = np.reshape(D_samples_gt, [height, width])


    return [train_label, test_label, D_label]#, val_label

def data_HSI_LIDATR(train_label, data_HSI, D_label, data_LiDAR, test_label):

    height, width, bands = data_HSI.shape
    #HSI
    train_label = np.reshape(train_label, (height * width))
    train_data_HSI = np.reshape(data_HSI, (height * width, bands))
    temp_index = np.where(train_label != 0)[0]
    train_data_HSI = train_data_HSI[temp_index]

    test_label = np.reshape(test_label, (height * width))
    test_data_HSI = np.reshape(data_HSI, (height * width, bands))
    temp_index = np.where(test_label != 0)[0]
    test_data_HSI = test_data_HSI[temp_index]

    D_label = np.reshape(D_label, (height * width))
    D_data_HSI = np.reshape(data_HSI, (height * width, bands))
    temp_index_D = np.where(D_label != 0)[0]
    D_data_HSI = D_data_HSI[temp_index_D]

    # 转到GPU
    train_data_HSI = torch.from_numpy(train_data_HSI).to(device)
    D_data_HSI = torch.from_numpy(D_data_HSI).to(device)
    test_data_HSI = torch.tensor(test_data_HSI).to(device)

    #LIDAR
    train_data_LIDAR = np.reshape(data_LiDAR, (height * width, 1))
    temp_index = np.where(train_label != 0)[0]
    train_data_LIDAR = train_data_LIDAR[temp_index]

    test_data_LIDAR = np.reshape(data_LiDAR, (height * width, 1))
    temp_index = np.where(test_label != 0)[0]
    test_data_LIDAR = test_data_LIDAR[temp_index]

    D_data_LIDAR = np.reshape(data_LiDAR, (height * width, 1))
    temp_index_D = np.where(D_label != 0)[0]
    D_data_LIDAR = D_data_LIDAR[temp_index_D]

    # 转到GPU
    train_data_LIDAR = torch.from_numpy(train_data_LIDAR).to(device)
    D_data_LIDAR = torch.from_numpy(D_data_LIDAR).to(device)
    test_data_LIDAR = torch.tensor(test_data_LIDAR).to(device)

    return train_data_HSI, D_data_HSI, train_data_LIDAR, D_data_LIDAR, test_data_HSI, test_data_LIDAR