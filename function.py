#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from sklearn.neighbors import NearestNeighbors
def ImR(y):
    Imr = []
    for i in range(y.shape[1]):
        count1 = np.sum(y[:, i])
        count0 = y.shape[0] - count1
        Imr.append(max(count1, count0) / (min(count1, count0) + 1e-6))  # 避免除以零
    return Imr
def Imbalance(y):
    countmatrix = np.sum(y, axis=0)
    maxcount = np.max(countmatrix)
    ImbalanceRatioMatrix = maxcount / (countmatrix + 1e-6)  # 避免除以零
    MeanIR = np.mean(ImbalanceRatioMatrix)
    return ImbalanceRatioMatrix, MeanIR, countmatrix
def Labeltype(y):
    ImbalanceRatioMatrix, MeanIR, _ = Imbalance(y)
    DifferenceImbalanceRatioMatrix = ImbalanceRatioMatrix - MeanIR
    MinLabelIndex = np.where(DifferenceImbalanceRatioMatrix > 0)[0]
    MajLabelIndex = np.where(DifferenceImbalanceRatioMatrix <= 0)[0]
    return MinLabelIndex, MajLabelIndex
def Scumble(y):
    ImbalanceRatioMatrix,MeanIR,_=Imbalance(y)
    DifferenceImbalanceRatioMatrix=[i-MeanIR for i in ImbalanceRatioMatrix]
    count=0
    for i in range(y.shape[1]):
        count+=math.pow(DifferenceImbalanceRatioMatrix[i],2)/(y.shape[1]-1)
    ImbalanceRatioSigma=math.sqrt(count)
    CVIR=ImbalanceRatioSigma/MeanIR
    SumScumble=0
    Scumble_i=[]
    for i in range(y.shape[0]):
        count=0
        prod=1
        SumIRLbl=0
        for j in range(y.shape[1]):
            IRLbl=1
            if y[i,j]==1:
                IRLbl=ImbalanceRatioMatrix[j]
                SumIRLbl+=IRLbl
                prod*=IRLbl
                count+=1
        if count==0:
            Scumble_i.append(0)
        else:
            IRLbl_i=SumIRLbl/count
            Scumble_i.append(1.0-((1.0/IRLbl_i) * math.pow(prod, 1.0/count)))
    scumble=sum(Scumble_i)/X.shape[0]
    return Scumble_i,scumble,CVIR
def CalcuNN(df1,n_neighbor):
    nbs=NearestNeighbors(n_neighbors=n_neighbor,metric='euclidean',algorithm='kd_tree').fit(df1)
    euclidean,indices= nbs.kneighbors(df1)
    return euclidean,indices
def calweight(df1, df2):
    n_neighbors = 5
    # 假设Labeltype和Imbalance是先前已经定义好的函数
    MinLabelindex, MaxLabelindex = Labeltype(df2)
    ImbalanceRatioMatrix, MeanIR, countmatrix = Imbalance(df2)
    
    C = np.zeros((df1.shape[0], df2.shape[1]))
    nbs = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean', algorithm='kd_tree').fit(df1)
    euclidean, indices = nbs.kneighbors(df1)

    for tail_label in MinLabelindex:
        for i in range(df1.shape[0]):
            if df2[i, tail_label] == 0:
                continue
            count1 = 0
            for j in indices[i, 1:]:
                if df2[i, tail_label] != df2[j, tail_label]:
                    count1 += 1
            C[i, tail_label] = count1 / n_neighbors
    
    W = np.zeros(df1.shape[0])
    tem = np.zeros((df2.shape[0], df2.shape[1]))

    for j in range(df2.shape[1]):    
        SumC = 0.0
        c = 0
        for i in range(df2.shape[0]):
            if C[i, j] < 1 and C[i, j] != 0:
                SumC += C[i, j]
                c += 1
        if SumC != 0.0 and c != 0:
            for i in range(df2.shape[0]):
                if C[i, j] < 1 and C[i, j] != 0:
                    tem[i, j] = C[i, j] / SumC
#         else:
#             for i in range(df2.shape[0]):  # 修正
#                 tem[i, j] = 0

    SumW = 0
    for i in range(df2.shape[0]):
        for j in MinLabelindex:
            if tem[i, j] != 0:
                W[i] += tem[i, j]
        W[i] += 1  # 每个值加1
        SumW += W[i]

    return W.tolist()  # 将numpy数组转换为list并返回
def limb(df1, df2):
    ImrMatrix = ImR(df2)
    n_neighbors = 5
    MinLabelindex, MaxLabelindex = Labeltype(df2)
    ImbalanceRatioMatrix, MeanIR, countmatrix = Imbalance(df2)

    C = np.zeros((df1.shape[0], df2.shape[1]))
    count = 0
    nbs = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean', algorithm='kd_tree').fit(df1)
    euclidean, indices = nbs.kneighbors(df1)

    for tail_label in MinLabelindex:
        for i in range(df1.shape[0]):
            if df2[i, tail_label] == 0:
                continue
            count1 = 0
            for j in indices[i, 1:]:
                if df2[i, tail_label] != df2[j, tail_label]:
                    count1 += 1
            C[i, tail_label] = count1 / n_neighbors

    row_sums_list =C.sum(axis=1).tolist()
    W = np.zeros(df1.shape[0])
    tem = np.zeros((df2.shape[0], df2.shape[1]))

    for j in range(df2.shape[1]):    
        SumC = 0.0
        c = 0
        for i in range(df2.shape[0]):
            if C[i, j] < 1 and C[i, j] != 0:
                SumC += C[i, j]
                c += 1
        if SumC != 0.0 and c != 0:
            for i in range(df2.shape[0]):
                if C[i, j] < 1 and C[i, j] != 0:
                    tem[i, j] = C[i, j] / SumC
        else:
            tem[i, j] = 0

    SumW = 0
    for i in range(df2.shape[0]):
        for j in range(df2.shape[1]):
            if tem[i, j] != 0:
                W[i] += tem[i, j]
        SumW += W[i]

    return row_sums_list
def minority_instance(y):
    ImbalanceRatioMatrix,MeanIR,countmatrix=Imbalance(y)
    imbalance_ratios = ImR(y)
#     high_imbalance_indices = sorted(range(len(imbalance_ratios)), key=lambda i: imbalance_ratios[i], reverse=True)[:k]
    MinLabelindex, _ = Labeltype(y)
    rows_with_ones = np.any(y[:, MinLabelindex] == 1, axis=1)
    result_indices = np.where(rows_with_ones)[0]
    return result_indices
def add_samples(data, labels,minority_indices):
    extra_samples_count = int(0.1 * data.size(0))
    extra_indices = np.random.choice(minority_indices, extra_samples_count, replace=False)
    extra_data = train_dataset[extra_indices][0]  # 获取额外样本的数据
    extra_labels = train_dataset[extra_indices][1]  # 获取额外样本的标签

    data = torch.cat((data, extra_data), dim=0)
    labels = torch.cat((labels, extra_labels), dim=0)
    return  data ,labels
def get_samples_from_indices(dataset, indices):
    x_samples = []
    y_samples = []

    for index in indices:
        _, x, y = dataset[index]  # 获取数据和标签，忽略返回的索引
        x_samples.append(x)
        y_samples.append(y)  # 直接添加y

    # 如果x_samples为空，则返回None或适当的默认值
    if not x_samples:
        return None, None

    # 将列表转换为Tensor
    x_samples = torch.stack(x_samples)
    y_samples = torch.stack(y_samples)  # 使用torch.stack堆叠多标签向量

    return x_samples, y_samples

