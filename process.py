from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
from config import *
import torch.nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

m_data = np.array(loadmat(m_data)['P1'])
print('data_shape:', m_data.shape)


#归一化
def Normalization(x):
    mx = max(x)
    mn = min(x)
    return [(float(i) - mn) / (mx - mn) for i in x]
# 图像比对
def picture():
    x = []
    for i in range(3522):
        x.append(i)
    for i in range(m_data.shape[0]):
        plt.plot(x, m_data[i], color='b')
        plt.plot(x, pro_data[i], color='r')
        plt.show()
# 降维比对
def plt_tsne():
    ori_data = m_data           #均使用归一化前的数据作比较
    predict_data = pro_data     #均使用归一化前的数据作比较
    color = np.squeeze(['blue'] * ori_data.shape[0] + ['red'] * predict_data.shape[0])
    tsne = TSNE(init='pca')
    predict_tsne = tsne.fit_transform(np.append(ori_data, predict_data, axis=0))
    pca = PCA(n_components=2)
    predict_pca = pca.fit_transform(np.append(ori_data, predict_data, axis=0))
    print("Org data shape is {} x {}. ""Embedded data dimension is {}".format(predict_data.shape[0],predict_data.shape[1],predict_tsne.shape[-1]))
    plt.figure()
    plt.subplot(121)
    plt.scatter(predict_tsne[:, 0], predict_tsne[:, 1], c=color, s=0.5, alpha=0.7)
    plt.subplot(122)
    plt.scatter(predict_pca[:, 0], predict_pca[:, 1], c=color, s=0.5, alpha=0.7)
    plt.show()
# def look():
#     ori_data =  np.array(loadmat("m_data/m6_5_10.mat")['P1'])
#     predict_data = pro_data[:85,:]
#     color = np.squeeze(['blue'] * ori_data.shape[0] + ['red'] * predict_data.shape[0])
#     tsne = TSNE(init='pca')
#     predict_tsne = tsne.fit_transform(np.append(ori_data, predict_data, axis=0))
#     pca = PCA(n_components=2)
#     predict_pca = pca.fit_transform(np.append(ori_data, predict_data, axis=0))
#     print("Org data shape is {} x {}. ""Embedded data dimension is {}".format(predict_data.shape[0],predict_data.shape[1],predict_tsne.shape[-1]))
#     plt.figure()
#     plt.subplot(121)
#     plt.scatter(predict_tsne[:, 0], predict_tsne[:, 1], c=color, s=0.5, alpha=0.7)
#     plt.subplot(122)
#     plt.scatter(predict_pca[:, 0], predict_pca[:, 1], c=color, s=0.5, alpha=0.7)
#     plt.show()
#均值滤波处理
def Mean_filtering(m_data):
    kernel=5
    pro_data = []
    # fit_data = []
    # for i in tqdm(range(num_samples)):
    #     fit_data.append(Normalization(m_data[i]))
    # fit_data = np.array(fit_data)
    for i in range(m_data.shape[0]):
        tmp=[]
        for x in[m_data[i,j:j+kernel] for j in range(0,3520,kernel)]:
            tmp.append(x.sum()/kernel)
        tmp=torch.Tensor(tmp)
        tmp=tmp.unsqueeze(0)
        tmp=tmp.unsqueeze(1)
        tmp=F.interpolate(tmp,3522,mode='linear')
        tmp=tmp.squeeze(1)
        tmp=tmp.squeeze(0)
        pro_data.append(np.array(tmp))
    return pro_data
#PCA降噪
def PCA_reduction(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    # 进行操作转换后的结果
    x_reduction = pca.transform(X)
    # 进行复原恢复后的结果
    x_restore = pca.inverse_transform(x_reduction)
    return x_restore



pro_data=PCA_reduction(m_data)

# pro_data=Mean_filtering(m_data)
# pro_data=np.array(pro_data)

fit_data = []
for i in tqdm(range(num_samples)):
    fit_data.append(Normalization(pro_data[i]))
fit_data = np.array(fit_data)
print(fit_data.shape)
np.save(np_data, fit_data)

#picture()
plt_tsne()
