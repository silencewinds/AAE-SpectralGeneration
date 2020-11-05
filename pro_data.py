import matplotlib
# matplotlib.use('Agg')
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from config import *
#从mat文件中读取数据，以二进制文件形式存储
def data_filter(mat_filename=m_data, num_samples=num_samples):
    m_data = np.array(loadmat(mat_filename)['P1'])
    print(type(m_data))
    fit_data = []
    print('data_shape:', m_data.shape)

    for i in tqdm(range(num_samples)):
        fit_data.append(Normalization(m_data[i]))

    fit_data = np.array(fit_data)
    #fit_data=fit_data[:1000,:]
    print(fit_data.shape)
    np.save(np_data, fit_data)

#归一化处理
def Normalization(x):
    '''
    Normalized the array x
    '''
    mx = max(x)
    mn = min(x)
    return [(float(i) - mn) / (mx - mn) for i in x]

#加载数据
def load_np_data(np_filename=np_data):
    return np.load(np_data)


# def data_cat():
#     fit_data = []
#     for i in range(5):
#         print(data_file_path + "M" + str(i) + "_15.mat")
#         m_data = np.array(loadmat(data_file_path + "M" + str(i) + "_15.mat")['P1'])
#         for i in tqdm(range(500)):
#             fit_data.append(Normalization(m_data[i]))
#     perm = np.arange(2500)
#     np.random.shuffle(perm)
#     np.save("total_M_15", fit_data[perm])


def draw_data(m_data):
    plt.plot(m_data)


if __name__ == '__main__':
    data_filter()
    # m_data = np.load("C:/Users/liweiyu/Desktop/WDMS115.npy")
    # for i in range(64):
    #     draw_data(m_data[i])
    #     plt.show()

