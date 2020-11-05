import os
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import imageio
import numpy as np
from config import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import csv



def data_generator(P_model_name, cnt,label):                       #生成器生成
    X_dim = 3522  # 1*3522
    # Decoder
    P = torch.nn.Sequential(
        torch.nn.Linear(z_dim, h_dim_3),
        torch.nn.BatchNorm1d(h_dim_3, momentum=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim_3, h_dim_2),
        torch.nn.BatchNorm1d(h_dim_2, momentum=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim_2, h_dim_1),
        torch.nn.BatchNorm1d(h_dim_1, momentum=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim_1, h_dim_0),
        torch.nn.BatchNorm1d(h_dim_0, momentum=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim_0, X_dim),
        torch.nn.Sigmoid()
    )
    P.load_state_dict(torch.load(P_model_name))

    predict_data = []
    for i in range(5):      #共生成100个样本
        z_real = Variable(torch.randn(mb_size, z_dim))

        samples = P(z_real).data.cpu().numpy()

        fig = plt.figure()

        for n, sample in enumerate(samples):
            predict_data.append(sample)
            plt.plot(sample)

        if not os.path.exists(predict_img_path):
            os.makedirs(predict_img_path)
        if i == 1:           #存取第一个样本的图片
            plt.savefig(predict_img_path + '/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
            plt.close(fig)
    if label!=0:
        np.save(predict_np_path + "_" +str(label)+'后少量强样本训练纠正_'+str(cnt) + ".npy", predict_data)
    else:
        np.save(predict_np_path + "_" + str(cnt) + ".npy", predict_data)

def convert2gif(file_path=out_img_path):
    images = []
    filenames = sorted((fn for fn in os.listdir(file_path) if fn.endswith('.png')))
    for filename in filenames:
        images.append(imageio.imread(os.path.join(file_path, filename)))
    imageio.mimsave(data_type + 'model_out.gif', images, duration=0.15)


def plt_tsne(predict_data_path=predict_np_path, cnt=0,label=0):
    ori_data = np.load("np_data/m6_15.npy")
    predict_data = np.load(predict_data_path +"_" +str(cnt) + ".npy")

    color = np.squeeze(['blue'] * ori_data.shape[0] + ['red'] * predict_data.shape[0])

    tsne = TSNE(init='pca')
    predict_tsne = tsne.fit_transform(np.append(ori_data, predict_data, axis=0))
    #predict_tsne = tsne.fit_transform(predict_data)

    pca = PCA(n_components=2)
    predict_pca = pca.fit_transform(np.append(ori_data, predict_data, axis=0))

    print("Org data shape is {} x {}. ""Embedded data dimension is {}".format(predict_data.shape[0], predict_data.shape[1], predict_tsne.shape[-1]))
    plt.figure()
    plt.subplot(121)
    plt.scatter(predict_tsne[:, 0], predict_tsne[:, 1], c=color, s=0.5, alpha=0.7)
    plt.subplot(122)
    plt.scatter(predict_pca[:, 0], predict_pca[:, 1], c=color, s=0.5, alpha=0.7)

    #从数据中取出点的XY坐标点数值并保存
    # data1 = pd.DataFrame(predict_tsne[:, 0])
    # data1.to_csv('data_x.csv',header = False, index = False)
    # data2 = pd.DataFrame(predict_tsne[:, 0])
    # data2.to_csv('data_y.csv', header=False, index=False)
    # csv_file = csv.reader(open('data_x.csv', 'r'))
    # box=[]
    # for x in csv_file:
    #     box.append(x)
    # print(len(box))

    if label!=0:
        plt.savefig(str(label)+'后少量强样本训练纠正'+str(cnt) + ".png")
    else:
        plt.savefig(str(cnt) + ".png")


# if __name__ == '__main__':
    # plt_tsne("predict_np/m6_1_5",15)
    # ori_data = np.load("predict_np/m6_5_10_45.npy")
    # print(ori_data.shape)
    # data_generator("models/m6_1_5/P_params_20后少量强样本训练纠正_20.pkl",999,20)
    # plt_tsne("predict_np/m6_1_5_20后少量强样本训练纠正",999,20)

    #plt_tsne("predict_np/m6_15",10,0)


#convert2gif(r"E:\学习\光谱\Lamos_local\out_img\WDMS")