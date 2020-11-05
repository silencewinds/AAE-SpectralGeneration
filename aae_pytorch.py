import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn
import torch.nn.functional as nn
import torch.optim as optim
import os
import utils
from torch.autograd import Variable
from tqdm import tqdm
from config import *
from data_loader import *
from utils import *


#梯度请零
def reset_grad():
    Q.zero_grad()
    P.zero_grad()
    D.zero_grad()

#取数据
def sample_X(size):
    X = m_data.next_batch(size)
    X = Variable(torch.from_numpy(X).type(torch.DoubleTensor))
    return X

torch.set_default_dtype(torch.double)
m_data = data_loader(np_data)       #m_data就是处理好的直接使用的数据

X_dim = m_data.get_data().shape[1]  # 1*3522

pre_loss = 1
cnt = 0
label=0
switch=False                        #切换训练集继续训练（小量强样本纠正）

#编码器，得到生成的z
Q = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim_0),
    torch.nn.BatchNorm1d(h_dim_0, momentum=0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim_0, h_dim_1),
    torch.nn.BatchNorm1d(h_dim_1, momentum=0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim_1, h_dim_2),
    torch.nn.BatchNorm1d(h_dim_2, momentum=0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim_2, h_dim_3),
    torch.nn.BatchNorm1d(h_dim_3, momentum=0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim_3, z_dim),
)

#解码器，由z重构x
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

#判别器，对生成的z二分类（判断正误）
D = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim_3),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim_3, 1),      #一维数据，打分的大小
    torch.nn.Sigmoid()
)

if switch:
    label = 20
    m_data = data_loader("np_data/m6_5_10.npy")     #改变m_data
    P.load_state_dict(torch.load("models/m6_1_5/P_params_" + str(label) + ".pkl"))
    Q.load_state_dict(torch.load("models/m6_1_5/Q_params_" + str(label) + ".pkl"))
    D.load_state_dict(torch.load("models/m6_1_5/D_params_" + str(label) + ".pkl"))

Q_solver = optim.Adam(Q.parameters(), lr=lr)
P_solver = optim.Adam(P.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)


for it in tqdm(range(epoches)):

    X = sample_X(mb_size)

    """ Reconstruction phase """
    #对自编码器部分优化
    z_sample = Q(X)
    X_sample = P(z_sample)

    recon_loss = nn.binary_cross_entropy(X_sample, X)

    recon_loss.backward()
    P_solver.step()
    Q_solver.step()
    reset_grad()

    """ Regularization phase """
    #训练判别器
    z_real = Variable(torch.randn(mb_size, z_dim))
    z_fake = Q(X)

    D_real = D(z_real)
    D_fake = D(z_fake)

    D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))

    D_loss.backward()
    D_solver.step()
    reset_grad()

    #训练优化生成器（自编码器中的编码器）
    z_fake = Q(X)
    D_fake = D(z_fake)

    G_loss = -torch.mean(torch.log(D_fake))

    G_loss.backward()
    Q_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:   #每XX次迭代计算各损失值，并存储一下生成结果；
        pre_loss = recon_loss.item()
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'.format(it, D_loss.item(), G_loss.item(), recon_loss.item()))

        #samples = P(z_real).data.cpu().numpy()[:128]          #生成
        samples = P(z_real).data.cpu().numpy()

        if cnt % 5 == 0:   #每XX次计算损失值，进行1次生成测试降维观察并保存参数；
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if switch:
                torch.save(P.state_dict(), model_path + '/P_params_' +str(label)+'后少量强样本训练纠正_' +str(cnt) + '.pkl')
                torch.save(Q.state_dict(), model_path + '/Q_params_' +str(label)+'后少量强样本训练纠正_' +str(cnt) + '.pkl')
                torch.save(D.state_dict(), model_path + '/D_params_' + str(label) + '后少量强样本训练纠正_' + str(cnt) + '.pkl')
                data_generator(model_path + '/P_params_' +str(label)+'后少量强样本训练纠正_' +str(cnt) + '.pkl', cnt,label)
                plt_tsne(predict_np_path + "_" +str(label)+'后少量强样本训练纠正',cnt,label)
            else:
                torch.save(P.state_dict(), model_path + '/P_params_' + str(cnt) + '.pkl')
                torch.save(Q.state_dict(), model_path + '/Q_params_' + str(cnt) + '.pkl')
                torch.save(D.state_dict(), model_path + '/D_params_' + str(cnt) + '.pkl')
                data_generator(model_path + '/P_params_' + str(cnt) + '.pkl', cnt,0)
                plt_tsne(cnt=cnt,label=0)

        fig = plt.figure()
        for i, sample in enumerate(samples[:10]):
            plt.plot(sample * 100)
            plt.plot(sample)

        if not os.path.exists(out_img_path):
            os.makedirs(out_img_path)

        plt.savefig(out_img_path + '/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)

