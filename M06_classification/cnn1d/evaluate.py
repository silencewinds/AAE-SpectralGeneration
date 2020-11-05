import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from cnn1d_model import cnn1d

cnn=cnn1d()
cnn.load_state_dict(torch.load('./cnn_param.pkl'))
for str in['5','10','15','20','25','30','35','40','45']:
    x_test=np.load('../../predict_np/m6_15_'+str+'.npy')[:,:3500]
    x_test=torch.from_numpy(x_test)
    x_test = x_test.unsqueeze(1)
    x_test = x_test.float()
    test_output = cnn(x_test)
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    box=0
    for i in range(x_test.shape[0]):
        if (pred_y[i] == 6):
            box += 1;
    acc = box / x_test.shape[0]
    print('分类准确率为:%.4f%%'%(acc*100))