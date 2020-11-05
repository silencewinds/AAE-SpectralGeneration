import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as Data

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn1=nn.RNN(input_size=350,hidden_size=64,num_layers=2,batch_first=True)
        self.rnn2=nn.RNN(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        self.out=nn.Linear(64,7)
    def forward(self, x):
        r_out1, hstate1 = self.rnn1(x,None)
        r_out2, hstate2 = self.rnn2(r_out1, hstate1)
        r_out3, hstate3 = self.rnn2(r_out2, hstate2)
        r_out4, hstate4 = self.rnn2(r_out3, hstate3)
        r_out5, hstate5 = self.rnn2(r_out4, hstate4)
        r_out6, hstate6 = self.rnn2(r_out5, hstate5)
        out=self.out(r_out6[:,-1,:])
        return out

rnn=RNN()
rnn.load_state_dict(torch.load('./rnn_param.pkl'))
for str in['5','10','15','20','25','30','35','40','45']:
    X_test=np.load('../predict_np/m6_15_'+str+'.npy')
    #print(X_test.shape)
    X_test=X_test[:,:3500]
    X_test=X_test.reshape(-1,10,350)
    X_test=torch.from_numpy(X_test)
    X_test=X_test.float()
    test_output=rnn(X_test)
    test_output=test_output.float()
    pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
    box=0
    for i in range(X_test.shape[0]):
        if(pred_y[i]==6):
            box+=1;
    acc=box/X_test.shape[0]
    print('分类准确率为:%.4f%%'%(acc*100))