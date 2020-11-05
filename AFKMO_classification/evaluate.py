import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.utils.data as Data

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.RNN(input_size=350,hidden_size=64,num_layers=2,batch_first=True)
        self.out=nn.Linear(64,5)
    def forward(self, x):
        r_out,hstate=self.rnn(x,None)
        out=self.out(r_out[:,-1,:])
        return out

rnn=RNN()
rnn.load_state_dict(torch.load('./rnn_param.pkl'))
X_test=np.load('../predict_np/A_10.npy')
X_test=X_test[:,:3500]
X_test=X_test.reshape(-1,10,350)
X_test=torch.from_numpy(X_test)
X_test=X_test.float()
test_output=rnn(X_test)
test_output=test_output.float()
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
box=0
print(X_test.shape[0])
for i in range(X_test.shape[0]):
    if(pred_y[i]==0):
        box+=1;
acc=box/X_test.shape[0]
print('分类准确率为:%.4f%%'%(acc*100))