import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np

class cnn1d(nn.Module):
    def __init__(self):
        super(cnn1d, self).__init__()
        self.layer1=nn.Sequential(
                nn.Conv1d(1,32,20,stride=5,padding=8),
                nn.ReLU(True),
                nn.MaxPool1d(8),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 20, stride=5, padding=4),
            nn.ReLU(True),
            nn.MaxPool1d(8),
        )
        self.out=nn.Sequential(
            nn.Linear(1*2*64,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,5),
        )
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=x.view(x.size(0),1*2*64)
        x=self.out(x)
        return x

cnn=cnn1d()
cnn.load_state_dict(torch.load('./cnn_param.pkl'))
x_test=np.load('C:/Users/Administrator/Desktop/AAE生成光谱实验(改)/predict_np/A_45.npy')
x_test=torch.from_numpy(x_test)
x_test = x_test.unsqueeze(1)
x_test = x_test.float()
test_output = cnn(x_test)
test_output = test_output.float()
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
y = np.zeros((1000))
y = y.reshape(1, -1).squeeze()
acc = (pred_y == y).sum() / y.size
print('分类准确率为:%.4f%%'%(acc*100))