import torch
from torch import nn
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
            nn.Linear(32,7),
        )
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=x.view(x.size(0),1*2*64)
        x=self.out(x)
        return x
