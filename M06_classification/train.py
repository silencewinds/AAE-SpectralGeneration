import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from M06_classification.pro_data import X,y

EPOCH=50
BATCH_SIZE=16
TIME_STEP=10
INPUT_SIZE=350
LEARNING_RATE=0.0001

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=666)
X_train=torch.from_numpy(X_train)
X_test=torch.from_numpy(X_test)
y_train=torch.from_numpy(y_train)
y_test=torch.from_numpy(y_test)
y_test = y_test.numpy()
y_test=y_test.reshape(1,-1).squeeze()
train_dataset = Data.TensorDataset(X_train, y_train)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn1=nn.RNN(input_size=INPUT_SIZE,hidden_size=64,num_layers=2,batch_first=True)
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
optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
criterion=nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    print("==>training......")
    for step,(x,y) in enumerate(train_loader):
        b_x=x.view(-1,10,350)
        b_x=b_x.float()
        output=rnn(b_x)
        output=output.float()
        b_y=Variable(y.long()).squeeze()
        loss=criterion(output,b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    X_test=X_test.view(-1,10,350)
    X_test=X_test.float()
    test_output=rnn(X_test)
    test_output=test_output.float()
    pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
    acc=(pred_y==y_test).sum()/y_test.size
    print('Epoch:',epoch+1,'|train loss:%.4f'%loss.item(),'|test acc:%.4f'%acc)

torch.save(rnn.state_dict(),'./rnn_param.pkl')

