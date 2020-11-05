import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from AFKMO_classification.cnn1d.pro_data import X, y

EPOCH = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=666)
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
train_dataset = Data.TensorDataset(X_train, y_train)
test_dataset = Data.TensorDataset(X_test,y_test)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=6798, shuffle=True)

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


cnn = cnn1d()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    print("==>training......")
    for step, (x, y) in enumerate(train_loader):
        b_x=x.unsqueeze(1)
        b_x = b_x.float()
        output = cnn(b_x)
        output = output.float()
        b_y = Variable(y.long()).squeeze()
        loss = criterion(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for step, (x, y) in enumerate(test_loader):
        x_test=x.unsqueeze(1)
        x_test = x_test.float()
        test_output = cnn(x_test)
        test_output = test_output.float()
        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
        y = y.numpy()
        y = y.reshape(1, -1).squeeze()
        acc = (pred_y == y).sum() / y.size
        print('Epoch:', epoch + 1, '|train loss:%.4f' % loss.item(), '|test acc:%.4f' % acc)

torch.save(cnn.state_dict(), './cnn_param.pkl')

