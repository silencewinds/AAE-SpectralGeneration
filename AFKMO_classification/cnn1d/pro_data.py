import numpy as np
import scipy.io as sio

a = sio.loadmat('..\dataset\A.mat')  # 6000*3522
f = sio.loadmat('..\dataset\F.mat')  # 6000*3522
k = sio.loadmat('..\dataset\K.mat')  # 6000*3522
m = sio.loadmat('..\dataset\M.mat')  # 6000*3522
o = sio.loadmat('..\dataset\O.mat')  # 3191*3522

def Normalization(x):
    mx=max(x)
    mn=min(x)
    return [(float(i)-mn)/(mx-mn) for i in x]

X = np.vstack((a['P1'], f['P1'], k['P1'], m['P1'],o['P1']))
X=X[:,:3500]
for i in range(27191):
    X[i]=Normalization(X[i])

y1 = np.full((a['P1'].shape[0],), 0)
y2 = np.full((f['P1'].shape[0],), 1)
y3 = np.full((k['P1'].shape[0],), 2)
y4 = np.full((m['P1'].shape[0],), 3)
y5 = np.full((o['P1'].shape[0],), 4)
y = np.hstack((y1, y2, y3, y4, y5)).reshape(-1, 1)
print(X.shape, y.shape)
