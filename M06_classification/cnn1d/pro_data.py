import numpy as np
import scipy.io as sio

m0 = sio.loadmat('..\dataset\m0_15.mat')['P1']      #2023*3522
m1 = sio.loadmat('..\dataset\m1_15.mat')['P1']      #1134*3522   +900
m2 = sio.loadmat('..\dataset\m2_15.mat')['P1']      #1343*3522   +700
m3 = sio.loadmat('..\dataset\m3_15.mat')['P1']      #1170*3522   +800
m4 = sio.loadmat('..\dataset\m4_15.mat')['P1']      #603*3522    +1400
m5 = sio.loadmat('..\dataset\m5_15.mat')['P1']      #82*3522     +1900
m6 = sio.loadmat('..\dataset\m6_15.mat')['P1']      #20*3522     +1900

def Normalization(x):
    mx=max(x)
    mn=min(x)
    return [(float(i)-mn)/(mx-mn) for i in x]
def up(m,number):
    row_rand_array = np.arange(m.shape[0])
    np.random.shuffle(row_rand_array)
    row_rand = m[row_rand_array[0:number]]
    m=np.vstack((m,row_rand))
    print("扩充后",m.shape)

up(m1,900)
up(m2,700)
up(m3,800)
m4=np.vstack((m4,m4,m4))
print("扩充后",m4.shape)
tmpm5=m5
for i in range(25):
    m5=np.vstack((m5,tmpm5))
print("扩充后",m5.shape)
tmpm6=m6
for i in range(100):
    m6=np.vstack((m6,tmpm6))
print("扩充后",m6.shape)

X = np.vstack((m0,m1,m2,m3,m4,m5,m6))
X=X[:,:3500]
for i in range(11631):
    X[i]=Normalization(X[i])

y0 = np.full((m0.shape[0],), 0)
y1 = np.full((m1.shape[0],), 1)
y2 = np.full((m2.shape[0],), 2)
y3 = np.full((m3.shape[0],), 3)
y4 = np.full((m4.shape[0],), 4)
y5 = np.full((m5.shape[0],), 5)
y6 = np.full((m6.shape[0],), 6)
y = np.hstack((y0, y1, y2, y3, y4, y5, y6)).reshape(-1, 1)
print(X.shape, y.shape)
