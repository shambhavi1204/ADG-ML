import numpy as np
import pandas as pd
import matplotlib as plt


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

data = pd.read_csv(r"C:\Users\sinha\Downloads\iris.csv", names=names)
data = data.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],[0, 1, 2])

data = data.iloc[1::,:]
data = data.sample(frac = 1)


X = data.drop("Class",axis=1)
Y = data['Class']


X = X.to_numpy()
Y = Y.to_numpy()
Y = Y.reshape(150,1)


#Y = Y[1::,:]
#X = X[1::,:]
#print(type(X))
'''
X_mn =np.mean(X,axis=0)
X_sd=np.std(X,axis=0)
X=(X-X_mn)/X_sd
Y_mn=np.mean(Y)
Y_sd=np.std(Y)
Y = (Y-Y_mn)/Y_sd
'''


X_train = X[0:100,:]
Y_train = Y[0:100,:]

X_test = X[101:150,:]
Y_test = Y[101:150,:]

#print(X_train.shape)
#print(X_test.shape)

X_train = X_train.astype(np.float)

w1 = np.random.randn(4,1)
w1 = w1.astype(np.float)
b1 = 0
w2 = np.random.randn(4,1)
w2 = w2.astype(np.float)
b2 = 0

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z))



Z1 = np.dot(X_train,w1) + b1
A1 = ReLU(Z1)
print(A1)
A2 = softmax(A1)
print(A2)


