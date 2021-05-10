import numpy as np
import pandas as pd
import matplotlib as plt


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

data = pd.read_csv(r"C:\Users\sinha\Downloads\iris.csv", names=names)
data = data.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],[0, 1, 2])

X = data.drop("Class",axis=1)
Y = data['Class']

X = X.to_numpy()
X_data = X[1::,:]
Y = Y.to_numpy()
Y = Y.reshape(151,1)
Y_data = Y[1::,:]
sz1 = X_data.shape

X_train = X_data[0:99,:]
Y_train = Y_data[0:99,:]
X_test = X_data[100:150,:]
Y_test = Y_data[100:150,:]


