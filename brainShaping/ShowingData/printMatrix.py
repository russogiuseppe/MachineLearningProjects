import numpy as np


X_TrainSet = np.load('/home/giuseppe/ml-project/data/X_train.npy')
brainCollection = np.reshape(X_TrainSet, (-1, 176, 208, 176))
youngBrain = brainCollection[2, 37, :, :]
oldBrain = brainCollection[4, 37, :, :]
middleAgeBrain = brainCollection[53, 37, :, :]

for i in range(0,100):
    for j in range(0,207):
        print youngBrain[i][j]