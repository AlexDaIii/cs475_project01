from data import load_data
from models import Perceptron
import numpy as np
import dataset as ds

fileName = "datasets/bio.train"
X, Y = load_data(fileName)

X = X.todense()
Y = Y.reshape(len(Y), 1)
print(X.shape)

print(Y[78])
print(Y[77])
print(Y[79])
print(Y[225])

# X = ds.data_preprocessing_project01(X, fileName)
X = ds.normalize(X)

print(np.amin(X))
print(np.amax(X))

model = Perceptron()
W = model.fit(X, Y, None, n_epoch=1, stop=225)
