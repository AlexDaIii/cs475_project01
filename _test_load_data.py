from data import load_data
from models import Perceptron
import numpy as np
import dataset as ds

fileName = "datasets/bio.train"
X, Y = load_data(fileName)

X = X.todense()
Y = Y.reshape(len(Y), 1)

# X = ds.data_preprocessing_project01(X, fileName)
X = ds.normalize(X)

print(np.amin(X))
print(np.amax(X))

model = Perceptron()
W = model.fit(X, Y, None)
