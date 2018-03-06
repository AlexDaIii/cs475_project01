from data import load_data
from models import Perceptron
import numpy as np

fileName = "datasets/bio.train"
X, Y = load_data(fileName)

X = X.todense()
Y = Y.reshape(len(Y), 1)

# X = ds.data_preprocessing_project01(X, fileName)
# X = ds.normalize(X)
# Y = ds.data_preprocess_y(Y)

print(X.shape)

model = Perceptron()
W = model.fit(X, Y, n_epoch=5, batch_size=np.size(X, 0))


