from out import activation
import numpy as np

act = activation.Sigmoid()
arr = np.array([[0.458, 0, 1, 0], [0, 10, -10, 0], [-0.00000458, 0, -1, 0], [1e-11, 0, 0, 0]])
print(np.log(act.activation_function(arr)))

act2 = activation.Tanh()
# print(act2.activation_function(arr))

act3 = activation.Relu()
# print(act3.activation_function(arr))
