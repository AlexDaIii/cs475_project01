import numpy as np
import math

# array = np.array([[1, 2, 3],[4, 2, 5],[6, 2, 7],[8, 2, 9],[1, 2, 3],[4, 2, 5],[6, 2, 7],[8, 2, 9]])
array = np.ones((15, 3))
array[0, 0] = 2
sigma = np.std(array, 0)
# batch = 10
# k = int(math.ceil(np.size(array, 0)/batch))
# for i in range(k-1):
#     print(array[i*batch:(i+1)*batch, :])
#     print()
# print(array[(k-1) * batch:, :])

col = 0
while col < np.size(array, 1):
    if sigma[col] == 0:
        array = np.delete(array, col, 1)
        sigma = np.delete(sigma, col, 0)
        col -= 1
    col += 1

print()
print(array)
