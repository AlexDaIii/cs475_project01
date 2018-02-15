import numpy as np
from _parse import load_wine_csv
from cost_function import CrossEntropy

cost_at_initial_W = 0.693
grad_at_initial_W = np.array([-0.1000, -12.0092, -11.2628])

initial_W = np.zeros((3, 1))
test_W = np.array([[-24], [0.2], [0.2]])

cost_at_test_W = 0.218
grad_at_test_W = np.array([0.043, 2.566, 2.647])

X, Y = load_wine_csv("_data.txt")

X = np.append(np.ones((len(X), 1)), X, 1)

per = CrossEntropy()
cost, grad = per.cost(initial_W, X, Y)

print("Cost at initial W (zeros): " + str(cost))
print("Expected cost at initial W (zeros): " + str(cost_at_initial_W))
print()
print("Grad at initial W (zeros): " + str(grad.T))
print("Expected gradient at initial W (zeros): " + str(grad_at_initial_W))

cost, grad = per.cost(test_W, X, Y)

print()

print("Cost at test W : " + str(cost))
print("Expected cost at test W : " + str(cost_at_test_W))
print()
print("Grad at test W : " + str(grad.T))
print("Expected gradient at test W : " + str(grad_at_test_W))
