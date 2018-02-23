# cs475 HW 1
In this assignment, we implemented a sum of features classifier and the perceptron algorithm. In the sum of features
classifier, it just adds the first half of the features and the second half of the features and compares them. If the
first half is greater than or equal to the second half, then we predict 1, else, predict 0.
In the perceptron algorithm, we implement the cost function from class. There is a log loss cost function in the
cost_function.py; however this is not the cost function we use for the Perceptron, as I learned form class. I just
wanted the cost function so I could display it in the Trainer class. It was pretty simple to change from log loss to 
zero one loss since I had made the trainer take in any cost function to do gradient descent to minimize it. The aim of
this was to make the code modular so I wouldn't have to rewrite a gradient descent and a trainer. Also, so it would be 
sort of like an unoptimized, less well structured, purely CPU version of Tensorflow. 

## Authors
Alexander Chang
JHED: achang56
