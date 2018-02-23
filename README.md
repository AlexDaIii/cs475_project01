# cs475 HW 1

## Author
Alexander Chang
JHED: achang56

## About this assignment
In this assignment, we implemented a sum of features classifier and the perceptron algorithm. In the sum of features
classifier, it just adds the first half of the features and the second half of the features and compares them. If the
first half is greater than or equal to the second half, then we predict 1, else, predict 0.
In the perceptron algorithm, we implement the cost function from class. There is a log loss cost function in the
cost_function.py; however this is not the cost function we use for the Perceptron, as I learned form class. I just
wanted the cost function so I could display it in the Trainer class. It was pretty simple to change from log loss to 
zero one loss since I had made the trainer take in any cost function to do gradient descent to minimize it. The aim of
this was to make the code modular so I wouldn't have to rewrite a gradient descent and a trainer. Also, so it would be 
sort of like an unoptimized, less well structured, purely CPU version of Tensorflow. 

## Discussion of Results
I didn't include the results of the sumoffeatures because I pretty much got the same as the other accuracies posted on
Piazza. The Perceptron algorithm's accuracy depended greatly on if I standardized the data or not (the normalization
included in the dataset.py does not work but should be fixed by next assignment). I know we didn't need it for this
assignment (because it wasn't mentioned in the pdf and the test accuracy also wasn't happy with standardizing the input)
but I kinda wanted to see what would happen if we did standardize the data. Somehow, it got a lower accuracy for most 
of the datasets - for some reason that is unknown to me. 
Another thing about my implementation is that it somehow gets one less correct than the results posted on Piazza. I
predict 1 if Wx >= 0, so I really don't know why the model sometimes is only 1 off. 

## Lessons Learned
1. If using logistic regression, remember to always standardize/normaize or you get like no gradient and you are sad. 
2. Remember to use the correct loss function, trying to do perception with log loss is not optimal. 
3. Print out everything.
4. Don't keep initialization massive identity matrices - it makes the code very slow
5. Find faster ways to loop through all the rows of a matrix to get rid of no/low variance features

## Results
bio.train - 284 features, 284 useful, 2000 training examples

    max: 44.71017781221589, min: -1.0191839793138975
    
    WITHOUT Standardization
    Accuracy Train: 0.991000 (1982/2000)
    Accuracy Dev: 0.965000 (193/200)
    Accuracy Test: 0.000000 (0/222)

    WITH Standardization
    Accuracy Train: 0.920000 (1840/2000)
    Accuracy Dev: 0.755000 (151/200)
    Accuracy Test: 0.000000 (0/222)  - the ys are all -1

easy.train - 10 features, 10 useful, 900 examples

    max: 3.771062324022551, min: -3.629691951551705
    
    WITHOUT Standardization
    Accuracy Train: 1.000000 (900/900)
    Accuracy Dev: 1.000000 (100/100)

    WITH Standardization
    Accuracy: 1.000000 (900/900)
    Accuracy: 1.000000 (100/100)

finance.train - 46 features, 46 useful, 550 training examples,

    max: 17.95894130677154, min: -3.349958540373614

    WITHOUT Standardization
    Accuracy: 0.643636 (354/550)
    Accuracy: 0.866667 (26/30)
    Accuracy: 0.000000 (0/73)

    WITH Standardization
    Accuracy: 0.843636 (464/550)
    Accuracy: 0.700000 (21/30)
    Accuracy: 0.000000 (0/73)

hard.train - 94 features, 94 useful, 900 examples

    max: 4.38436264109483, min: -4.726133990766483

    WITHOUT Standardization
    Accuracy: 0.534444 (481/900)
    Accuracy: 0.540000 (54/100)

    WITH Standardization
    Accuracy: 0.548889 (494/900)
    Accuracy: 0.450000 (45/100)

nlp.train - 54471 features, 1000 examples

    max: 31.606961258558965, min: -0.92040958659559
    very slow if we have regularization - probably because we initialize an identity matrix that is like 54000 big

    WITHOUT Standardization
    Accuracy: 0.998000 (998/1000)
    Accuracy: 0.786667 (236/300)
    Accuracy: 0.000000 (0/300)

    WITH Standardization
    NA - too slow

speech.train - 617 features, 617 useful, 400 training examples

    max: 17.972433664476377, min: -6.983893867834724
    
    WITHOUT Standardization
    Accuracy: 0.925000 (370/400)
    Accuracy: 0.850000 (85/100)
    Accuracy: 0.000000 (0/99)

    WITH Standardization
    Accuracy: 0.992500 (397/400)
    Accuracy: 0.900000 (90/100)
    Accuracy: 0.000000 (0/99)

vision.train - 19 features, 18 useful, 500 training examples - there is 1 repeat one

    max: 11.668342100980727, min: -5.723733151147001
    
    WITHOUT Standardization
    Accuracy: 0.994000 (497/500)
    Accuracy: 0.987500 (79/80)
    Accuracy: 0.000000 (0/80)

    WITH Standardization
    Accuracy: 0.996000 (498/500)
    Accuracy: 0.987500 (79/80)
    Accuracy: 0.000000 (0/80)
