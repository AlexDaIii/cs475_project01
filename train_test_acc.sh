python3 classify.py --mode train --algorithm perceptron --model-file trained/finance.perceptron.model --data datasets/finance.train
python3 classify.py --mode test --model-file trained/finance.perceptron.model --data datasets/finance.train --predictions-file predictions/finance.train.predictions
python3 classify.py --mode test --model-file trained/finance.perceptron.model --data datasets/finance.dev --predictions-file predictions/finance.dev.predictions
python3 classify.py --mode test --model-file trained/finance.perceptron.model --data datasets/finance.test --predictions-file predictions/finance.test.predictions
python3 compute_accuracy.py datasets/finance.train predictions/finance.train.predictions
python3 compute_accuracy.py datasets/finance.dev predictions/finance.dev.predictions
python3 compute_accuracy.py datasets/finance.test predictions/finance.test.predictions