python3 classify.py --mode train --algorithm perceptron --model-file trained/bio.perceptron.model --data datasets/bio.train
python3 classify.py --mode test --model-file trained/bio.perceptron.model --data datasets/bio.train --predictions-file predictions/bio.train.predictions
python3 compute_accuracy.py datasets/bio.train predictions/bio.train.predictions