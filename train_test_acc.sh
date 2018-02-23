python3 classify.py --mode train --algorithm perceptron --model-file trained/vision.perceptron.model --data datasets/vision.train
python3 classify.py --mode test --model-file trained/vision.perceptron.model --data datasets/vision.train --predictions-file predictions/vision.train.predictions
python3 classify.py --mode test --model-file trained/vision.perceptron.model --data datasets/vision.dev --predictions-file predictions/vision.dev.predictions
python3 classify.py --mode test --model-file trained/vision.perceptron.model --data datasets/vision.test --predictions-file predictions/vision.test.predictions
python3 compute_accuracy.py datasets/vision.train predictions/vision.train.predictions
python3 compute_accuracy.py datasets/vision.dev predictions/vision.dev.predictions
python3 compute_accuracy.py datasets/vision.test predictions/vision.test.predictions