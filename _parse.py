import csv
from collections import OrderedDict
import numpy as np


def load_wine_csv(fileName, classify=False):
    # open a csv
    fp = open(fileName)
    csv_fp = csv.reader(fp)

    # declare lists
    # is the y
    quality = []
    # is the x
    features = []

    # iterate through the csv file and append it to the list
    for row in csv_fp:
        quality.append(row[2])
        # somehow the last num is non inclusive
        features.append(row[0:2])

    # close fp
    fp.close()

    # remove the headers
    # quality.pop(0)
    # features.pop(0)

    # declare x as an np array
    x = np.zeros((len(features), len(features[0])))

    # convert to int the features
    for listIdx in range(len(features)):
        for featIdx in range(len(features[0])):
            x[listIdx, featIdx] = float(features[listIdx][featIdx])

    # default = regression
    y = np.zeros((len(quality), 1), dtype=np.int)
    # convert into an int
    idx = 0
    for row in quality:
        y[idx, 0] = int(row)
        idx += 1


    return x, y
