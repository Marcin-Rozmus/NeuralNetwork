# Copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Source: https://cs231n.github.io/neural-networks-case-study/

import numpy as np


def generate_data(samples: int, classes: int):
    X = np.zeros((samples*classes, 2))  # data matrix (each row = single example)
    y = np.zeros(samples*classes, dtype='uint8')  # class labels
    for class_no in range(classes):
        ix = range(samples*class_no, samples*(class_no+1))
        r = np.linspace(0.0, 1, samples)  # radius
        t = np.linspace(class_no*4, (class_no+1)*4, samples) + np.random.randn(samples)*0.2  # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_no

    return X, y
