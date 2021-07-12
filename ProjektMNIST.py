import gzip
import os
import numpy as np

train_data = os.path.join("..", "data", "mnist", "train-images-idx3-ubyte.gz")
train_labels = os.path.join("..", "data", "mnist", "train-labels-idx3-ubyte.gz")

test_data = os.path.join("..", "data", "mnist", "test-images-idx3-ubyte.gz")
test_labels = os.path.join("..", "data", "mnist", "test-labels-idx3-ubyte.gz")