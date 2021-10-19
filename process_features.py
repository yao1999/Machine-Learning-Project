'''
Machine Learning Project: Processing Features

Run python3 process_features.py to watch the magic happen!
'''

import pandas as pd
import numpy as np


# MODIFY THESE VALUES FOR YOUR SYSTEM
DATASET_FOLDER = 'mnist/'
# OUTPUT_FOLDER = '.'
OUTPUT_FOLDER = '.output/'

# TODO(): Download MNIST
train_file_path = "./" + DATASET_FOLDER + "train.csv"
test_file_path = "./" + DATASET_FOLDER + "test.csv"
train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

# TODO(): Process MNIST
def processData():
    Y_train = train["label"]
    print(Y_train.shape)
    X_train = train.drop(labels = ["label"],axis = 1) 
    print(X_train.shape)
    pass

# TODO(): Scrub Features
def scrubData(data):
    pass

# TODO(): Output Results
def writeData(data):
    pass

if __name__ == '__main__':
    processData()