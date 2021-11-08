'''
Machine Learning Project: Processing Features

Run python3 process_features.py to watch the magic happen!
'''

import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import PCA

# MODIFY THESE VALUES FOR YOUR SYSTEM
DATASET_FOLDER = './mnist/'
OUTPUT_FOLDER = './output/'

# TODO(): Download MNIST
train_file_path = DATASET_FOLDER + "train.csv"
test_file_path = DATASET_FOLDER + "test.csv"
train = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# TODO(): Process MNIST
def processData():
    Y_train = train["label"]
    print(Y_train.shape)
    X_train = train.drop(labels = ["label"],axis = 1) 

    # convert value that is > 0 (positive) to 1
    X_train=np.where(X_train > 0, 1, 0) 
    print(X_train.shape)

    test=np.where(test_data > 0, 1, 0)
    print(test.shape)
    return X_train, Y_train, test

# TODO(): Scrub Features
def scrubData(data):
    XP_train, YP_train, P_test = processData()

    pca = PCA(n_components=5)
    pca.fit(XP_train)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

# TODO(): Output Results
def writeData(data, filename):
    csv_file = OUTPUT_FOLDER + filename
    with open(csv_file,'w',newline='') as myFile:
        myWriter=csv.writer(myFile)
        myWriter.writerow(['ImageId','Label'])
        for i in range(len(data)):
            myWriter.writerow([i+1,str(data[i])])
    # dataframe = pd.DataFrame(data)
    # dataframe.to_csv(csv_file)
    # pass

# if __name__ == '__main__':
#     processData()