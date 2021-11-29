'''
Machine Learning Project: Processing Features

Run python3 process_features.py to watch the magic happen!
'''

import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# MODIFY THESE VALUES FOR YOUR SYSTEM
DATASET_FOLDER = './mnist/'
OUTPUT_FOLDER = './output/'

# TODO(): Download MNIST
train_file_path = DATASET_FOLDER + "train.csv"
test_file_path = DATASET_FOLDER + "test.csv"
train = pd.read_csv(train_file_path, engine = 'python',encoding='UTF-8')
test = pd.read_csv(test_file_path, engine = 'python',encoding='UTF-8')

# TODO(): Process MNIST
def processData():
    # train_np = train.values
    # Y_train = train_np[:,0]
    # X_train = train_np[:,1:]
    Y_train = train["label"]
    print(Y_train.shape)
    X_train = train.drop(labels = ["label"],axis = 1)

    # convert value that is > 0 (positive) to 1
    # X_train=np.where(X_train > 0, 1, 0) 
    print(X_train.shape)

    # test_np = test.values

    # test=np.where(test_data > 0, 1, 0)
    print(test.shape)
    return X_train, Y_train, test

# # TODO(): Scrub Features
def scrubData(train, test, whiten, n_components):
    pca = PCA(whiten=whiten, n_components=n_components)

    pca.fit(train)
    train_data = pca.transform(train)

    test_data = pca.transform(test)
    return train_data, test_data



# TODO(): Output Results
def writeData(data, filename):
    csv_file = OUTPUT_FOLDER + filename
    with open(csv_file,'w',newline='') as myFile:
        myWriter=csv.writer(myFile)
        myWriter.writerow(['ImageId','Label'])
        for i in range(len(data)):
            myWriter.writerow([i+1,str(data[i])])
    print("Finish writeData")
    # dataframe = pd.DataFrame(data)
    # dataframe.to_csv(csv_file)
    # pass

# if __name__ == '__main__':
#     processData()