from RandomForestModel import randome_forest_model
from KNN import knn_model
from process_features import processData

if __name__ == '__main__':
    X_train, Y_train, test = processData()
    randome_forest_model(X_train, Y_train, test)
    knn_model(X_train, Y_train, test)
