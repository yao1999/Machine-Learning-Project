from sklearn.neighbors import KNeighborsClassifier
from process_features import processData, writeData
# from grader import get_score
def knn_classifier(X_train, Y_train):
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', weights='distance', p=3)
    knn_model = knn.fit(X_train,Y_train)
    return knn_model

def knn_model(X_train, Y_train, test):
    print("Start KNN model")
    knn_model = knn_classifier(X_train, Y_train)
    predict = knn_model.predict(test)
    writeData(predict, "knn.csv")
    print("Finish")
    
if __name__ == '__main__':
   X_train, Y_train, test = processData()
   knn_model(X_train, Y_train, test)