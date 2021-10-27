from sklearn.ensemble import RandomForestClassifier
from process_features import processData, writeData


def random_forest_classifier(X_train, Y_train):
    rf = RandomForestClassifier()
    rf_model = rf.fit(X_train,Y_train)
    return rf_model

def randome_forest_model(X_train, Y_train, test):
    print("Start Ransom forest model")
    rf_model = random_forest_classifier(X_train, Y_train)
    predict = rf_model.predict(test)
    writeData(predict, "random_forest.csv")
    print("Finish")

if __name__ == '__main__':
   X_train, Y_train, test = processData()
   randome_forest_model(X_train, Y_train, test)
    