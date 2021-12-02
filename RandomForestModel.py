from sklearn.ensemble import RandomForestClassifier
from process_features import processData, writeData

def random_forest_classifier(X_train, y_train):
    rf = RandomForestClassifier(criterion = "entropy", max_depth=12, min_samples_leaf=1, min_samples_split=6, n_estimators=50)
    rf_model = rf.fit(X_train, y_train)
    return rf_model

def randome_forest_model(X_train, Y_train, test):
    print("Start Random forest model")
    rf_model = random_forest_classifier(X_train, Y_train)
    predict = rf_model.predict(test)
    writeData(predict, "random_forest.csv")
    print("Finish")

# if __name__ == '__main__':
#    X_train, Y_train, test = processData()
#    randome_forest_model(X_train, Y_train, test)
    