from sklearn.ensemble import RandomForestClassifier
from process_features import processData, writeData
from grader import get_score

def random_forest_classifier(X_train, Y_train):
    rf = RandomForestClassifier(n_estimators=400, n_jobs=4, verbose=1, oob_score=True, random_state=10)
    rf_model = rf.fit(X_train,Y_train)
    # get_score(rf, X_train, Y_train, "Randome forest")
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
    