from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import StandardScalar
from process_features import processData, writeData

def svc_classifier(X_train, y_train):
    print("Start SVR model")
    clf = make_pipeline(StandardScalar(), SVC(gamma='auto'))
    clf.fit(X_train,y_train)
    Pipeline(steps=[('standardscaler', StandardScalar()),('svc', SVC(gamma='auto'))])
    print("Finish")
    return clf