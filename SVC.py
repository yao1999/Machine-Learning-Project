from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
from process_features import writeData
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def scrubData(X_train, test):
    pca = PCA(whiten=True, n_components=0.85)

    pca.fit(X_train)
    train_data = pca.transform(X_train)

    test_data = pca.transform(test)
    return train_data, test_data


def svc_model(X_train, y_train, test):
    # clf = make_pipeline(StandardScalar(), SVC(gamma='auto'))
    # clf.fit(X_train,y_train)
    # Pipeline(steps=[('standardscaler', StandardScalar()),('svc', SVC(gamma='auto'))])
    # clf_svm = SVC(kernel='rbf', gamma=5, C=0.001)

    print("Start PCA")
    X_train, test = scrubData(X_train, test)
    print("Finish PCA")

    print("Start SVR model")
    svc_model = SVC()
    svc_model.fit(X_train, y_train)
    predict = svc_model.predict(test)

    # get_score(predict, test, "Random forest")
    writeData(predict, "svc.csv")
    print("Finish")