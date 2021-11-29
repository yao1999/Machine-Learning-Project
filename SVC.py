from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
from process_features import writeData, scrubData
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


    print("Start PCA")
    X_train, test = scrubData(X_train, test, True, 0.85)
    print("Finish PCA")

    print("Start SVC model")
    svc_model = SVC(C=2.065921, gamma=0.02)
    svc_model.fit(X_train, y_train)
    predict = svc_model.predict(test)

    writeData(predict, "svc.csv")
    print("Finish")