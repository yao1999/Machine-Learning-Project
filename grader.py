from sklearn.model_selection import cross_val_score
import numpy as np


def get_score(model, X_train, Y_train, model_name):
    scores = cross_val_score(model, X_train, Y_train, cv=10, scoring='accuracy')
    print(f'{model_name} Score: {round(np.mean(scores), 5)} ~ {round(np.amax(scores), 5)}')