import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def qda(x_train, x_test, y_train, y_test):
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred