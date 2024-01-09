import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

def ab(x_train, x_test, y_train, y_test):
    adaBoost = AdaBoostClassifier(n_estimators=125, random_state=42)
    adaBoost.fit(x_train, y_train)
    y_predictions = adaBoost.predict(x_test)
    return y_predictions