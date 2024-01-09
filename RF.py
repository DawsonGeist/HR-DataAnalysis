import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def rf(x_train, x_test, y_train, y_test):
    rforest = RandomForestClassifier(max_depth=19, random_state=42)
    rforest.fit(x_train, y_train)
    y_predictions = rforest.predict(x_test)
    return y_predictions