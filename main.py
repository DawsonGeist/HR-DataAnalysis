import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA

from LoadDataIntoPandas import ImportAndCleanData
from RF import rf
from AdaBoost import ab
from QuadraticDiscriminantAnalysis import qda

# Main script
if __name__ == "__main__":
    # Gather and Prepare our data
    df, dataMapping = ImportAndCleanData()
    # Convert to Column vectors
    data = df.to_numpy()
    print(data)
    
    # To Predict Attrition we have to split out the Attrition Column from the rest of the data
    y = np.array(data.T[0])
    X = data.T[1:].T
    print("y: ")
    print(y)
    print("X: ")
    print(X)

    print(X.shape)
    print(y.shape)

    # Split our data into train-test with a 70:30 ratio
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Random Forest
    rf_y_predictions = rf(x_train, x_test, y_train, y_test)
    # Check Scores
    print('RF accuracy: {0:0.4f}'.format(accuracy_score(y_test, rf_y_predictions)))

    # AdaBoost
    ab_y_predictions = ab(x_train, x_test, y_train, y_test)
    # Check Scores
    print('AB accuracy: {0:0.4f}'.format(accuracy_score(y_test, ab_y_predictions)))
    
    # AdaBoost
    qda_y_predictions = qda(x_train, x_test, y_train, y_test)
    # Check Scores
    print('QDA accuracy: {0:0.4f}'.format(accuracy_score(y_test, qda_y_predictions)))

    # Colinear Variables are found in QDA

    # Do PCA and repeat
    transformer = KernelPCA(n_components=3, kernel='linear')
    X_transformed = transformer.fit_transform(X)
    

    # Split our data into train-test with a 70:30 ratio
    x_train, x_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.33, random_state=42)

    # Random Forest
    rf_y_predictions = rf(x_train, x_test, y_train, y_test)
    # Check Scores
    print('RF accuracy: {0:0.4f}'.format(accuracy_score(y_test, rf_y_predictions)))

    # AdaBoost
    ab_y_predictions = ab(x_train, x_test, y_train, y_test)
    # Check Scores
    print('AB accuracy: {0:0.4f}'.format(accuracy_score(y_test, ab_y_predictions)))
    
    # AdaBoost
    qda_y_predictions = qda(x_train, x_test, y_train, y_test)
    # Check Scores
    print('QDA accuracy: {0:0.4f}'.format(accuracy_score(y_test, qda_y_predictions)))

    # Plot actual values to visualize data
    x_red = []
    x_blue = []
    for index, point in enumerate(X_transformed):
        if y[index] == 0:
            x_blue.append(point)
        else:
            x_red.append(point)
    x_red = np.array(x_red)
    x_blue = np.array(x_blue)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    #ax.scatter(x_blue.T[0], x_blue.T[1], x_blue.T[2], marker='o')
    ax.scatter(x_red.T[0], x_red.T[1], x_red.T[2], marker='x')

    plt.show()

    # Now 2D
    
    # Do PCA and repeat
    transformer = KernelPCA(n_components=2, kernel='linear')
    X_transformed = transformer.fit_transform(X)
    
    # Plot actual values to visualize data
    x_red = []
    x_blue = []
    for index, point in enumerate(X_transformed):
        if y[index] == 0:
            x_blue.append(point)
        else:
            x_red.append(point)
    x_red = np.array(x_red)
    x_blue = np.array(x_blue)

    fig, axs = plt.subplots(2)

    axs[0].plot(x_blue.T[:1], x_blue.T[1:], 'b+') 
    axs[0].plot(x_red.T[:1], x_red.T[1:], 'r+') 
    #Y = -2790, Y = 1810, y = 5500, y = 8900
    axs[0].axhline(y = -2790, color = 'g', linestyle = '-') 
    axs[0].axhline(y = 1810, color = 'g', linestyle = '-') 
    axs[0].axhline(y = 5500, color = 'g', linestyle = '-') 
    axs[0].axhline(y = 8900, color = 'g', linestyle = '-')

    #Test

    # Plot one person
    employee = 1
    x_demo = transformer.transform(np.array([X[employee]]))
    print(x_demo)
    axs[1].plot(x_demo.T[0][0], x_demo.T[1][0], ('b+' if y[employee] == 0 else 'r+'))
    axs[1].axhline(y = -2790, color = 'g', linestyle = '-') 
    axs[1].axhline(y = 1810, color = 'g', linestyle = '-') 
    axs[1].axhline(y = 5500, color = 'g', linestyle = '-') 
    axs[1].axhline(y = 8900, color = 'g', linestyle = '-')
    plt.show()





    