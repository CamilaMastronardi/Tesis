from KNN import KNN_timeSeries
from KNN import DTW
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
digits = load_digits()

import matplotlib.pyplot as plt
import seaborn as sn


def test_clase_KNN_usa_los_vecinos_mas_cercanos():
    X_toy = np.random.random((100,10))
    y_toy = np.random.randint(0,2, (100))
    X_toy_train, X_toy_test, y_toy_train, y_toy_test = train_test_split(X_toy, y_toy, test_size=0.33, random_state=42)

    val_1, val_2, val_3 = y_toy_train[1], y_toy_train[2], y_toy_train[3]
    if val_1 == val_2 or val_1 == val_3:
        expected = val_1
    else:
        expected = val_2

    class distance_for_KNN_test(object):
        
        def dist_matrix(X_test, X_train): 
            M = np.ones([len(X_test), len(X_train)])*np.inf
            M[:,3] = 0
            M[:,2] = 1
            M[:,1] = 1
            return M


    KNN = KNN_timeSeries(distance_for_KNN_test, n_neighbors= 2)
    KNN.fit(X_toy_train, y_toy_train)
    y_pred_labels, y_pred_probas = KNN.predict(X_toy_test)
    assert (y_pred_labels == expected * np.ones(len(y_pred_labels))).all()

#test_clase_KNN_usa_los_vecinos_mas_cercanos()

def test_clase_DTW():

    x = np.linspace(0,100, 1000)
    x2 = np.linspace(0,101, 1010)
    cos_1 = 10*np.cos(x)
    cos_2 = 10*np.cos(x2) + np.random.normal(0, 1, 1010)
    cos_3 = 10*np.cos(x) +  np.random.normal(0, 0.1, 1000)
    sin_1 = 10*np.sin(x2)

    X = [cos_1, cos_2, cos_3, sin_1]
    
    dtw_calculator = DTW()
    M = dtw_calculator.dist_matrix(X,X)

    return M

#M = test_clase_DTW()
#print(M)

def test_KNN_with_DTW(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=4)
    KNN = KNN_timeSeries(metric_calculator = DTW(), n_neighbors=2)
    KNN.fit(X_train, y_train)

    y_pred, y_prob = KNN.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(7,5))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig('temp.png')

    return y_pred, y_test

def test_KNN_with_DTW_works_with_pandas_series():
    x = np.linspace(0,100, 1000)
    x2 = np.linspace(0,101, 1010)
    f_1 = 10*np.cos(x)
    f_2 = 10*np.cos(x2) + np.random.normal(0, 1, 1010)
    f_3 = 10*np.cos(x) +  np.random.normal(0, 0.1, 1000)
    f_4 = 10*np.sin(x)
    f_5 = 10*np.cos(x)
    f_6 = 10*np.cos(x2) + np.random.normal(0, 1, 1010)
    f_7 = 9*np.cos(x) +  np.random.normal(0, 0.1, 1000)
    f_8 = 10*np.sin(x)+  np.random.normal(0, 0.1, 1000)
    f_9 = 2 + np.sin(x)
    f_10 = 2 + np.cos(x)

    X = pd.Series([f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10])
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])

    test_KNN_with_DTW(X, y)

test_KNN_with_DTW_works_with_pandas_series()