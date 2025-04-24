from KNN import KNN_timeSeries, DTW, outbound_to_inbound
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
digits = load_digits()

import matplotlib.pyplot as plt
import seaborn as sn


def test_clase_KNN_usa_los_vecinos_mas_cercanos():
    X_toy = np.random.random((100,200)) #100 series de 200 datos 
    y_toy = np.random.randint(0,1, (100)) #el tipo de las 100 series
    X_toy_train, X_toy_test, y_toy_train, y_toy_test = train_test_split(X_toy, y_toy, test_size=0.33, random_state=42)

    expected = []
    for i in range(len(y_toy_test)):
        val1 = y_toy_train[i+1]
        val2 = y_toy_train[i+2]
        val3 = y_toy_train[i+3]
        if val1 == val2 or val1==val3:
            expected.append(val1)
        else: 
            expected.append(val2)

    class distance_for_KNN_test(object):      
        def dist_matrix(X_test, X_train): 
            M = np.ones([len(X_test), len(X_train)])*np.inf
            for j in range(len(X_test)):
                M[j,j+1] = 1
                M[j,j+2] = 2
                M[j,j+3] = 3
            return M
            
    KNN = KNN_timeSeries(distance_for_KNN_test, n_neighbors= 3, use_weights=False)
    KNN.fit(X_toy_train, y_toy_train)
    y_pred_labels, y_pred_probas = KNN.predict(X_toy_test)
    assert (y_pred_labels == np.array(expected)).all()
    print('Test KNN usa los K vecinos más cercanos: pasado')

def test_clase_KNN_usa_bien_el_peso():
    X_toy = np.random.random((100,10))
    y_toy = np.random.randint(0,2, (100))
    X_toy_train, X_toy_test, y_toy_train, y_toy_test = train_test_split(X_toy, y_toy, test_size=0.33, random_state=42)

    expected = []
    for i in range(len(y_toy_test)):
        val1 = y_toy_train[i+1]
        expected.append(val1)

    class distance_for_KNN_test(object):      
        def dist_matrix(X_test, X_train): 
            M = np.ones([len(X_test), len(X_train)])*np.inf
            for j in range(len(X_test)):
                M[j,j+1] = 1
                M[j,j+2] = 100
                M[j,j+3] = 100
            return M

    KNN = KNN_timeSeries(distance_for_KNN_test, n_neighbors= 3, use_weights=True)
    KNN.fit(X_toy_train, y_toy_train)
    y_pred_labels, y_pred_probas = KNN.predict(X_toy_test)
    assert (y_pred_labels == expected * np.ones(len(y_pred_labels))).all()
    print('Test de KNN usa bien el peso: pasado')


def test_clase_DTW():

    x = np.random.randint(0,3*np.pi, (100))
    x2 = np.random.randint(np.pi, 5*np.pi, (200))
    cos_1 = 10*np.cos(x)
    cos_2 = x2**2 + np.random.normal(0, 1, 200)
    cos_3 = 10*np.cos(x) +  np.random.normal(0, 0.1, 100)
    sin_1 = 10*np.sin(x2)

    X = [cos_1, cos_2, cos_3, sin_1]
    
    dtw_calculator = DTW()
    M = dtw_calculator.dist_matrix(X,X)
    assert((M[i,j]==0 for (i,j) in range(len(M))) and (M[i,j]==M[j,i] for (i,j) in zip(range(len(M)), range(len(M)))) and (M[0,1] > M[0,3]))
    print('Test DTW calcula correctamente similaridad: pasado')


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

#test_KNN_with_DTW_works_with_pandas_series()

def test_outbound_to_inbound():
    # Caso 1: posX creciente → no debería invertir
    df1 = pd.DataFrame({'posX': np.linspace(0, 10, 11)})
    result1 = outbound_to_inbound(df1)
    assert result1.equals(df1), "Error: posX creciente fue invertido incorrectamente."

    # Caso 2: posX decreciente → debería invertir
    df2 = pd.DataFrame({'posX': np.linspace(10, 0, 11)})
    expected2 = df2.iloc[::-1].reset_index(drop=True)
    result2 = outbound_to_inbound(df2).reset_index(drop=True)
    assert result2.equals(expected2), "Error: posX decreciente no fue invertido correctamente."

    # Caso 3: posX constante → debería dejarlo igual
    df3 = pd.DataFrame({'posX': np.ones(10)})
    result3 = outbound_to_inbound(df3)
    assert result3.equals(df3), "Error: posX constante fue modificado incorrectamente."

    print("Test funcion outbound to inbound pasado")


i = 0
while i<10: 
    i=i+1
    print(f'iteracion {i}')
    test_clase_KNN_usa_los_vecinos_mas_cercanos()
    test_clase_DTW()
    test_clase_KNN_usa_bien_el_peso()
    test_outbound_to_inbound()
