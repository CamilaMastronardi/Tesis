# Librerias basicas para manejo de datos
import numpy as np
import pandas as pd
import os
import sys
import time

#Traigo KNN y DTW
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from KNN import KNN_timeSeries, DTW, completeData, trainingData

#Performance
def modelKNN(year: str, mww: int = 1000, K: int = 1, weights: bool = False):
    #Cargar training data:

    print('Cargando datos de entrenamiento')
    start_time = time.time()

    train_data = trainingData(['Group1','Group2','Group3','Group4'])

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Datos de entrenamiento cargados en {execution_time}')

    #Descarga y preparación de datos:

    print('Descargando y preparando datos de campo magnetico a clasificar')
    start_time = time.time()

    str_name = f'fechas_{year}'
    data_to_be_classified = completeData(str_name)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Datos a ser clasificada descargada y preparada en {execution_time}')

    X = data_to_be_classified['B']
    dates = data_to_be_classified['Fecha']
    times = data_to_be_classified['time']

    X_train = train_data['B']
    y_train = train_data['tipo'].to_numpy().astype(int)

    dtw_calculator = DTW(max_warping_window = mww)
    KNN = KNN_timeSeries(metric_calculator = dtw_calculator, n_neighbors = K, use_weights = weights)

    KNN.fit(X_train, y_train)
    
    y_pred, y_prob = KNN.predict(X)   
    result = pd.DataFrame({'Fecha': np.array(dates) ,'X': list(X), 'time': list(times),
        'y_pred': y_pred, 'y_prob': y_prob})
    result['X'] = result['X'].apply(lambda x: ', '.join(map(str, x)))
    result['time'] = result['time'].apply(lambda x: ', '.join(map(str, x)))
    
    path_file = f'/app/KNN/Clasificador/CampoMagnetico_{year}_{K}vecinos_{mww}DTW.csv'
    result.to_csv(path_file)

start_time = time.time()
modelKNN('2015')
end_time = time.time()
print(f'Terminado. Tiempo de ejecución: {end_time-start_time} segundos')