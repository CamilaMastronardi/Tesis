# Librerias basicas para manejo de datos
import numpy as np
import pandas as pd
import os
import sys

# Librerias para graficar
import matplotlib.pyplot as plt
import seaborn as sn

#Librerias para medición de tiempos
from tqdm import tqdm
import time

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

# Funciones de otros archivos

from KNN import DTW, KNN_timeSeries, completeData
from Algoritmo3clusters import trainingData3Clusters
from Algoritmo3clusters import cut_orbits_in_n_min


def _KNN_3clusters(data_to_classify, training_data, folder: str, filename: str, mww: int = 1000, K: int = 25, use_weights: bool = True) -> None:   
    """
    Performs KNN classification using DTW distance metric
    Arguments
    ---------
    data_to_classify : pd.DataFrame
        DataFrame containing data to classify with columns 'B', 'Fecha', 'orbita', 'time', 'Bx', 'By', 'Bz', 'posX', 'posY', 'posZ'
    training_data : pd.DataFrame
        DataFrame containing training data with columns 'B' and 'tipo'
    mww : int   
        Maximum warping window for DTW
    K : int
        Number of neighbors for KNN
    use_weights : bool  
        If True, use weights for KNN
    folder : str    
        Folder name for saving results
    """
    # Crear carpeta si no existe
    if not os.path.exists(f'/app/KNN/Clasificador/{folder}'):
        os.makedirs(f'/app/KNN/Clasificador/{folder}')

    X_train = training_data['B']
    y_train = training_data['tipo'].to_numpy().astype(int)
    X = data_to_classify['B']

    dtw_calculator = DTW(max_warping_window = mww)
    KNN = KNN_timeSeries(metric_calculator = dtw_calculator, n_neighbors = K, use_weights=use_weights)
    
    s = time.time()

    KNN.fit(X_train, y_train)
    y_pred, y_prob = KNN.predict(X)
    
    Fecha = data_to_classify['Fecha']
    orbita = data_to_classify['orbita']
    times = data_to_classify['time']
    posX = data_to_classify['posX']
    posY = data_to_classify['posY']
    posZ = data_to_classify['posZ']
    B = np.array(X)

    dict = {"Fecha": Fecha, "orbita": orbita, 'pred': y_pred,
            "B": B, "time": np.array(times)
            , "posX": np.array(posX), "posY": np.array(posY), "posZ": np.array(posZ)}

    result = pd.DataFrame(dict)
    
    path_file = f'/app/KNN/Clasificador/{folder}/{filename}.csv'
    result.to_csv(path_file)

def KNN_3clusters(file_dates: str, save_file:str):
    
    start_time = time.time()
    print('Descargando y ordenando data de entrenamiento')
    training_data = trainingData3Clusters(['Group1','Group2','Group3','Group4'], use_cache = True, mins = 2)
    print('Preparando data a clasificar')
    df = completeData(file_dates, use_cache=True)
    rows = []
    for i in range(df.shape[0]):

        list_B = cut_orbits_in_n_min(df.iloc[i]['B'], 2)
        list_time = cut_orbits_in_n_min(df.iloc[i]['time'], 2)
        list_posX = cut_orbits_in_n_min(df.iloc[i]['posX'], 2)
        list_posY = cut_orbits_in_n_min(df.iloc[i]['posY'], 2)
        list_posZ = cut_orbits_in_n_min(df.iloc[i]['posZ'], 2)
       
        for j in range(len(list_B)): 

            rows.append({
                "Fecha": df.iloc[i]['Fecha'],
                "orbita": df.iloc[i]['orbita'],
                "B": list_B[j],
                "time": list_time[j],
                "posX": list_posX[j],
                "posY": list_posY[j],
                "posZ": list_posZ[j]
            })

    data_to_classify = pd.DataFrame(rows) 
    print('Clasificando con KNN')  
    _KNN_3clusters(data_to_classify, training_data, folder = 'Prueba', filename = save_file, mww = 1000, K = 25, use_weights = True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo total de ejecución: {execution_time:.2f} segundos")

if __name__ == "__main__":
    file_dates = 'fechas_prueba'
    filename = 'prueba'
    KNN_3clusters(file_dates, filename)