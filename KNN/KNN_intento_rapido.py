#Librerias para manejo de datos
import numpy as np
import pandas as pd
import os
import sys

#Librerias para etiquetado de datos
from typing import Callable, Any, Iterable
import collections

#Librerias para graficar
import matplotlib.pyplot as plt
import seaborn as sn

#Librerias para medicion de tiempos
import time
from tqdm import tqdm

#Librerias para acelerar la performance
import numba
from numba import njit, prange

# Librerias para KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Funciones de otros codigos que tengo que traer
from DescargaTrainingData import cargarTrainingData

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from PreprocesamientoDatos.CorteVignes import filtradoVignes 
from PreprocesamientoDatos.PromedioPorVentana import n_orbita
    
def is_mpb_orbit(orbit_df: pd.DataFrame, mpb_times: pd.DataFrame, delta_sec: int) -> bool:
    has_mpb = False
    for time in mpb_times:
        has_mpb = has_mpb or (abs(orbit_df["time"] - time) < (delta_sec / 3600)).any()
    return has_mpb

def data_for_KNN(j: int, YYYY: str, MM: str, DD: str, orbita: int, df: pd.DataFrame, data: pd.DataFrame, is_MPB_orbit: bool) -> pd.DataFrame:
    data.loc[j, ["Fecha", "orbita", "tipo", "B"]] = [
        f"{YYYY}-{MM}-{DD}",
        orbita,
        int(is_MPB_orbit),
        df["mod_B"].to_numpy(), 
    ]
    return data

start_time = time.time()

MPB_crosses_df = cargarTrainingData(groups=['Group1', 'Group2', 'Group3', 'Group4'])
data_to_complete = pd.DataFrame(columns=["Fecha", "orbita", "B", "tipo"])

for i, (YYYY, MM, DD) in tqdm(enumerate(zip(MPB_crosses_df.YYYY, MPB_crosses_df.MM, MPB_crosses_df.DD), start=1), total=len(MPB_crosses_df), desc="Procesando fechas"):
    orbitas = n_orbita(YYYY, MM, DD)
    for n in range(1, orbitas): 
        df_not_marked = filtradoVignes(YYYY, MM, DD, n)
        if len(df_not_marked)!=0:
            time_MPB = MPB_crosses_df.loc[(MPB_crosses_df['YYYY'] == YYYY) & (MPB_crosses_df['MM'] == MM) & (MPB_crosses_df['DD'] == DD)].MPB_time
            is_MPB_orbit = is_mpb_orbit(df_not_marked, time_MPB, 3)
            data_for_KNN(len(data_to_complete), YYYY, MM, DD, n, df_not_marked, data_to_complete, is_MPB_orbit)
            data_KNN_completed = data_to_complete

# Calcular tiempo total de ejecución
end_time = time.time()
execution_time = end_time - start_time

print(f"Tiempo de ejecución: {execution_time:.2f} segundos")


@njit
def euclidean(x, y):
    return abs(x - y)

@njit
def squared_metric(x, y):
    return (x - y) ** 2

metrics = {
    "euclidean": euclidean,
    "squared": squared_metric
}

@njit
def similarity(ts_a: Iterable[float], ts_b: Iterable[float], max_warping_window: int, metric: callable) -> float:
    """Returns the DTW similarity distance between two 2-D
    timeseries numpy arrays.

    Arguments
    ---------
    time_series_a, time_series_b : Iterable of shape n_timepoints
        Two arrays containing n_samples of timeseries data
    
    d : DistanceMetric object (default = euclidean)
    
    Returns
    -------
    DTW distance between A and B
    """
    M, N = len(ts_a), len(ts_b)
    cost = np.full((M, N), np.inf)
    cost[0, 0] = metric(ts_a[0], ts_b[0])
    
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + metric(ts_a[i], ts_b[0])
    
    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + metric(ts_a[0], ts_b[j])
    
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window), min(N, i + max_warping_window)):
            cost[i, j] = min(cost[i-1, j-1], cost[i, j-1], cost[i-1, j]) + metric(ts_a[i], ts_b[j])
    
    return cost[-1, -1]

class DTW(object):
    """ Calculates the matrix of Dynamic Time Warping between two Iterables
    
    Arguments
    ---------
    max_warping_window :  int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function          
    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    metric: metric to calculate distance matrix 
    """

    def __init__(self, max_warping_window: int = 10000, subsample_step: int = 1, metric: str = "euclidean"):
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
        self.metric = metrics[metric]

    def dtw_distance(self, time_series_a: Iterable[float], time_series_b: Iterable[float]) -> float:
        ts_a = np.array(time_series_a)[::self.subsample_step]
        ts_b = np.array(time_series_b)[::self.subsample_step]
        return similarity(ts_a, ts_b, self.max_warping_window, self.metric)
    
    def dist_matrix(self, X_test: Iterable[Iterable[float]], X_train: Iterable[Iterable[float]]) -> np.array:
        """Computes the M x N distance matrix between the training
        dataset and testing dataset using the DTW distance measure
        
        Arguments
        ---------
        X_test : Iterable of testing_n_samples arrays with equal or different shapes
        
        X_train : Iterable of training_n_samples arrays with equal or different shapes
        
        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """
        X_test, X_train = list(X_test), list(X_train)
        x_s, y_s = len(X_test), len(X_train) 
        dm = np.zeros((x_s, y_s))
        for i in prange(x_s):
            for j in prange(y_s):
                dm[i, j] = self.dtw_distance(X_test[i], X_train[j])
                print(f'Parametro {y_s*i+j} de {y_s*x_s} DTW')
        return dm

if __name__== '__main__' :
    mww = 500
    K = 2
    dtw_calculator = DTW(max_warping_window = mww).dtw_distance
    KNN = KNeighborsClassifier(metric = dtw_calculator, n_neighbors = K)

    X = data_KNN_completed['B']
    y = data_KNN_completed['tipo'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    y_test = y_test.astype('int')
    y_train = y_train.astype('int')

    KNN.fit(X_train, y_train)
    s = time.time()
    y_pred, y_prob = KNN.predict(X_test)

    path_file = f'/app/KNN/Resultados/{K}vecinos_{mww}mww_DTW_without_tree.txt'

    with open(path_file, 'w') as file: 
        file.write(f'Tiempo de ejecución de KNN: {time.time()-s} \n')
        file.write(f'y_pred, y_prob \n')
        for val1, val2 in zip(y_pred,y_prob):
            file.write(val1, val2 + f'\n')

    cm = confusion_matrix(y_test, y_pred)
        
    plt.figure(figsize=(7,5))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig(f'/app/KNN/Resultados/{K}vecinos_{mww}mww_DTW_without_tree.png')
