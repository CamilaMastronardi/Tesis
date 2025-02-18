import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.pyplot as plt 
from tqdm import tqdm
from typing import Callable, Any, Iterable

import numba
from numba import njit, prange

from scipy.spatial.distance import squareform
import collections
import itertools
from scipy.stats import mode
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split

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
        is_MPB_orbit,
        df["mod_B"].to_numpy(), 
    ]
    return data

start_time = time.time()

MPB_crosses_df = cargarTrainingData(group='Group1')
data_to_complete = pd.DataFrame(columns=["Fecha", "orbita", "B", "tipo"])

for i, (YYYY, MM, DD) in tqdm(enumerate(zip(MPB_crosses_df.YYYY, MPB_crosses_df.MM, MPB_crosses_df.DD), start=1), total=len(MPB_crosses_df), desc="Procesando fechas"):
    orbitas = n_orbita(YYYY, MM, DD)
    for n in range(1, orbitas): 
        df_not_marked = filtradoVignes(YYYY, MM, DD, n)
        time_MPB = MPB_crosses_df.loc[(MPB_crosses_df['YYYY'] == YYYY) & (MPB_crosses_df['MM'] == MM) & (MPB_crosses_df['DD'] == DD)].MPB_time
        is_MPB_orbit = is_mpb_orbit(df_not_marked, time_MPB, 3)
        data_for_KNN(len(data_to_complete), YYYY, MM, DD, n, df_not_marked, data_to_complete, is_MPB_orbit)
        data_KNN_completed = data_to_complete
# Calcular tiempo total de ejecución
end_time = time.time()
execution_time = end_time - start_time

print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
 
X = data_KNN_completed['B'].tolist()
y = data_KNN_completed['tipo']

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
        return dm


class KNN_timeSeries(object):
    """K-nearest neighbor classifier using an indicated metric for series
    
    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN
        
    metric_calculator: str, optional (default= 'dtw') 
            Metric for measure distances between series
    """
    
    def __init__(self, metric_calculator, n_neighbors: int =5):
        self.n_neighbors = n_neighbors
        self.metric_calculator = metric_calculator
    
    def fit(self, X_train: Iterable[Iterable[float]], y_train: Iterable[bool]):
        """Fit the model using X as training data and y as class labels
        
        Arguments
        ---------
        X_train : Iterable of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer
            
        y_train : Iterable of shape [n_samples]
            Training labels for input into KNN classifier
        """
        
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X: Iterable[Iterable[float]]) -> list[np.array, np.array]:
        """Predict the class labels or probability estimates for 
        the provided data

        Arguments
        ---------
          X : Iterable of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified
          
        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels 
              (2) the knn label count probability
        """
        
        dm = self.metric_calculator.dist_matrix(X, self.X_train)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.y_train[knn_idx]
        
        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()

dtw_calculator = DTW()
KNN = KNN_timeSeries(metric_calculator = dtw_calculator)

X_toy = np.random.random((100,10))
y_toy = np.random.randint(0,2, (100))
X_toy_train, X_toy_test, y_toy_train, y_toy_test = train_test_split(X_toy, y_toy, test_size=0.33, random_state=42)

KNN.fit(X_toy_train, y_toy_train)
s = time.time()