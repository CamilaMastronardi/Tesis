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

# Libreria para etiquetar inputs y outputs de funciones
from typing import Callable, Any, Iterable

# Libreria para acelerar loops
from numba import njit, prange

# Librerias para DTW
from scipy.spatial.distance import squareform
import collections
import itertools

# Funciones para estadistica
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from scipy.stats import mode

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

# Funciones de otros archivos
from DescargaTrainingData import cargarTrainingData, cargarData
from PreprocesamientoDatos.CorteVignes import filtradoVignes 
from PreprocesamientoDatos.PromedioPorVentana import n_orbita

#para fechas no clasificadas
def data_for_KNN(j: int, YYYY: str, MM: str, DD: str, orbita: int, df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    data.loc[j, ["Fecha", "orbita", "B", "Bx", "By", "Bz", "time", "posX", "posY", "posZ"]] = [
        f"{YYYY}-{MM}-{DD}", orbita, df["mod_B"].to_numpy(), df["time"].to_numpy(), 
        df["posX"].to_numpy(), df["posY"].to_numpy(), df["posZ"].to_numpy(),

    ]
    return data

def outbound_to_inbound(df):
    posx = df['posX'][0]
    posXdt = np.gradient(posx)
    type = np.mean(posXdt)
    if type>=0:
        return df
    if type<0: 
        return df.iloc[::-1]

def completeData(groups_of_dates: str, use_cache: bool) -> pd.DataFrame: 
    """
    Returns organized data for KNN 

    Arguments
    ---------
    groups_of_dates: str with name of the path with dates to be classified

    Returns
    -------
    pd.DataFrame with "Fecha", "orbita" and "B"
    """
    data_to_complete = pd.DataFrame(columns=["Fecha", "orbita", "B", "Bx", "By", "Bz", "time", "posX", "posY", "posZ"])
    data = cargarData(groups_of_dates)

    for i, (YYYY, MM, DD) in tqdm(enumerate(zip(data.YYYY, data.MM, data.DD), start=1), total=len(data), desc="Procesando fechas"):
        orbitas = n_orbita(YYYY, MM, DD)
        for n in range(1, orbitas): 
            df_to_classified = filtradoVignes(YYYY, MM, DD, n, use_cache)
            if len(df_to_classified)!=0:
                data_for_KNN(len(data_to_complete), YYYY, MM, DD, n, df_to_classified, data_to_complete)
                data_KNN_completed = outbound_to_inbound(data_to_complete)
    return data_KNN_completed

#para entrenamiento
def is_mpb_orbit(orbit_df: pd.DataFrame, mpb_times: pd.DataFrame, delta_sec: int) -> bool:
    """
    Returns True if the orbit has the time corresponded to a mpb crossing and 
    False in othe other case. 

    Arguments
    ---------
    orbit_df: time serie to be classified
    
    mpb_times: list of times of mpb crosses

    delta_sec: interval of seconds to look for the crossing

    Returns
    -------
    DTW distance between A and B
    """
    has_mpb = False
    for time in mpb_times:
        has_mpb = has_mpb or (abs(orbit_df["time"] - time) < (delta_sec / 3600)).any()
    return has_mpb

def data_for_train(j: int, YYYY: str, MM: str, DD: str, orbita: int, df: pd.DataFrame, data: pd.DataFrame, is_MPB_orbit: bool|int) -> pd.DataFrame:
    data.loc[j, ["Fecha", "orbita", "tipo", "B", "Bx", "By", "Bz", "time", "posX", "posY", "posZ"]] = [
        f"{YYYY}-{MM}-{DD}", orbita, int(is_MPB_orbit), df["mod_B"].to_numpy(), df["Bx"].to_numpy(), df["By"].to_numpy(),
         df["Bz"].to_numpy(), df["time"].to_numpy(), 
        df["posX"].to_numpy(), df["posY"].to_numpy(), df["posZ"].to_numpy(),
    ]
    return data

def trainingData(groups_of_dates: list[str], use_cache: bool) -> pd.DataFrame: 
    MPB_crosses_df = cargarTrainingData(groups_of_dates)
    data_to_complete = pd.DataFrame(columns=["Fecha", "orbita", "tipo", "B", "Bx", "By", 
                                             "Bz", "time", "posX", "posY", "posZ"])

    for i, (YYYY, MM, DD) in tqdm(enumerate(zip(MPB_crosses_df.YYYY, MPB_crosses_df.MM, MPB_crosses_df.DD), start=1), total=len(MPB_crosses_df), desc="Procesando fechas"):
        orbitas = n_orbita(YYYY, MM, DD)
        for n in range(1, orbitas): 
            df_not_marked = filtradoVignes(YYYY, MM, DD, n, use_cache)
            if len(df_not_marked)!=0:
                time_MPB = MPB_crosses_df.loc[(MPB_crosses_df['YYYY'] == YYYY) & (MPB_crosses_df['MM'] == MM) & (MPB_crosses_df['DD'] == DD)].MPB_time
                is_MPB_orbit = is_mpb_orbit(df_not_marked, time_MPB, 3)
                data_for_train(len(data_to_complete), YYYY, MM, DD, n, df_not_marked, data_to_complete, is_MPB_orbit)
                data_KNN_completed = outbound_to_inbound(data_to_complete)
    return data_KNN_completed

#DTW

@njit
def euclidean(x, y):
    return abs(x - y)/np.sqrt(x**2 + y**2)

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

#KNN

class KNN_timeSeries(BaseEstimator, ClassifierMixin):
    """K-nearest neighbor classifier using an indicated metric for series
    
    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN
        
    metric_calculator: str, optional (default= 'dtw') 
            Metric for measure distances between series
    
    use_weights: bool, optional (default=False)
            If false, knn will use a weight d_i/d_max
    """
    
    def __init__(self, metric_calculator, n_neighbors: int =5, use_weights: bool = False):
        self.use_weights = use_weights
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
        use_weights = self.use_weights

        # Identify the k nearest neighbors
        knn_idx = np.argsort(dm, axis=1)[:, :self.n_neighbors]
        knn_dists = np.take_along_axis(dm, knn_idx, axis=1)
        knn_labels = self.y_train[knn_idx]

        # Compute weights: w_i = d_k / d_i
        d_k = knn_dists[:, -1][:, np.newaxis]  
        weights = d_k / (knn_dists + 1e-9)

        # Weighted voting
        if use_weights: 
            unique_labels = np.unique(self.y_train)
            weighted_votes = np.zeros((X.shape[0], len(unique_labels)))
            for i, label in enumerate(unique_labels):
                weighted_votes[:, i] = np.sum(weights * (knn_labels == label), axis=1)

            predicted_labels = unique_labels[np.argmax(weighted_votes, axis=1)]
            confidence_scores = np.max(weighted_votes, axis=1) / np.sum(weighted_votes, axis=1)
        
        else: 
            predicted_labels = mode(knn_labels, axis = 1)[0]
            confidence_scores = mode(knn_labels, axis = 1)[1]/self.n_neighbors

        return predicted_labels, confidence_scores
    
        def get_params(self, deep=True):
            return {"n_neighbors": self.n_neighbors, "metric_calculator": self.metric_calculator}

        def set_params(self, **params):
            for param, value in params.items():
                setattr(self, param, value)
            return self

def cross_validation_KNN(n_splits: int, training_data, mww: int, K: int, use_weights: bool, folder: str) -> None:   
    """
    Performs cross-validation for KNN classifier using DTW distance metric
    Arguments
    ---------
    n_splits : int
        Number of splits for cross-validation
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
    if not os.path.exists(f'/app/KNN/Cross_Validation/{folder}'):
        os.makedirs(f'/app/KNN/Cross_Validation/{folder}')

    X = training_data['B']
    y = training_data['tipo'].to_numpy().astype(int)

    shuf = ShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state= 50)

    dtw_calculator = DTW(max_warping_window = mww)
    KNN = KNN_timeSeries(metric_calculator = dtw_calculator, n_neighbors = K, use_weights=use_weights)
    
    s = time.time()

    for i, (train_index, test_index) in enumerate(shuf.split(X, y)):          
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        KNN.fit(X_train, y_train)
        y_pred, y_prob = KNN.predict(X_test)
    
        print(f'{i+1} de 5 finalizado/s, {round((time.time()-s)/60,2)} minutos')
        print(len(y_pred))

        result = pd.DataFrame({'X_test': np.array(X_test), 'y_real': y_test, 
        'y_pred': y_pred, 'y_prob': y_prob})
        result['X_test'] = result['X_test'].apply(lambda x: ', '.join(map(str, x)))
    
        weighted_string = 'weighted' if use_weights else 'unweighted'
        path_file = f'/app/KNN/Cross_Validation/{folder}/CV_{K}vecinos_{mww}DTW_{i}_{weighted_string}.csv'
        result.to_csv(path_file)
        
if __name__== '__main__' :
    print('Descargando y ordenando data de entrenamiento')
    start_time = time.time()
    data_KNN_completed = trainingData(['Group1','Group2','Group3','Group4'], use_cache = True)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Tiempo de descarga y ordenado: {execution_time:.2f} segundos")

    cross_validation_KNN(n_splits=5, training_data=data_KNN_completed, mww=1000, K=1, use_weights=True, folder='Resultados')