import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.pyplot as plt 
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
import sktime

from DescargaTrainingData import cargarTrainingData

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from PreprocesamientoDatos.CorteVignes import filtradoVignes 
from PreprocesamientoDatos.PromedioPorVentana import n_orbita

def add_index(los_B: pd.Series) -> list[tuple[np.array, np.array]] :
    """
    Input: los_B es un pd.Series de arrays de valores
    Output: Una lista de tuplas de arrays (valores, indices)
    """
    return list([(np.array(range(len(val))), val) for val in los_B]) 
    
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

#toy dataset 
X = data_KNN_completed['B'].apply(lambda x: pd.DataFrame(x)).tolist()
y = data_KNN_completed['tipo']

print(type(X))
print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#sktime.datatypes.check_raise(X_train, 'df-list')

clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='dtw')
from sktime.registry import all_estimators
print(all_estimators(filter_tags={'capability:unequal_length':True},estimator_types='classifier'))
clf.fit(X_train, y_train)
