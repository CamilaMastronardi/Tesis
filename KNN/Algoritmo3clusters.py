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
from DescargaTrainingData import cargarTrainingData
from PreprocesamientoDatos.CorteVignes import filtradoVignes 
from PreprocesamientoDatos.PromedioPorVentana import n_orbita
from KNN import data_for_train, cross_validation_KNN

def clusterize(orbit_data: pd.DataFrame, MPB_time: float, BS_time: float) -> list[pd.DataFrame]:
    """
    Returns three dataframes: SW, magnetosheath and ionosphere.

    Arguments
    ---------
    data: Dataframe with the data to be clustered
    MPB_time: time of the mpb crossing
    BS_time: time of the bow shock crossing
    """
    if BS_time < MPB_time:
        SW = orbit_data[(orbit_data["time"] < BS_time)]
        magnetosheath = orbit_data[(orbit_data["time"] > BS_time) & (orbit_data["time"] < MPB_time)]
        ionosphere = orbit_data[(orbit_data["time"] > MPB_time)]
        return [SW, magnetosheath, ionosphere]
    else:
        SW = orbit_data[(orbit_data["time"] > BS_time)]
        magnetosheath = orbit_data[(orbit_data["time"] < BS_time) & (orbit_data["time"] > MPB_time)]
        ionosphere = orbit_data[(orbit_data["time"] < MPB_time)]
        return [SW, magnetosheath, ionosphere]


def is_orbit_clasified(orbit_df: pd.DataFrame, mpb_time: float) -> bool:
    has_mpb = (orbit_df["time"]).min() < mpb_time and mpb_time < max(orbit_df["time"])
    return has_mpb

def trainingData3Clusters(groups_of_dates: list[str], use_cache: bool, mins: int) -> pd.DataFrame: 
    MPB_crosses_df = cargarTrainingData(groups_of_dates)
    data_to_complete = pd.DataFrame(columns=["Fecha", "orbita", "tipo", "B", "Bx", "By", 
                                             "Bz", "time", "posX", "posY", "posZ"])

    for i, (YYYY, MM, DD) in tqdm(enumerate(zip(MPB_crosses_df.YYYY, MPB_crosses_df.MM, MPB_crosses_df.DD), start=1), total=len(MPB_crosses_df), desc="Procesando fechas"):
        orbitas = n_orbita(YYYY, MM, DD)
        for n in range(1, orbitas): 
            df_not_marked = filtradoVignes(YYYY, MM, DD, n, use_cache, 50, 120)
            time_MPB = MPB_crosses_df.loc[(MPB_crosses_df['YYYY'] == YYYY) & (MPB_crosses_df['MM'] == MM) & (MPB_crosses_df['DD'] == DD)].MPB_time
            time_BS = MPB_crosses_df.loc[(MPB_crosses_df['YYYY'] == YYYY) & (MPB_crosses_df['MM'] == MM) & (MPB_crosses_df['DD'] == DD)].BS_time
            for mpb, bs in zip(time_MPB, time_BS):
                if is_orbit_clasified(df_not_marked, mpb):
                    SW, magnetosheath, ionosphere = clusterize(df_not_marked, mpb, bs)
                    pedacitos_SW = cut_orbits_in_n_min(SW, mins)
                    pedacitos_magnetosheath = cut_orbits_in_n_min(magnetosheath, mins)
                    pedacitos_ionosphere = cut_orbits_in_n_min(ionosphere, mins)
                    for pedacito in pedacitos_SW:
                        data_for_train(len(data_to_complete), YYYY, MM, DD, n, pedacito, data_to_complete, 0)
                    for pedacito in pedacitos_magnetosheath:
                        data_for_train(len(data_to_complete), YYYY, MM, DD, n, pedacito, data_to_complete, 1)
                    for pedacito in pedacitos_ionosphere:   
                        data_for_train(len(data_to_complete), YYYY, MM, DD, n, pedacito, data_to_complete, 2)
                else:
                    continue
    return data_to_complete

def cut_orbits_in_n_min(data: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    lista_pedacitos = [data[x:x+n*60] for x in range(0, len(data), n*60)]
    result = []
    for pedacito in lista_pedacitos:
        if len(pedacito) > n/2*60:
            result.append(pedacito)
    return result


if __name__== '__main__' :
    # Descargar Training data
    print('Descargando y ordenando data de entrenamiento')
    start_time = time.time()
    data_KNN_completed = trainingData3Clusters(['Group1','Group2','Group3','Group4'], use_cache = True, mins = 10)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Tiempo de descarga y ordenado: {execution_time:.2f} segundos")

    for k in range(1, 16, 3):
        cross_validation_KNN(n_splits=5, training_data=data_KNN_completed, mww=1000, K=k, use_weights=False, folder = 'Resultados3clusters_2min')   
