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
from DescargaTrainingData import cargarTrainingData, cargarData
from PreprocesamientoDatos.CorteVignes import filtradoVignes 
from PreprocesamientoDatos.PromedioPorVentana import n_orbita
from KNN import cross_validation_KNN, outbound_to_inbound, data_for_train
from Algoritmo3clusters import is_orbit_clasified, cut_orbits_in_n_min

def trainingDataBinario(groups_of_dates: list[str], use_cache: bool, mins: int) -> pd.DataFrame: 
    MPB_crosses_df = cargarTrainingData(groups_of_dates)
    data_to_complete = pd.DataFrame(columns=["Fecha", "orbita", "tipo", "B", "Bx", "By", 
                                             "Bz", "time", "posX", "posY", "posZ"])

    for i, (YYYY, MM, DD) in tqdm(enumerate(zip(MPB_crosses_df.YYYY, MPB_crosses_df.MM, MPB_crosses_df.DD), start=1), total=len(MPB_crosses_df), desc="Procesando fechas"):
        orbitas = n_orbita(YYYY, MM, DD)
        for n in range(1, orbitas): 
            df_not_marked = filtradoVignes(YYYY, MM, DD, n, use_cache, 50, 50)
            time_MPB = MPB_crosses_df.loc[(MPB_crosses_df['YYYY'] == YYYY) & (MPB_crosses_df['MM'] == MM) & (MPB_crosses_df['DD'] == DD)].MPB_time
            for mpb in time_MPB:
                if is_orbit_clasified(df_not_marked, mpb):
                    pedacitos = cut_orbits_in_n_min(df_not_marked, mins)
                    for pedacito in pedacitos:
                        pedacito_inbound = outbound_to_inbound(pedacito)
                        if is_orbit_clasified(pedacito, mpb):
                            data_for_train(len(data_to_complete), YYYY, MM, DD, n, pedacito_inbound, data_to_complete, 1)
                        else:
                            data_for_train(len(data_to_complete), YYYY, MM, DD, n, pedacito_inbound, data_to_complete, 0)
                else:
                    continue
    return data_to_complete

if __name__== '__main__' :
    # Descargar Training data
    print('Descargando y ordenando data de entrenamiento')
    start_time = time.time()
    data_KNN_completed = trainingDataBinario(['Group1','Group2','Group3','Group4'], use_cache = True, mins = 2)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Tiempo de descarga y ordenado: {execution_time:.2f} segundos")

    for k in range(10, 20, 2):
        cross_validation_KNN(n_splits=5, training_data=data_KNN_completed, mww=1000, K=k, use_weights=True, folder = 'ResultadosBinario_2min')   

