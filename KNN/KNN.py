import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
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

MPB_crosses_df = cargarTrainingData(group='Group3')
i = 1
for YYYY, MM, DD in zip(MPB_crosses_df.YYYY, MPB_crosses_df.MM, MPB_crosses_df.DD):
    orbitas = n_orbita(YYYY, MM, DD)
    for n in range(1, orbitas): 
        df_not_marked = filtradoVignes(YYYY, MM, DD, n)
        time_MPB = MPB_crosses_df.loc[(MPB_crosses_df['YYYY'] == YYYY) & (MPB_crosses_df['MM']== MM) & (MPB_crosses_df['DD']==DD)].MPB_time
        is_MPB_orbit = is_mpb_orbit(df_not_marked, time_MPB, 3)
    
    print(f'fecha {i} de {len(MPB_crosses_df)}')
    print(f'{is_MPB_orbit}')
    i = i + 1

'''
training_data = 
validation_data = 
'''