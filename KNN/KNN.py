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

def mark_mpb_crossing(df: pd.DataFrame, times: pd.DataFrame, delta_sec: int) -> pd.DataFrame:
    df['MPB_crossing'] = False
    for time in times:
        df["MPB_crossing"] |= (abs(df["time"] - time) < (n / 3600))
    return df

def plot_b_vs_time(df: pd.DataFrame, YYYY: int, MM: int, DD: int, orbita: int):
    plt.figure(figsize=(10, 5))

    # Plot points where MPB_crossing is True
    plt.scatter(df.loc[df["MPB_crossing"]==True]["time"], df.loc[df["MPB_crossing"]==True]["mod_B"], 
                color="red", label="MPB Crossing", s=50)

    # Plot points where MPB_crossing is False
    plt.scatter(df.loc[df["MPB_crossing"]==False]["time"], df.loc[df["MPB_crossing"]==False]["mod_B"], 
                color="blue", label="Non-MPB Crossing", s=10)

    plt.xlabel("Time")
    plt.ylabel("B")
    plt.legend()
    plt.title(f"MPB: {len(df.loc[df['MPB_crossing']==True])} , Not MPB: {len(df.loc[df['MPB_crossing']==False])}")
    plt.savefig(f'temp_{YYYY}_{MM}_{DD}_{orbita}.jpg')

MPB_crosses_df = cargarTrainingData(group='Group2')
i = 1
for YYYY, MM, DD in zip(MPB_crosses_df.YYYY, MPB_crosses_df.MM, MPB_crosses_df.DD):
    orbitas = n_orbita(YYYY, MM, DD)
    for n in range(1, orbitas): 
        df_not_marked = filtradoVignes(YYYY, MM, DD, n)
        time_MPB = MPB_crosses_df.loc[(MPB_crosses_df['YYYY'] == YYYY) & (MPB_crosses_df['MM']== MM) & (MPB_crosses_df['DD']==DD)].MPB_time
        df_marked = mark_mpb_crossing(df_not_marked, time_MPB, 2)
        plot_b_vs_time(df_marked, YYYY, MM, DD, n)
    
    
        
    print(f'fecha {i} de {len(MPB_crosses_df)}')
    i = i + 1

'''
training_data = 
validation_data = 
'''