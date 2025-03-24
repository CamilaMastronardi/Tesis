# Librerias basicas para manejo de datos
import numpy as np
import pandas as pd
import os
import sys

#para la visualización
import matplotlib.pyplot as plt 
import seaborn as sns

#para estadistica
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit

root_dir = os.path.dirname(os.path.dirname(__file__))
from KNN import completeData

sys.path.append(os.path.join(root_dir, 'KNN/Cross_Validation'))
from CV_OptimizationAnalysis import CV_scores

sys.path.append(os.path.join(root_dir, 'PreprocesamientoDatos'))
from AcomodarDatosB import acomodarDatos
from CorteVignes import _filtradoVignes 

total_data_1vecino, score_1vecino = CV_scores(1, cm = False)

#SOLO PARA TEST DATA  
def testData(iteracion) -> pd.DataFrame:
    data = completeData()
    dates = data['Fecha']
    orbit = data['orbita']
    shuf = ShuffleSplit(n_splits=5, test_size=0.2, random_state= 50)
    reordered_dates = pd.DataFrame(index=None)
    reordered_orbits = pd.DataFrame(index=None)
    for i, (train_index, test_index) in enumerate(shuf.split(dates)):
        dates_fold = dates[test_index]
        orbit_fold = orbit[test_index]
        reordered_dates = pd.concat([reordered_dates, dates_fold])
        reordered_orbits = pd.concat([reordered_orbits, orbit_fold])
        if i == iteracion:
            return reordered_dates.reset_index(), reordered_orbits.reset_index()

iteracion0_1vecino = total_data_1vecino['iteration_0']
dates, orbits = testData(iteracion=0)
todo_junto = pd.concat([iteracion0_1vecino, dates, orbits], axis=1)

sep_date = [grupo for _, grupo in todo_junto.groupby('Fecha')]

def plot_B_vs_time(sep_date):
    for j in range(len(sep_date)): 
        data = sep_date[j].reset_index()
        date = data['Fecha'].iloc[0]
        YYYY, MM, DD = date.split('-')
        orbits = data['orbita']

        raw = acomodarDatos(YYYY, MM, DD)
        time = raw['time']
        B = raw['mod_B']

        plot_flag = False
        fig, ax = plt.subplots(figsize=(24, 7))
        ax.plot(time, B, label='B field', color='black', alpha=0.5)

        for n in orbits: 
            mask = data['orbita'] == n
            y_pred = data.loc[mask, 'y_pred'].values[0]
            y_real = data.loc[mask, 'y_real'].values[0]

            if y_pred == y_real == 1:
                plot_flag = True
                event_data = _filtradoVignes(YYYY, MM, DD, n)
                ax.axvspan(event_data['time'].iloc[0], event_data['time'].iloc[-1], color='green', alpha=0.3, label='True Positive')
            
            elif y_pred == 1 and y_real == 0:
                plot_flag = True
                event_data = _filtradoVignes(YYYY, MM, DD, n)
                ax.axvspan(event_data['time'].iloc[0], event_data['time'].iloc[-1], color='red', alpha=0.3, label='False Positive')
            
            elif y_pred == 0 and y_real == 1:
                plot_flag = True
                event_data = _filtradoVignes(YYYY, MM, DD, n)
                ax.axvspan(event_data['time'].iloc[0], event_data['time'].iloc[-1], color='blue', alpha=0.3, label='False Negative')

        if plot_flag:
            ax.set_xlabel('Time')
            ax.set_ylabel('|B|')
            ax.set_title(f'B field on {YYYY}-{MM}-{DD}')
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(f'temp{j}.png')
            plt.show()
        else:
            plt.close(fig)


plot_B_vs_time(sep_date)