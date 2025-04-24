# Librerias basicas para manejo de datos
import numpy as np
import pandas as pd

#para la visualización
import matplotlib.pyplot as plt 
import seaborn as sns

#para estadistica
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit

from KNN import completeData, trainingData

from CV_OptimizationAnalysis import CV_scores

from AcomodarDatosB import acomodarDatos
from CorteVignes import _filtradoVignes 

total_data_1vecino, score_1vecino = CV_scores(1, cm = False, cv_iter = [0,1,2,3])

#SOLO PARA TEST DATA  
def testData(iteracion, train: bool = True) -> pd.DataFrame:
    if train:
        data = trainingData(['Group1','Group2','Group3','Group4'], use_cache=True)
    else:
        data = completeData(['Group1','Group2','Group3','Group4'])
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

def plot_B_vs_time(sep_date: pd.DataFrame):
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

def str_to_list(data: str) -> list: 
    '''
    Arguments: 

    data: str with data in a str of a list '[a, b, c, ...]'

    Returns
    list of floats containing the str data
    '''
    data = data.replace('[','')
    data = data.replace(']','')
    list_data = data.split(',')
    list_data = [float(x) for x in list_data]

    return list_data

path = "KNN/Clasificador/CampoMagnetico_2015_1vecinos_1000DTW.csv"

def plot_B_t(file_path: str): 
    df = pd.read_csv(path, sep = ',', index_col = 0)
    dates = df['Fecha']
    B = df['X']
    MPB = df['y_pred']
    time = df['time']
    
    last_date = None
    type = ''
    for i in range(len(dates)):
        YYYY, MM, DD = dates[i].split('-')

        raw_data = pd.read_csv(f'DatosCrudos/datos_campo_magnetico_crudos/datos_{DD}-{MM}-{YYYY}.csv')

        raw_data_acomodada = acomodarDatos(YYYY, MM, DD)
        B_raw = raw_data_acomodada['mod_B'].rolling(10).sum()/10
        time_raw = raw_data_acomodada['time']

        B_n = str_to_list(B[i])
        t_n = str_to_list(time[i])

        if dates[i]!= last_date:
            plt.figure(figsize=(24,7))
            plt.plot(time_raw[::-1], B_raw[::-1], label='B field', color='black', alpha = 0.5)
            type = ''
        last_date = dates[i]

        if MPB[i] == 1:
            plt.axvspan(t_n[0],t_n[-1], color='blue', alpha=0.3, label='MPB zone detected')
            type = 'detected'
        elif MPB[i] == 0:
            plt.axvspan(t_n[0],t_n[-1], color='red', alpha=0.3, label='Analized zone')

        plt.xlabel('Time', fontsize = 20)
        plt.ylabel('|B|', fontsize = 20)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.ylim(-1,60)
        plt.title(f'B field on {YYYY}-{MM}-{DD}', fontsize = 24)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', 
                   bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize = 18)
        plt.tight_layout()
            
        plt.savefig(f'KNN/Clasificador/Figuras/{dates[i]}_{type}_reconstruccion.png')

#plot_B_t(path)