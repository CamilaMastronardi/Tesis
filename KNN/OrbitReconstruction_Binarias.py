# Librerias basicas para manejo de datos
import numpy as np
import pandas as pd

#para la visualización
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.patches as mpatches
plt.style.use("./matplotlibStyles.txt")

# Librerias basicas para manejo de datos
import numpy as np
import pandas as pd
import os
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

# Funciones de otros archivos
from DescargaTrainingData import cargarData
from PreprocesamientoDatos.CorteVignes import filtradoVignes 
from PreprocesamientoDatos.PromedioPorVentana import n_orbita

def reconstruction_orbits(numero_vecinos: int, cv_iter: list = [0,1,2,3,4], use_weights: bool = True):
    dataframes = {} 
    total_data = pd.DataFrame()
    if use_weights == True:
        is_weighted_string = 'weighted'
    else: 
        is_weighted_string = 'unweighted'
    
    for cv_index in cv_iter: 
        path = f'KNN/Cross_Validation/ResultadosBinario_2min/CV_{numero_vecinos}vecinos_1000DTW_{cv_index}_{is_weighted_string}.csv'
        data = pd.read_csv(path, index_col = 0)
        dataframes[f'iteration_{cv_index}'] = pd.DataFrame(data)
        total_data = pd.concat([dataframes[f'iteration_{cv_index}'], total_data], ignore_index=True)
    
    grupos = total_data.groupby(['Fecha','orbita'])
    df_igual_orbita = {
    (fecha, orbita): grupo.reset_index(drop=True)
    for (fecha, orbita), grupo in grupos}

    return df_igual_orbita

def plot_orbit_reconstructed(data_tot):
        dict_keys = data_tot.keys()

        for j,key in enumerate(dict_keys): 
            fecha, orbita = key
            YYYY, MM, DD = fecha.split('-')
            df = filtradoVignes(YYYY, MM, DD, int(orbita))
            data = data_tot[key]
            t_marked = []

            fig, ax = plt.subplots(figsize = (18,10))
            ax.plot(df['time'], df['mod_B'], color = 'darkturquoise')
            for i in range(len(data)):
                t_str = data['time'].iloc[i]
                t = [float(ti) for ti in t_str.split(',')]

                y_pred = data['pred'].iloc[i]
                y_true = data['tipo'].iloc[i]
                t_min = min(t)
                t_max = max(t)
                if y_true ==1:
                    if y_pred == 1: 
                        if ([t_min,t_max] in t_marked):
                            continue
                        else:
                            ax.axvspan(t_min, t_max, color = 'green', alpha = 0.1, label = 'MPB detectada')
                            t_marked.append([t_min,t_max])
                    else:
                        if ([t_min,t_max] in t_marked):
                            continue
                        else:
                            ax.axvspan(t_min, t_max, color = 'red', alpha = 0.2, label = 'MPB no detectada')
                            t_marked.append([t_min,t_max])
                
                elif y_pred==1:
                    if ([t_min,t_max] in t_marked):
                        continue
                    else:
                        ax.axvspan(t_min, t_max, color = 'gold', alpha = 0.3, label = 'Falsa MPB')
                        t_marked.append([t_min,t_max])
                
                elif y_pred==0: 
                    continue
            
            plt.legend()
            plt.tick_params(labelsize=25)
            plt.tight_layout()
            plt.grid(color='black')
            plt.savefig(f'rec_{YYYY}-{MM}-{DD}_orbita{orbita}.png')
            plt.close()

def plot_bin_tesis(data_tot, type: str):
    keys_FP = [('2015-02-15', 1),('2018-07-12', 1), ('2018-12-13', 1), ('2015-02-17', 3)]
    keys_FN = [('2015-01-02', 3), ('2015-01-08', 5),('2018-11-22',1), ('2020-09-18', 2)]
    keys_TP = [('2018-07-05', 4), ('2020-09-24', 4), ('2018-09-30', 3), ('2018-10-10', 1)]

    if type == 'TP':
        keys = keys_TP
    if type == 'FP':
        keys = keys_FP
    if type == 'FN':
        keys = keys_FN

    fig, axs = plt.subplots(2, 2, figsize=(20,12))
    axs_flat = axs.flat
    for (key, ax) in zip(keys,axs_flat):
        fecha, orbita = key
        YYYY, MM, DD = fecha.split('-')
        data = data_tot[key]
        t_marked = []

        df = filtradoVignes(YYYY, MM, DD, int(orbita))
        ax.plot(df['time'], df['mod_B'], color = 'darkturquoise', label = f'{fecha}  ')
        ax.legend(loc='best', handlelength=0)

        for i in range(len(data)):
            t_str = data['time'].iloc[i]
            t = [float(ti) for ti in t_str.split(',')]

            y_pred = data['pred'].iloc[i]
            y_true = data['tipo'].iloc[i]
            t_min = min(t)
            t_max = max(t)
            if y_true ==1:
                if y_pred == 1: 
                    if ([t_min,t_max] in t_marked):
                        continue
                    else:
                        ax.axvspan(t_min, t_max, color = 'green', alpha = 0.1)
                        t_marked.append([t_min,t_max])
                else:
                    if ([t_min,t_max] in t_marked):
                        continue
                    else:
                        ax.axvspan(t_min, t_max, color = 'red', alpha = 0.2)
                        t_marked.append([t_min,t_max])
            
            elif y_pred==1:
                if ([t_min,t_max] in t_marked):
                    continue
                else:
                    ax.axvspan(t_min, t_max, color = 'gold', alpha = 0.3)
                    t_marked.append([t_min,t_max])
            
            elif y_pred==0: 
                continue
        
        ax.tick_params(labelsize=25)
        ax.grid(True, color='black')
    fig.text(0.08, 0.25, 'Campo magnético |B| (nT)', ha= 'center', rotation = 90,fontsize= 30)
    fig.text(0.5, 0.03, 'Tiempo (hs)', ha= 'center', fontsize= 30)
    red = mpatches.Patch(color='red', alpha = 0.2, label='MPB no detectada')
    yellow = mpatches.Patch(color='gold', alpha = 0.3, label='Falsa MPB')
    green = mpatches.Patch(color='green', alpha = 0.1, label='MPB detectada')
    handles = [red, yellow, green]
    fig.legend(bbox_to_anchor = (0.8, 0.52), handles=handles, ncols = 3)
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f'rec_bin_{type}.png', bbox_inches = 'tight')
    plt.close()
        
        


data_tot = reconstruction_orbits(1)
plot_orbit_reconstructed(data_tot)