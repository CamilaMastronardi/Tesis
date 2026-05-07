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
from DescargaTrainingData import cargarTrainingData
from PreprocesamientoDatos.CorteVignes import filtradoVignes 
from PreprocesamientoDatos.PromedioPorVentana import n_orbita

def reconstruction_orbits(numero_vecinos: int ,cv_iter: list = [0,1,2,3,4], use_weights: bool = True):
    dataframes = {} 
    total_data = pd.DataFrame()
    if use_weights == True:
        is_weighted_string = 'weighted'
    else: 
        is_weighted_string = 'unweighted'
    
    for cv_index in cv_iter: 
        path = f'KNN/Cross_Validation/Resultados3clusters_2min/CV_{numero_vecinos}vecinos_1000DTW_{cv_index}_{is_weighted_string}.csv'
        data = pd.read_csv(path, index_col = 0)
        dataframes[f'iteration_{cv_index}'] = pd.DataFrame(data)
        total_data = pd.concat([dataframes[f'iteration_{cv_index}'], total_data], ignore_index=True)
    
    grupos = total_data.groupby(['Fecha','orbita'])
    df_igual_orbita = {
    (fecha, orbita): grupo.reset_index(drop=True)
    for (fecha, orbita), grupo in grupos}
    return df_igual_orbita

def mpb_y_bs_orbita(YYYY: str, MM: str, DD: str, df_orbita: pd.DataFrame, bs_mpb_times: pd.DataFrame): 
    MPB_times = bs_mpb_times[(bs_mpb_times['YYYY'] == YYYY) & (bs_mpb_times['MM'] == MM) & (bs_mpb_times['DD'] == DD)]['MPB_time']
    BS_times = bs_mpb_times[(bs_mpb_times['YYYY'] == YYYY) & (bs_mpb_times['MM'] == MM) & (bs_mpb_times['DD'] == DD)]['BS_time']
    t_min = df_orbita['time'].min()
    t_max = df_orbita['time'].max()
    mpb_time = None
    bs_time = None
    for mpb in MPB_times:
        if mpb < t_min or mpb > t_max:
            continue
        else:
            mpb_time = mpb
    for bs in BS_times:
        if bs < t_min or bs > t_max:
            continue
        else:
            bs_time = bs
    
    return mpb_time, bs_time


def plot_orbit_reconstructed(data_tot):
        dict_keys = data_tot.keys()
        bs_mpb_times = cargarTrainingData(['Group1','Group2','Group3','Group4'])

        for j,key in enumerate(dict_keys): 
            fecha, orbita = key
            YYYY, MM, DD = fecha.split('-')
            df = filtradoVignes(YYYY, MM, DD, int(orbita), band_size_min= 50, band_size_max=120)
            data = data_tot[key]

            mpb_time, bs_time = mpb_y_bs_orbita(YYYY, MM, DD, df, bs_mpb_times)

            fig, ax = plt.subplots(figsize = (18,10))
            ax.plot(df['time'], df['mod_B'], color = 'lightsteelblue')
            if mpb_time != None:
                ax.axvline(x=mpb_time, color='crimson', linestyle='--', lw = 2, label = 'MPB')
            if bs_time != None:
                ax.axvline(x=bs_time, color='blue', linestyle='--', lw = 2, label = 'BS')
            for i in range(len(data)):
                t_str = data['time'].iloc[i]
                b_str = data['B'].iloc[i]
                t = [float(ti) for ti in t_str.split(',')]
                B = [float(bi) for bi in b_str.split(',')]

                y_pred = data['pred'].iloc[i]
                y_true = data['tipo'].iloc[i]

                if y_pred == 0:
                    ax.plot(t, B, color = 'mediumspringgreen', label = 'SW')
                    
                elif y_true == 1:
                    ax.plot(t, B, color = 'darkorchid', label = 'Magnetosfera')
                
                elif y_true==2:
                    ax.plot(t, B, color = 'teal', label = 'Ionosfera')
                            
            #plt.legend()
            plt.tick_params(labelsize=25)
            plt.tight_layout()
            plt.grid(color='black')
            plt.savefig(f'KNN/Cross_Validation/Figuras_Cross_Val_3clusters/3C_{YYYY}-{MM}-{DD}_orbita{orbita}.png')
            plt.close()

def plot_bin_tesis(data_tot, type: str):
    keys_Lindos = [("2015-02-15",3),("2015-02-17",3), ("2017-07-30",3), ("2020-09-15", 5)]
    keys_Feos = [("2015-10-02",1), ("2017-07-29",2)]

    if type == 'L':
        keys = keys_Lindos
    if type == 'F':
        keys = keys_Feos

    fig, axs = plt.subplots(1, 2, figsize=(20,6))
    axs_flat = axs.flat
    bs_mpb_times = cargarTrainingData(['Group1','Group2','Group3','Group4'])
    for (key, ax) in zip(keys,axs_flat):
        fecha, orbita = key
        YYYY, MM, DD = fecha.split('-')
        df = filtradoVignes(YYYY, MM, DD, int(orbita), band_size_min= 50, band_size_max=120)
        data = data_tot[key]

        mpb_time, bs_time = mpb_y_bs_orbita(YYYY, MM, DD, df, bs_mpb_times)

        ax.plot(df['time'], df['mod_B'], color = 'lightsteelblue', label = f'{fecha}  ')
        ax.legend(loc='best', handlelength=0)
        if mpb_time != None:
            ax.axvline(x=mpb_time, color='crimson', linestyle='--', lw = 2, label = 'MPB', zorder=1000)
        if bs_time != None:
            ax.axvline(x=bs_time, color='blue', linestyle='--', lw = 2, label = 'BS', zorder = 1000)
        for i in range(len(data)):
            t_str = data['time'].iloc[i]
            b_str = data['B'].iloc[i]
            t = [float(ti) for ti in t_str.split(',')]
            B = [float(bi) for bi in b_str.split(',')]

            y_pred = data['pred'].iloc[i]
            y_true = data['tipo'].iloc[i]

            if y_pred == 0:
                ax.plot(t, B, color = 'mediumspringgreen', label = 'SW')
                
            elif y_true == 1:
                ax.plot(t, B, color = 'darkorchid', label = 'Magnetosfera')
            
            elif y_true==2:
                ax.plot(t, B, color = 'teal', label = 'Ionosfera')
        
        ax.tick_params(labelsize=25)
        ax.grid(True, color='black')
    axs[0].set_ylabel( 'Campo magnético |B| (nT)', fontsize= 30)
    fig.text(0.5,-0.03, 'Tiempo (hs)', ha= 'center', fontsize= 30)
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[1], handles[2], handles[3], handles[4], handles[8]]

    fig.legend(bbox_to_anchor = (0.9, 1.2), handles= handles, ncols = 5)
    plt.subplots_adjust(wspace=0.5)
    plt.tick_params(labelsize=25)
    plt.grid(color='black')
    plt.tight_layout()
    plt.savefig(f'KNN/Cross_Validation/Figuras_Cross_Val_3clusters/3Ctot.png', bbox_inches = 'tight')
    plt.close()
        
data_tot = reconstruction_orbits(30, use_weights = True)
plot_bin_tesis(data_tot, type = 'F')

