# Librerias basicas para manejo de datos
import numpy as np
import pandas as pd
import os
import sys
from typing import Union

#para la visualización
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.gridspec as gridspec

#para estadistica
import sklearn.metrics
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

def CV_scores(numero_vecinos: int, cm: bool = True, 
    cv_iter: list = [0,1,2,3,4]) -> Union[pd.DataFrame , float]:
    
    dataframes = {} 
    total_data = pd.DataFrame()
    for i in cv_iter:
        path = f'KNN/Cross_Validation/Resultados3clusters_2min/CV_{numero_vecinos}vecinos_1000DTW_{i}_unweighted.csv'
        data = pd.read_csv(path, index_col = 0)
        dataframes[f'iteration_{i}'] = pd.DataFrame(data)

        total_data = pd.concat([dataframes[f'iteration_{i}'], total_data], ignore_index=True)

    if cm:
        f, ax =plt.subplots(figsize = (5,5))
        cm = confusion_matrix(total_data['tipo'], 
                                total_data['pred'])
        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="white", xticklabels=["No MPB", "MPB"], 
                    yticklabels=["No MPB", "MPB"], fmt="d", cbar=False, ax=ax, cmap = "icefire")
        plt.title(f'{numero_vecinos} vecinos', fontsize = 18)
        plt.xlabel("Predicción KNN", fontsize = 14)
        plt.ylabel("Clasificación manual", fontsize = 14)
        plt.savefig(f'KNN/Cross_Validation/Figuras_Cross_Val_3clusters/cv_fig{numero_vecinos}vecinos_unweighted.png')
        plt.show()
        plt.close()

    y_true, y_pred = total_data['tipo'], total_data['pred']
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_sep = recall_score(y_true, y_pred, average=None)
    recall_2 = (recall_sep[1]+recall_sep[2])/2

    print(f'{numero_vecinos}, {recall_weighted}, {recall_2}')
    return dataframes, recall_weighted, recall_macro, recall_micro, recall_sep
    
def plot_todo_CV(cv_iter: list = [0,1,2,3,4], Ks: list = [1,3,4,5,6,7,9,11]):
    
    fig = plt.figure(figsize=(5, 5))  # 8 subplots
    gs = gridspec.GridSpec(1, 1, figure=fig)
    axs = [[],[],[]]
    axs[0] = fig.add_subplot(gs[0, 0:1])
    #axs[1] = fig.add_subplot(gs[0, 1:2])
    #axs[2] = fig.add_subplot(gs[0, 2:3])
    

    for j, k in enumerate(Ks):
        numero_vecinos = k

        dataframes = {}
        total_data = pd.DataFrame()

        for i in cv_iter:
            path = f'KNN/Cross_Validation/Resultados3clusters_2min/CV_{numero_vecinos}vecinos_1000DTW_{i}_unweighted.csv'
            data = pd.read_csv(path, index_col=0)
            dataframes[f'iteration_{i}'] = data
            total_data = pd.concat([total_data, data], ignore_index=True)

        # Confusion matrix y heatmap
        cm = confusion_matrix(total_data['tipo'], total_data['pred'])
        sns.set(font_scale=1.4)
        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="white",
                    xticklabels=["SW", "Magnetofunda", "Ionosfera"],
                    yticklabels=["SW", "Magnetofunda", "Ionosfera"], fmt="d", cbar=False,
                    ax=axs[j], cmap="icefire", annot_kws={"size": 16}, )
        
        axs[j].set_title(f'{numero_vecinos} vecinos', fontsize=20)
        axs[j].tick_params('x', rotation=0)
        plt.xlabel("Predicción KNN", fontsize = 20)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylabel("Clasificación manual", fontsize = 20)
        # Métricas
        y_true, y_pred = total_data['tipo'], total_data['pred']
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_sep = recall_score(y_true, y_pred, average=None)
        recall_2 = (recall_sep[1]+recall_sep[2])/2

        print(f'{numero_vecinos}, {recall_2}')


    # Ajustar layout, guardar figura y mostrar
    plt.tight_layout()
    plt.savefig('KNN/Cross_Validation/Figuras_Cross_Val_3clusters/figura_heatmaps_todos_unweighted.png')
    plt.show()
    plt.close()

def scores_plot():
    path_unweighted = f'temp_3c_u.txt'
    path_weighted = f'temp_3c_w.txt'

    data_weighted = pd.read_csv(path_weighted, sep = ',', names = ['k', 'score_w', 'score_2'], header = None)
    data_unweighted = pd.read_csv(path_unweighted, sep = ',', names = ['k', 'score'], header = None)

    ks_w  = data_weighted['k']
    ks_u  = data_unweighted['k']
    score_w = data_weighted['score_w']
    score_u = data_unweighted['score']
    score_2 = data_weighted['score_2']

    fig, ax = plt.subplots(figsize=(12,7))

    ax.plot(ks_w, score_w,linestyle = '--', lw = 2, color = 'darkturquoise')
    ax.scatter(ks_w, score_w, s = 200 ,marker='*', color = 'darkturquoise', label = '3C ponderado')
    ax.plot(ks_u, score_u,linestyle = '--', lw = 2, color = 'darkblue')
    ax.scatter(ks_u, score_u, s = 100 ,marker='o', color = 'darkblue', label = '3C no ponderado')
    ax.legend(loc = (0.55,0.05), fontsize = 22)
    ax.set_xlabel('Número de Vecinos K', fontsize = 24)
    ax.set_ylabel('Recall Score', fontsize= 24)
    ax.grid(True)
    ax.set_xticks(ticks = [1,3,5,7,9,11,13,15, 20, 25, 30])
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig('temp.png')


if __name__=='__main__': 
    scores_plot()
