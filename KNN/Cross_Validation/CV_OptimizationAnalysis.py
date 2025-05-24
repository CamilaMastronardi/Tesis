# Librerias basicas para manejo de datos
import numpy as np
import pandas as pd
import os
import sys
from typing import Union

#para la visualización
import matplotlib.pyplot as plt 
import seaborn as sns

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
    return dataframes, recall_weighted, recall_macro, recall_micro, recall_sep
    
def plot_todo_CV(cv_iter: list = [0,1,2,3,4]):

    Ks = [1, 3, 4, 5, 6, 7, 9, 11]
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 4, figsize=(14, 8))  # 8 subplots
    axs = axs.flatten()  # para indexarlos como lista

    for j, k in enumerate(Ks):
        numero_vecinos = k

        dataframes = {}
        total_data = pd.DataFrame()

        for i in cv_iter:
            path = f'KNN/Cross_Validation/ResultadosBinario_2min/CV_{numero_vecinos}vecinos_1000DTW_{i}_unweighted.csv'
            data = pd.read_csv(path, index_col=0)
            dataframes[f'iteration_{i}'] = data
            total_data = pd.concat([total_data, data], ignore_index=True)

        # Confusion matrix y heatmap
        cm = confusion_matrix(total_data['tipo'], total_data['pred'])
        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="white",
                    xticklabels=["No MPB", "MPB"],
                    yticklabels=["No MPB", "MPB"], fmt="d", cbar=False,
                    ax=axs[j], cmap="icefire", annot_kws={"size": 14}, )
        
        axs[j].set_title(f'{numero_vecinos} vecinos', fontsize=14)

        # Métricas
        y_true, y_pred = total_data['tipo'], total_data['pred']
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_sep = recall_score(y_true, y_pred, average=None)

        print(f'{numero_vecinos}, {recall_sep[1]}')
    
    fig.supxlabel('Predicción KNN', fontsize=16)
    fig.supylabel('Clasificación manual', fontsize=16)

    # Ajustar layout, guardar figura y mostrar
    plt.tight_layout()
    plt.savefig('KNN/Cross_Validation/Figuras_Cross_Val_binario/figura_heatmaps_todos_unweighted.png')
    plt.show()
    plt.close()

def scores_plot():
    path_unweighted = f'temp_unweighted.txt'
    path_weighted = f'temp_weighted.txt'

    data_weighted = pd.read_csv(path_weighted, sep = ',', names = ['k', 'score'], header = None)
    data_unweighted = pd.read_csv(path_unweighted, sep = ',', names = ['k', 'score'], header = None)

    ks  = data_weighted['k']
    score_w = data_weighted['score']
    score_u = data_unweighted['score']

    fig, ax = plt.subplots(figsize=(15,7))

    ax.plot(ks, score_w,linestyle = '--', lw = 2, color = 'darkturquoise')
    ax.scatter(ks, score_w, s = 200 ,marker='*', color = 'darkturquoise', label = 'KNN ponderado')
    ax.plot(ks, score_u, linestyle = '--', color = 'mediumorchid')
    ax.scatter(ks, score_u, s =100, marker='X', color = 'mediumorchid', label = 'KNN no ponderado')
    plt.legend(loc = (1.01,0.8), fontsize = 18)
    ax.set_xlabel('Número de Vecinos K', fontsize = 20)
    ax.set_ylabel('Recall Score', fontsize= 20)
    ax.set_xticks(ticks = [1,3,4,5,6,7,9,11])
    ax.tick_params(labelsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig('temp.png')


if __name__=='__main__': 

    Ks_w = [1,3,5,7,9,11,13,15]
    Ks_u = [1, 4, 7, 10, 13]
    for k in Ks_u:
        CV_scores(k)