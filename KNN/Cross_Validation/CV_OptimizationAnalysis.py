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

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

def CV_scores(numero_vecinos, cm = True, score_method = sklearn.metrics.recall_score):
    dataframes = {} 
    tot_data = {}
    for i in range(0,5):
        path = f'KNN/Cross_Validation/Resultados/CV_{numero_vecinos}vecinos_1000DTW_{i}.csv'
        data = pd.read_csv(path, index_col = 0)
        dataframes[f'iteration_{i}'] = pd.DataFrame(data)

    total_data = pd.concat([dataframes['iteration_0'],dataframes['iteration_1'],dataframes['iteration_2'],
                    dataframes['iteration_3'],dataframes['iteration_4']], ignore_index=True)

    if cm:
        f, ax =plt.subplots(figsize = (5,5))
        cm = confusion_matrix(total_data['y_real'], 
                                total_data['y_pred'])
        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="white", xticklabels=["No MPB", "MPB"], 
                    yticklabels=["No MPB", "MPB"], fmt="d", cbar=False, ax=ax, cmap = "icefire")
        plt.title(f'{numero_vecinos} vecinos', fontsize = 18)
        plt.xlabel("Predicción KNN", fontsize = 14)
        plt.ylabel("Clasificación manual", fontsize = 14)
        plt.savefig(f'KNN/Figuras_Cross_Val/{numero_vecinos}vecinos_total.png')

    score = score_method(total_data['y_real'], total_data['y_pred'])
    return total_data, score
    
if __name__=='__main__': 
    for i in range(1,4):
        data, score = CV_scores(i, cm= False)
        print(f'Recall score f{i} vecino/s {round(score,2)}')