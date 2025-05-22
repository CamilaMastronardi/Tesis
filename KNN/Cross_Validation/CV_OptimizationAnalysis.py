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
    score_method: callable = sklearn.metrics.recall_score, 
    cv_iter: list = [0,1,2,3,4]) -> Union[pd.DataFrame , float]:
    
    dataframes = {} 
    total_data = pd.DataFrame()
    for i in cv_iter:
        path = f'KNN/Cross_Validation/Resultados3clusters_2min/CV_{numero_vecinos}vecinos_1000DTW_{i}_weighted.csv'
        data = pd.read_csv(path, index_col = 0)
        dataframes[f'iteration_{i}'] = pd.DataFrame(data)

        total_data = pd.concat([dataframes[f'iteration_{i}'], total_data], ignore_index=True)

    if cm:
        f, ax =plt.subplots(figsize = (5,5))
        cm = confusion_matrix(total_data['y_real'], 
                                total_data['y_pred'])
        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="white", xticklabels=["No MPB", "MPB"], 
                    yticklabels=["No MPB", "MPB"], fmt="d", cbar=False, ax=ax, cmap = "icefire")
        plt.title(f'{numero_vecinos} vecinos', fontsize = 18)
        plt.xlabel("Predicción KNN", fontsize = 14)
        plt.ylabel("Clasificación manual", fontsize = 14)
        plt.savefig(f'temp.png')
        plt.show()
        plt.close()

    y_true, y_pred = total_data['y_real'], total_data['y_pred']
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    return dataframes, recall_weighted
    
if __name__=='__main__': 
    vecinos = 26
    data, recall_weighted = CV_scores(vecinos, cm = True)
    print(f'Recall weighted score {vecinos} vecinos v2: {round(recall_weighted,2)}')