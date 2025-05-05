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
from sklearn.metrics import confusion_matrix

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

def CV_scores(numero_vecinos: int, cm: bool = True, 
    score_method: callable = sklearn.metrics.recall_score, 
    cv_iter: list = [0,1,2,3,4]) -> Union[pd.DataFrame , float]:
    
    dataframes = {} 
    total_data = pd.DataFrame()
    for i in cv_iter:
        path = f'KNN/Cross_Validation/Resultados/CV_{numero_vecinos}vecinos_1000DTW_{i}_weighted_v2.csv'
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
        plt.savefig(f'KNN/Cross_Validation/Figuras_Cross_Val/{numero_vecinos}vecinos_total_weighted_v2.png')
        plt.show()
        plt.close()

    score = score_method(total_data['y_real'], total_data['y_pred'])
    return dataframes, score
    
if __name__=='__main__': 
    data, score = CV_scores(1, cm = True, cv_iter=[0,1,2,3])
    print(f'Recall score 3 vecinos v2: {round(score,2)}')