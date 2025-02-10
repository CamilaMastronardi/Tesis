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

df = cargarTrainingData(group='Group1')
i = 0
for YYYY, MM, DD in zip(df.YYYY, df. MM, df.DD):
    orbitas = n_orbita(YYYY, MM, DD)
    for n in range(1, orbitas+1): 
        filtradoVignes(YYYY, MM, DD, n)
    print(f'fecha {i} de {len(df)}')
    i = i + 1