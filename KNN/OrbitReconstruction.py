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
def testData() -> pd.DataFrame:
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
    return reordered_dates.reset_index(), reordered_orbits.reset_index()

dates, orbits = testData()
todo_junto = pd.concat([total_data_1vecino, dates, orbits], axis=1)

sep_date = [grupo for _, grupo in todo_junto.groupby('Fecha')]

for j in range(len(sep_date)): 
    print(len(sep_date[j]))
    data = sep_date[j].reset_index()
    date = data['Fecha'].iloc[0]
    YYYY, MM, DD = date.split('-')
    orbits = data['orbita']

    df = acomodarDatos(YYYY, MM, DD)
    time = df['time']
    B = df['mod_B']
    r = df['r_sat']
        #event_data = _filtradoVignes(YYYY, MM, DD, n)

