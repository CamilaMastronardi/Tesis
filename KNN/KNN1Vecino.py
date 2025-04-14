# Librerias basicas para manejo de datos
import numpy as np
import pandas as pd
import os
import sys

# Librerias para graficar
import matplotlib.pyplot as plt
import seaborn as sn

#Traigo KNN y DTW
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from KNN import KNN_timeSeries, DTW, completeData

print('Descargando y preparando datos de campo magnetico a clasificaar')
start_time = time.time()

path = ''

data_to_be_classified = completeData()
end_time = time.time()
execution_time = end_time - start_time
print(f'Data a ser clasificada descargada y preparada en {execution_time}')

