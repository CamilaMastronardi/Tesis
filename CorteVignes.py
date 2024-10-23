import sys
import os
import numpy as np
import pandas as pd
from PromedioPorVentana import filtrarVentana
import matplotlib.pyplot as plt
from FiteoVignes import rVignesmax, rVignesmin

#defino funcion que se queda solo con datos entre el deltaL que defini
def filtradoVignes(YYYY, MM, DD, orbita):
    PathDataVignes = '/app/LVignes'
    archivo = os.path.join(PathDataVignes, f"L_vignes")
    
    with open(archivo, 'rb') as archivo:
        L, deltaL = np.load(archivo)

    data = filtrarVentana(YYYY, MM, DD, orbita)

    print(L, deltaL)

#    data_vignes = (data[(x_min[i] < data['posX'][i] for i in range(len(x_min)))])
#    return data_vignes

if __name__== '__main__' :

  if len(sys.argv) !=3: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 FiteoVignes.py YYYY-MM-DD n°orbira")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 3:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
    orbita = int(sys.argv[2])
  # Llama a la función para descargar datos de campo magnetico
    filtradoVignes(YYYY,MM,DD,orbita)
    print('hecho')