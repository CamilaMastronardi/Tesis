# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:18:51 2024

@author: cami9
"""

import numpy as np
from matplotlib import pyplot as plt
from AcomodarDatosB import acomodarDatos
import pandas as pd
import sys
import os
import csv

plt.style.use("./matplotlibStyles.txt")

#hago la función que hace el promedio por ventanas a cada coordenada del campo magnetico 
def filtrarVentana(YYYY,MM,DD):

    data = acomodarDatos(YYYY, MM, DD) #es un pd.dataframe
    Bx_filtrado = data.Bx.rolling(100, center = True).mean()
    By_filtrado = data.By.rolling(100, center = True).mean()
    Bz_filtrado = data.Bz.rolling(100, center = True).mean()
    B_vector_filtrado = np.array([Bx_filtrado,By_filtrado,Bz_filtrado]).transpose()
    B_filtrado = np.zeros(len(B_vector_filtrado[:,0]))
    for i in range(len(B_vector_filtrado[:,0])):
        B_f = np.linalg.norm(B_vector_filtrado[i])
        B_filtrado[i] = B_f
    time_filtrado = data.time.rolling(100, center = True).mean()
   
    Bx = data.Bx
    By = data.By
    Bz = data.Bz
    time = data.time
    B = data.mod_B
    r_sat = data.r_sat
    
    Path = '/app/datos_campo_magnetico_ventana'
    if not os.path.exists(Path):
      os.makedirs(Path)

#Crea u archivo para meter los datos que salen de la API
    archivoDestino = os.path.join(Path, f"ventana_{DD}-{MM}-{YYYY}.csv")
    #por ultimo escribe en el archivo lo que sale de la API
    data_filtrada = pd.DataFrame({'time': time_filtrado, 'mod_B': B_filtrado, 'Bx': Bx_filtrado, 'By':By_filtrado, 'Bz':Bz_filtrado, 'r_sat': r_sat})
    data_filtrada.to_csv(archivoDestino)

    return(B, Bx, By, Bz, time, B_filtrado, Bx_filtrado, By_filtrado, Bz_filtrado, time_filtrado, r_sat)

# Ploteos
def graficadora(B, Bx, By, Bz, t, r_sat, DD, MM, YYYY, caso):
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 14))
    ax1.set_title(f'{DD}-{MM}-{YYYY}')
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()
    ax1.plot(t, B)
    ax2.plot(t, r_sat, '-', alpha = 0.7)
    ax1.set_xlabel('tiempo [hs]')
    ax1.set_ylabel('|B| [nT]')
    ax2.set_ylabel('altura [Km]')
    ax1.grid()
    ax3.plot(t, Bx, label='B en X')
    ax3.plot(t, By, label='B en Y')
    ax3.plot(t, Bz, label='B en Z')
    ax3.legend()
    ax4.plot(t, r_sat, '-', alpha = 0.7)
    ax3.set_xlabel('tiempo [hs]')
    ax3.set_ylabel('Campo Magnetico [nT]')
    ax4.set_ylabel('altura [Km]')
    ax3.grid()
    
    Path = '/app/datos_campo_magnetico_ventana/Ploteos'
    if not os.path.exists(Path):
      os.makedirs(Path)
    plt.savefig(f'/app/datos_campo_magnetico_ventana/Ploteos/{YYYY}_{MM}_{DD}_{caso}')

if __name__== '__main__' :

  if len(sys.argv) !=2: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 PromedioPorVentana.py YYYY-MM-DD")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
    B, Bx, By, Bz, time, B_filtrado, Bx_filtrado, By_filtrado, Bz_filtrado, time_filtrado, r_sat = filtrarVentana(YYYY,MM,DD)
    graficadora(B, Bx, By, Bz, time, r_sat, DD, MM, YYYY, 'crudo')
    graficadora(B_filtrado, Bx_filtrado, By_filtrado, Bz_filtrado, time_filtrado, r_sat, DD, MM, YYYY, 'ventana')
  # Llama a la función para descargar datos de campo magnetico
    print('hecho')