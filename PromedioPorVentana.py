# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:18:51 2024

@author: cami9
"""

import numpy as np
from matplotlib import pyplot as plt
from SepararOrbitas import separar_orbitas
import pandas as pd
import sys
import os
import csv

plt.style.use("./matplotlibStyles.txt")

#hago la función que hace el promedio por ventanas a cada coordenada del campo magnetico 
def filtrarVentana(YYYY,MM,DD):

    data = separar_orbitas(YYYY, MM, DD) #es un pd.dataframe
    n_orbitas = max(data['orbita'])
    for orbita in range(1, n_orbitas+1):
      data_orbita = data[data['orbita']==orbita]
      Bx_filtrado = data_orbita.Bx.rolling(20, center = True).mean() #a un dato por segundo esto es hacer promedio cada 20 segundos
      By_filtrado = data_orbita.By.rolling(20, center = True).mean()
      Bz_filtrado = data_orbita.Bz.rolling(20, center = True).mean()
      B_vector_filtrado = np.array([Bx_filtrado,By_filtrado,Bz_filtrado]).transpose()
      B_filtrado = np.zeros(len(B_vector_filtrado[:,0]))
      for i in range(len(B_vector_filtrado[:,0])):
          B_f = np.linalg.norm(B_vector_filtrado[i])
          B_filtrado[i] = B_f
      time_filtrado = data_orbita.time.rolling(100, center = True).mean()
   
      Bx = data_orbita.Bx
      By = data_orbita.By
      Bz = data_orbita.Bz
      time = data_orbita.time
      B = data_orbita.mod_B
      r_sat = data_orbita.r_sat
      x_sat = data_orbita.posX
    
      Path = f'/app/datos_campo_magnetico_ventana'
      if not os.path.exists(Path):
        os.makedirs(Path)

#Crea u archivo para meter los datos que salen de la API
      archivoDestino = os.path.join(Path, f"ventana_{DD}-{MM}-{YYYY}_orbita{orbita}.csv")
    #por ultimo escribe en el archivo lo que sale de la API
      data_filtrada = pd.DataFrame({'time': time_filtrado, 'mod_B': B_filtrado, 'Bx': Bx_filtrado, 'By':By_filtrado, 'Bz':Bz_filtrado, 'r_sat': r_sat})
      data_filtrada.to_csv(archivoDestino)

    return(n_orbitas)

# Ploteos
def graficadora(DD, MM, YYYY, n_orbitas):
  Path = f'/app/datos_campo_magnetico_ventana'
  for orbita in range(1, n_orbitas+1):
    archivoDestino = os.path.join(Path, f"ventana_{DD}-{MM}-{YYYY}_orbita{orbita}.csv")
    df = pd.read_csv(archivoDestino)
    t, B, Bx, By, Bz, r_sat = df.time, df.mod_B, df.Bx, df.By, df.Bz, df.r_sat

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
    
    PathFig = '/app/datos_campo_magnetico_ventana/Ploteos'
    if not os.path.exists(PathFig):
      os.makedirs(PathFig)
    plt.savefig(f'/app/datos_campo_magnetico_ventana/Ploteos/{YYYY}_{MM}_{DD}_ventana_{orbita}')

if __name__== '__main__' :

  if len(sys.argv) !=2: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 PromedioPorVentana.py YYYY-MM-DD")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
    n_orbita = filtrarVentana(YYYY,MM,DD)
    graficadora(DD, MM, YYYY, n_orbita)
  # Llama a la función para descargar datos de campo magnetico
    print('hecho')