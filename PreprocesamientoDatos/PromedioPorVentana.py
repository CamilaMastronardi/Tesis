# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:18:51 2024

@author: cami9
"""

import numpy as np
from matplotlib import pyplot as plt
from SepararHemisferios import separarHemisferios
from SepararOrbitas import separarOrbitas
import pandas as pd
import sys
import os
import csv

#plt.style.use("./matplotlibStyles.txt")

def n_orbita(YYYY: str, MM: str, DD: str) -> int:
  cache_dir = os.path.join(os.path.dirname(__file__),'cache/') 
  path = os.path.join(cache_dir, f'numero_de_orbitas.csv')
  if not os.path.exists(path):
    os.makedirs(cache_dir, exist_ok=True)
    data_init = {'YYYY': 'YYYY', 'MM': 'MM', 'DD': 'DD', 'n_orbita': 'n_orbita'}
    df_init = pd.DataFrame(data_init, index=[0])
    df_init.to_csv(path, index=False)

  df = pd.read_csv(path)
  repetidos = df.loc[(df['YYYY'] == YYYY) & (df['MM']==MM) & (df['DD']==DD)]
  if len(repetidos)==0:
    data = separarOrbitas(YYYY, MM, DD)[0]
    n_orbita = max(data['orbita'])
    new_row = pd.DataFrame({'YYYY': YYYY, 'MM':MM, 'DD': DD, 'n_orbita': n_orbita}, index=[0])
    df = pd.concat([df, new_row])
    df.to_csv(path, index=False)
    return int(new_row['n_orbita'].iloc[0])
  else: 
    return int(repetidos['n_orbita'].iloc[0])


def filtrarVentana(YYYY: str,MM: str, DD: str, orbita: int):
  data = separarHemisferios(YYYY, MM, DD) #es un pd.dataframe
  data_orbita = data.loc[(data['orbita'] == orbita)]
  
  Bx_promediado = data_orbita.Bx.rolling(20, center=True).mean().dropna() # Eliminamos filas con NaN
  By_promediado = data_orbita.By.rolling(20, center=True).mean().dropna()
  Bz_promediado = data_orbita.Bz.rolling(20, center=True).mean().dropna()

  B_vector_promediado = np.array([Bx_promediado, By_promediado, Bz_promediado]).transpose()
  B_promediado = np.zeros(len(B_vector_promediado[:,0]))
  for i in range(len(B_vector_promediado[:,0])):
      B_f = np.linalg.norm(B_vector_promediado[i])
      B_promediado[i] = B_f
  
  # Eliminar NaNs
  time_promediado = data_orbita.time.rolling(20, center=True).mean().dropna()
  r_sat_promediado = data_orbita.r_sat.rolling(20, center=True).mean().dropna()
  x_sat = data_orbita.posX.rolling(20, center=True).mean().dropna()
  y_sat = data_orbita.posY.rolling(20, center=True).mean().dropna()
  z_sat = data_orbita.posZ.rolling(20, center=True).mean().dropna()

  Bx = data_orbita.Bx.dropna()
  By = data_orbita.By.dropna()
  Bz = data_orbita.Bz.dropna()
  time = data_orbita.time.dropna()
  B = data_orbita.mod_B.dropna()
  return pd.DataFrame({'time': time_promediado, 'mod_B': B_promediado, 
                      'Bx': Bx_promediado, 'By': By_promediado, 
                      'Bz': Bz_promediado, 'r_sat': r_sat_promediado, 'posX' : x_sat,
                      'posY' : y_sat, 'posZ' : z_sat})


#hago la función que hace el promedio por ventanas a cada coordenada del campo magnetico  
def guardarTodasLasOrbitasFiltradas(YYYY,MM,DD):

  Path = f'/app/DatosCrudos/datos_campo_magnetico_ventana'
  if not os.path.exists(Path):
    os.makedirs(Path)

  n = n_orbita(YYYY, MM, DD)
  for orbita in range(1, n+1):
    #Crea u archivo para meter los datos que salen de la API
    archivoDestino = os.path.join(Path, f"ventana_{DD}-{MM}-{YYYY}_orbita{orbita}.csv")
    # Escribe en lo que sale de filtrar la orbita
    data_filtrada = filtrarVentana(YYYY, MM, DD, orbita)
    data_filtrada.to_csv(archivoDestino)

# Ploteos
def graficadora(YYYY, MM, DD):

  Path = f'/app/DatosCrudos/datos_campo_magnetico_ventana'
  n = n_orbita(YYYY, MM, DD)
  for orbita in range(1, n+1):
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
    
    PathFig = '/app/temp'
    if not os.path.exists(PathFig):
      os.makedirs(PathFig)
    plt.savefig(f'/app/temp/PromedioPorVentana/{YYYY}_{MM}_{DD}_ventana_{orbita}')

if __name__== '__main__' :

  if len(sys.argv) !=3: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 PromedioPorVentana.py YYYY-MM-DD orbita")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 3:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
    orbita = sys.argv[2]
    df = filtrarVentana(YYYY, MM, DD, int(orbita))
    print(df)  