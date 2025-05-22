import sys
import os
import numpy as np
import pandas as pd
from PromedioPorVentana import filtrarVentana
import matplotlib.pyplot as plt

#defino datos que necesito para sacar la posicion angular
radio_marte_prom = 3389.5
X0 = 0.78*radio_marte_prom #esto lo saco del paper de Vignes
epsilon = 0.9 #valor en vignes
L = 0.96

#Necesito definir estas funciones de nuevo porque para importarlas se ejecuta todo el codigo de FiteoVignes desde cero. 
def rVignes(theta: np.array, L: float) -> float: 
    return X0 + L*radio_marte_prom/(1+epsilon*np.cos(theta))

def cilindricas(L: list):
    x, y, z = L[0], L[1], L[2] 
    rho = np.sqrt(y**2+z**2)
    r = np.sqrt(rho**2+x**2)
    theta = np.arccos(x/r) 
    return rho, theta

#defino funcion que se queda solo con datos entre el deltaL que defini
def _filtradoVignes(YYYY: str, MM: str, DD: str, orbita: int, band_size_min : int, band_size_max : int) -> pd.DataFrame:
    PathDataVignes = '/app/AnalisisVignes/LVignes'
    archivo = os.path.join(PathDataVignes, "L_vignes")

    data = filtrarVentana(YYYY, MM, DD, orbita).reset_index() #le hago un reset index porque por el tratamiento a los datos los indices estaban mal
    posX, posY, posZ = data['posX'], data['posY'], data['posZ']
    rho, theta = cilindricas([posX,posY,posZ])
    deltaL_min = L*band_size_min/100
    deltaL_max = L*band_size_max/100

    for (angulo,i) in zip(theta,range(len(theta))): 
        rho_max = rVignes(angulo,L+deltaL_max)
        rho_min = rVignes(angulo,L-deltaL_min)

        if rho.iloc[i] > rho_max or rho.iloc[i] < rho_min: #si se va de los limites
            data = data.drop(i)
        else:
            continue

    return data

def filtradoVignes(YYYY: str, MM: str, DD: str, orbita: int, use_cache: bool = True, band_size_min: int = 50, band_size_max: int = 50):
    
    CorteVignes_cache = os.path.join(os.path.dirname(__file__),'cache/CorteVignes') 
    if band_size_min == band_size_max:
        band_size_string = f'{band_size_max}' 
    else:
        band_size_string = f'{band_size_max}_{band_size_min}'
    path = os.path.join(CorteVignes_cache, f'{YYYY}_{MM}_{DD}-{orbita}-bs={band_size_string}.csv')
    if use_cache and os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else: 
        os.makedirs(CorteVignes_cache, exist_ok=True)
        df = _filtradoVignes(YYYY, MM, DD, orbita, band_size_min, band_size_max)
        df.to_csv(path)
        return df
    
if __name__== '__main__' :

  if len(sys.argv) !=3: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 CorteVignes.py YYYY-MM-DD n°orbira")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 3:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
    orbita = int(sys.argv[2])
  # Llama a la función para filtrar
    L = 0.96
    deltaL=50*L/100
    theta = np.linspace(0, np.pi/2,100)
    df_1 = filtradoVignes(YYYY, MM, DD, int(orbita), band_size_min=50, band_size_max=150)
    df_2 = filtradoVignes(YYYY, MM, DD, int(orbita), band_size_min=50, band_size_max=50)

    fig = plt.figure(figsize=(15, 10))
    time = df_1['time']
    B = df_1['mod_B']
    time2 = df_2['time']
    B2 = df_2['mod_B']
    plt.plot(time, B)
    plt.ylabel(r'|$B_{3C}$| (nT)', fontsize = 32)
    plt.xlabel('Time (hs)', fontsize = 32)
    plt.xticks(fontsize = 28)
    plt.yticks(fontsize = 28)
    plt.savefig('temp.png')