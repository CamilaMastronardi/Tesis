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
def _filtradoVignes(YYYY: str, MM: str, DD: str, orbita: int, band_size : int) -> pd.DataFrame:
    PathDataVignes = '/app/AnalisisVignes/LVignes'
    archivo = os.path.join(PathDataVignes, "L_vignes")

    data = filtrarVentana(YYYY, MM, DD, orbita).reset_index() #le hago un reset index porque por el tratamiento a los datos los indices estaban mal
    posX, posY, posZ = data['posX'], data['posY'], data['posZ']
    rho, theta = cilindricas([posX,posY,posZ])
    deltaL = L*band_size/100

    for (angulo,i) in zip(theta,range(len(theta))): 
        rho_max = rVignes(angulo,L+deltaL)
        rho_min = rVignes(angulo,L-deltaL)

        if rho.iloc[i] > rho_max or rho.iloc[i] < rho_min: #si se va de los limites
            data = data.drop(i)
        else:
            continue

    return data

def filtradoVignes(YYYY: str, MM: str, DD: str, orbita: int, use_cache: bool = True, band_size: int = 50):
    
    CorteVignes_cache = os.path.join(os.path.dirname(__file__),'cache/CorteVignes') 
    path = os.path.join(CorteVignes_cache, f'{YYYY}_{MM}_{DD}-{orbita}-bs={band_size}.csv')
    if use_cache and os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else: 
        os.makedirs(CorteVignes_cache, exist_ok=True)
        df = _filtradoVignes(YYYY, MM, DD, orbita, band_size)
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
    df = _filtradoVignes(YYYY, MM, DD, int(orbita), band_size=50)
    r_max = rVignes(theta,L+deltaL)
    r_min = rVignes(theta,L-deltaL)
    x_max, y_max = r_max*(np.cos(theta), np.sin(theta))
    x_min, y_min = r_min*(np.cos(theta), np.sin(theta))

    x, y, z = df['posX'],df['posY'],df['posZ']
    rho, angulo = cilindricas([x,y,z])
    x_data = rho*np.cos(angulo)
    y_data = rho*np.sin(angulo)
    plt.plot(x_data, y_data)
    plt.plot(x_min,y_min, color='pink')
    plt.plot(x_max,y_max, color = 'pink')
    plt.savefig('temp.png')