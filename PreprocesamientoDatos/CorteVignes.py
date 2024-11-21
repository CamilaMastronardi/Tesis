import sys
import os
import numpy as np
import pandas as pd
from PromedioPorVentana import filtrarVentana
import matplotlib.pyplot as plt

#defino datos que necesito para sacar la posicion angular
radio_marte_prom = 3389.5
X0 = -0.78*radio_marte_prom #esto lo saco del paper de Vignes
epsilon = 0.9 #valor en vignes

#Necesito definir estas funciones de nuevo porque para importarlas se ejecuta todo el codigo de FiteoVignes desde cero. 
def rVignes(theta, L): 
    return L*radio_marte_prom/(1+epsilon*np.cos(theta))

def cilindricas(L, X0):
    x, y, z = L[0], L[1], L[2] 
    s = x + X0
    rho = np.sqrt(y**2+z**2)
    r = np.sqrt(rho**2+s**2)
    theta = np.arccos(s/r) 
    return theta

#defino funcion que se queda solo con datos entre el deltaL que defini
def filtradoVignes(YYYY, MM, DD, orbita):
    PathDataVignes = '/app/AnalisisVignes/LVignes'
    archivo = os.path.join(PathDataVignes, f"L_vignes")
    
    with open(archivo, 'rb') as archivo:
        L, deltaL = np.load(archivo)

    data = filtrarVentana(YYYY, MM, DD, orbita).reset_index() #le hago un reset index porque por el tratamiento a los datos los indices estaban mal
    posX, posY, posZ = data['posX'], data['posY'], data['posZ']
    theta = cilindricas([posX,posY,posZ], X0)

    for (angulo,i) in zip(theta,range(len(theta))): 
        r_max = rVignes(angulo,L+10*deltaL)
        r_min = rVignes(angulo,L-10*deltaL)
        x_max = r_max*np.cos(angulo)
        x_min = r_min*np.cos(angulo)

        if posX.iloc[i] > x_max or posX.iloc[i] < x_min: 
            data = data.drop(i)
            print(f'se elimino columna {i}')
        else: 
            continue

    return data

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