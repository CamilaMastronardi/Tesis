import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from MPBestimativoVignes import dataframeMPBAOjo
import matplotlib.pyplot as plt

dataframe = pd.read_csv(f'/app/data.txt', header=0, sep='\s*,\s*', dtype={x : 'str' for x in ['YYYY', 'MM', 'DD'] })
def posicionesCercanasEnTiempo(df, time):
    fila_cercana = df[((df.time - time).abs()) <= ((df.time - time).abs().min())]
    try:
        return fila_cercana['posX'].iloc[0], fila_cercana['posY'].iloc[0], fila_cercana['posZ'].iloc[0]
    except IndexError: #cuando fila_cercana analiza una fila vacia no se puede acceder a loc [0] y ocurre un indexerror
        return np.nan, np.nan, np.nan

r_MPB = []
r_err = []
for label, content in dataframe.iterrows():
    year, month, day, orbita, time, delta_time = content['YYYY'], content['MM'], content['DD'], content['orbita'], content['MPB_time'], content['dt']
    datos_para_analisis = dataframeMPBAOjo(year, month, day, int(orbita))
    x_MPB, y_MPB, z_MPB = posicionesCercanasEnTiempo(datos_para_analisis, time)
    x_min, y_min, z_min = posicionesCercanasEnTiempo(datos_para_analisis, time - delta_time)
    x_max, y_max, z_max = posicionesCercanasEnTiempo(datos_para_analisis, time + delta_time)
    x_err, y_err, z_err = x_max - x_min, y_max - y_min, z_max - z_min
    
    r_MPB.append([x_MPB, y_MPB, z_MPB])
    r_err.append([x_err, y_err, z_err])

def polares(L, X0):
    x, y, z = L[0], L[1], L[2] 
    r = np.sqrt((x-X0)**2+z**2)
    theta = np.arctan((x-X0)/z)
    return r, theta

radio_marte_prom = 3389.5
X0 = -0.78*radio_marte_prom #esto lo saco del paper de Vignes
epsilon = 0.9

posicion = np.array([polares(i, X0) for i in r_MPB])

r = posicion[:,0] #desde el foco
theta = posicion[:,1]

def rVignes(theta, L): #Importa los datos acomodados de campo magnetico y hace el ajuste
    return L*radio_marte_prom/(1+epsilon*np.cos(theta))

popt, pcov = curve_fit(rVignes, theta, r, nan_policy='omit') #hago un ajuste por cuadrados minimos
L = popt 
print(L)

theta_fit = np.linspace(-np.pi/2,np.pi/2, 1000)
r_fit = rVignes(theta_fit, L)

plt.plot(r, theta, 'o')
plt.plot(r_fit, theta_fit)
plt.xlabel('r(m)')
plt.ylabel('theta')
plt.savefig('pruebavignes.png')

'''
def intervaloVignes(r_MPB_estimativo):
    popt, pcov = curve_fit(r_MPB_estimativo['r'],r_MPB_estimativo['theta'], r_fiteoVignes) #hago un ajuste por cuadrados minimos
    L = popt 
    L_err = np.sqrt(np.linalg(pcov))
    deltaL = 0
    intervaloL = np.array([L-deltaL, L+deltaL])
    return(intervaloL)

def filtradoVignes(intervaloL, YYYY, MM, DD): 
    def r_min(theta): 
        epsilon = 1
        L = intervaloL[0]
        return L/(1-epsilon*np.cos(theta))
    def r_max(theta): 
        epsilon = 1
        L = intervaloL[1]
        return L/(1-epsilon*np.cos(theta))
    indice
    for i in range(len(r_MPB_estimativo)): 
        


if __name__== '__main__' :

  if len(sys.argv) !=2: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 DescargaDatosB.py YYYY-MM-DD")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
  # Llama a la función para descargar datos de campo magnetico
    descargarDatosCampo(YYYY,MM,DD)
    print('hecho')
'''