import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from MPBestimativoVignes import dataframeMPBAOjo

dataframe = pd.read_csv(f'/app/data.txt', header=0, sep='\s*,\s*', dtype={x : 'str' for x in ['YYYY', 'MM', 'DD'] })
def posicionesCercanasEnTiempo(df, time):
    fila_cercana = df[((df.time - time).abs()) <= ((df.time - time).abs().min())]
    return fila_cercana['posX'], fila_cercana['posY'], fila_cercana['posZ']

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

print(r_MPB)

def rVignes(theta, L): #Importa los datos acomodados de campo magnetico y hace el ajuste
    epsilon = 1
    X0 = 1
    return X0 + L/(1-epsilon*np.cos(theta))

#popt, pcov = curve_fit(r_MPB_estimativo['r'],r_MPB_estimativo['theta'], r_fiteoVignes) #hago un ajuste por cuadrados minimos
#L = popt 

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