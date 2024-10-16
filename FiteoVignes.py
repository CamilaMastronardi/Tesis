import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from MPBestimativoVignes import dataframeMPBAOjo
import matplotlib.pyplot as plt

#defino valores que voy a usar a lo largo del codigo
radio_marte_prom = 3389.5
X0 = -0.78*radio_marte_prom #esto lo saco del paper de Vignes
epsilon = 0.9 #esto lo saco del paper de Vignes

#leo el archivo con los datos de tiempos cruces de MPB sacados a ojo
dataframe = pd.read_csv(f'/app/data.txt', header=0, sep='\s*,\s*', dtype={x : 'str' for x in ['YYYY', 'MM', 'DD'] })

#como el tiempo no tiene porque coincidir con un valor de tiempo en el dataframe creo la siguiente funcion que determina el X a partir del t estimado
def posicionesCercanasEnTiempo(df, time):
    fila_cercana = df[((df.time - time).abs()) <= ((df.time - time).abs().min())]
    try:
        return fila_cercana['posX'].iloc[0], fila_cercana['posY'].iloc[0], fila_cercana['posZ'].iloc[0]
    except IndexError: #cuando fila_cercana analiza una fila vacia no se puede acceder a loc [0] y ocurre un indexerror
        return np.nan, np.nan, np.nan
    
#defino listas de posiciones en el MPB y uso la funcion anterior para obtener las posiciones del MPB
x_MPB = np.array([])
x_MPB_err = np.array([])
y_MPB = np.array([])
y_MPB_err = np.array([])
z_MPB = np.array([])
z_MPB_err = np.array([])

for label, content in dataframe.iterrows():
    year, month, day, orbita, time, delta_time = content['YYYY'], content['MM'], content['DD'], content['orbita'], content['MPB_time'], content['dt']
    datos_para_analisis = dataframeMPBAOjo(year, month, day, int(orbita))
    xmpb, ympb, zmpb = posicionesCercanasEnTiempo(datos_para_analisis, time)
    x_min, y_min, z_min = posicionesCercanasEnTiempo(datos_para_analisis, time - delta_time)
    x_max, y_max, z_max = posicionesCercanasEnTiempo(datos_para_analisis, time + delta_time)
    x_err, y_err, z_err = x_max - x_min, y_max - y_min, z_max - z_min
    
    x_MPB = np.append(x_MPB, xmpb)
    y_MPB = np.append(y_MPB, ympb)
    z_MPB = np.append(z_MPB, zmpb)

    x_MPB_err = np.append(x_MPB_err, x_err)
    y_MPB_err = np.append(y_MPB_err, y_err)
    z_MPB_err = np.append(z_MPB_err, z_err)

#Defino funciones para el ajuste de Vignes

def polares(x,y,z):
    r = np.sqrt((x-X0)**2+z**2)
    theta = np.arctan(z/(x-X0))
    return r, theta

posicion = np.array([polares(x_MPB[i],y_MPB[i],z_MPB[i]) for i in range(len(x_MPB))])
r = posicion[:,0] #desde el foco
theta = posicion[:,1]
print(r)

def rVignes(theta, L):
    return L*radio_marte_prom/(1+epsilon*np.cos(theta))

#Hago el ajuste y grafico
popt, pcov = curve_fit(rVignes, theta, r, nan_policy='omit', p0 = 2)
L = popt 
print(L)

theta_fit = np.linspace(0,np.pi/2, 1000)
r_fit = rVignes(theta_fit, L)
x_fit = r_fit*np.cos(theta_fit)
z_fit = r_fit*np.sin(theta_fit)

fig, ax = plt.subplots()
ax.plot(x_fit, z_fit, '-')
ax.scatter(x_MPB, np.sqrt(y_MPB**2+z_MPB**2), s=20, color = 'darkblue')
ax.set_xlabel('x(m)')
ax.set_ylabel('ryz')
plt.savefig('pruebavignes.png')

fig, ax = plt.subplots()
ax.plot(theta_fit, r_fit, '-')
ax.scatter(polares(x_MPB, y_MPB, z_MPB)[0],polares(x_MPB, y_MPB, z_MPB)[1], s=20, color = 'darkblue')
plt.savefig('pruebavignestheta.png')

'''
def intervaloVignes(cartesianas_MPB_estimativo):
    popt, pcov = curve_fit(cartesianas_MPB_estimativo['r'],cartesianas_MPB_estimativo['theta'], r_fiteoVignes) #hago un ajuste por cuadrados minimos
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
    for i in range(len(cartesianas_MPB_estimativo)): 
        


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