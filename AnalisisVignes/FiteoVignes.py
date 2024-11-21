import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from PreprocesamientoDatos.PromedioPorVentana import filtrarVentana
import matplotlib.pyplot as plt

plt.style.use("./matplotlibStyles.txt")

dataframe = pd.read_csv(f'/app/AnalisisVignes/data.txt', header=0, sep='\s*,\s*', dtype={x : 'str' for x in ['YYYY', 'MM', 'DD'] })
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
    datos_para_analisis = filtrarVentana(year, month, day, int(orbita))
    x_MPB, y_MPB, z_MPB = posicionesCercanasEnTiempo(datos_para_analisis, time)
    x_min, y_min, z_min = posicionesCercanasEnTiempo(datos_para_analisis, time - delta_time)
    x_max, y_max, z_max = posicionesCercanasEnTiempo(datos_para_analisis, time + delta_time)
    x_err, y_err, z_err = x_max - x_min, y_max - y_min, z_max - z_min
    
    r_MPB.append([x_MPB, y_MPB, z_MPB])
    r_err.append([x_err, y_err, z_err])
'''
Defino las coordenadas en cilindricas al rededor del eje x, ya que asi
puedo utilizar (debido a la simetria de revolucion) solo las coordenadas
rho y x. En otras palabras, puedo utilizar solo el radio de cilindricas rho
y el angulo theta definido entre x y z. 
'''
def cilindricas(L, X0):
    x, y, z = L[0], L[1], L[2] 
    s = x + X0
    rho = np.sqrt(y**2+z**2)
    r = np.sqrt(rho**2+s**2)
    #phi = np.arctan(y/z)
    theta = np.arccos(s/r) 
    return rho ,s, r, theta

radio_marte_prom = 3389.5
X0 = -0.78*radio_marte_prom #esto lo saco del paper de Vignes
epsilon = 0.9 #valor en vignes

posicion = np.array([cilindricas(i, X0, radio_marte_prom) for i in r_MPB])
posicion_err = np.array([cilindricas(i, X0, radio_marte_prom) for i in r_err])

rho = posicion[:,0]
s = posicion[:,1]
r = posicion[:,2]
theta = posicion[:,3]
rho_err = posicion_err[:,0]

def rVignes(theta, L): #Importa los datos acomodados de campo magnetico y hace el ajuste
    return L*radio_marte_prom/(1+epsilon*np.cos(theta))

popt, pcov = curve_fit(rVignes, theta, rho, nan_policy='omit') #hago un ajuste por cuadrados minimos
L = popt[0]
L_err = np.sqrt(np.diag(pcov))[0]
print(f'{L}+-{L_err}')

theta_fit = np.linspace(0,max(theta), 50000)
r_fit = rVignes(theta_fit, L)
x_fit = r_fit*np.cos(theta_fit)
z_fit = r_fit*np.sin(theta_fit)

#Defino parametros para limites de datos
deltaL = 10*L_err

r_min = rVignes(theta_fit, L-deltaL)
r_max = rVignes(theta_fit, L+deltaL)
x_max = r_max*np.cos(theta_fit)
x_min = r_min*np.cos(theta_fit)
z_max = r_max*np.sin(theta_fit)
z_min = r_min*np.sin(theta_fit)

PathDataVignes = '/app/AnalisisVignes/LVignes'

if not os.path.exists(PathDataVignes):
    os.makedirs(PathDataVignes)
    
archivoDestino = os.path.join(PathDataVignes, f"L_vignes")

with open(archivoDestino, "wb") as archivo:
    np.save(archivo, np.array([L, deltaL]))

'''
Para plotear
plt.plot(s, rho, 'o')
plt.plot(x_fit,z_fit)
plt.plot(x_max,z_max,'--', color='darkblue')
plt.plot(x_min,z_min,'--', color='darkblue')
plt.xlabel('$\~{x}$') 
plt.ylabel('$\~{z}$')
plt.xlim(min(x_min),max(x_max))
plt.ylim(min(z_min),max(z_max))
plt.grid(True)

plt.savefig('pruebavignes.png')
'''