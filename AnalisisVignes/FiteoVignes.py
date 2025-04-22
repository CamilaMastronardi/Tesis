import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, 'PreprocesamientoDatos'))
from AcomodarDatosB import acomodarDatos


#defino datos que necesito para sacar la posicion angular
radio_marte_prom = 3389.5
X0 = 0.78*radio_marte_prom #esto lo saco del paper de Vignes
epsilon = 0.9 #valor en vignes
L = 0.96

L_BS = 2.04
epsilon_BS = 1.03

#Necesito definir estas funciones de nuevo porque para importarlas se ejecuta todo el codigo de FiteoVignes desde cero. 
def rVignes(theta: np.array, L: float) -> float: 
    return X0 + L*radio_marte_prom/(1+epsilon*np.cos(theta))

def BSVignes(theta: np.array) -> float: 
    return X0 + L_BS*radio_marte_prom/(1+epsilon_BS*np.cos(theta))

def cilindricas(L: list):
    x, y, z = np.array(L[0]), np.array(L[1]), np.array(L[2])
    rho = np.sqrt(y**2+z**2)
    r = np.sqrt(rho**2+x**2)
    theta = np.arccos(x/r) 
    return rho, theta

def posicionesCercanasEnTiempo(df, time):
    fila_cercana = df[((df.time - time).abs()) <= ((df.time - time).abs().min())]
    try:
        return fila_cercana['posX'].iloc[0], fila_cercana['posY'].iloc[0], fila_cercana['posZ'].iloc[0]
    except IndexError: #cuando fila_cercana analiza una fila vacia no se puede acceder a loc [0] y ocurre un indexerror
        return np.nan, np.nan, np.nan
    
dataframe = pd.read_csv(f'/app/AnalisisVignes/data_test.txt', header=0, sep='\s*,\s*', dtype={x : 'str' for x in ['YYYY', 'MM', 'DD'] })
def pos():
    X = []
    Y = []
    Z = []
    i = 1
    for label, content in dataframe.iterrows():
        print(f'{i} de {len(dataframe)}')
        i = i + 1
        try:
            year, month, day, time = content['YYYY'], content['MM'], content['DD'], content['MPB_time']
            datos_para_analisis = acomodarDatos(year, month, day)
            x_MPB, y_MPB, z_MPB = posicionesCercanasEnTiempo(datos_para_analisis, time)  
            if z_MPB>0:
                X.append(x_MPB)
                Y.append(y_MPB)
                Z.append(z_MPB)
        except:
            continue

    pos_MPB = [X, Y, Z]

    return pos_MPB

pos_MPB = pos()

def graficoDeltaL(bandsize: int):
    rho_MPB, theta_MPB = cilindricas(pos_MPB)
    x_MPB = pos_MPB[0]

    theta = np.linspace(-np.pi, np.pi, 100)
    deltaL = L*bandsize/100
    r = rVignes(theta, L)
    r_max = rVignes(theta,L+deltaL)
    r_min = rVignes(theta,L-deltaL)
    x,y = r*(np.cos(theta), np.sin(theta))
    x_max, y_max = r_max*(np.cos(theta), np.sin(theta))
    x_min, y_min = r_min*(np.cos(theta), np.sin(theta))

    r_BS = BSVignes(theta)
    x_BS, y_BS = r_BS*(np.cos(theta), np.sin(theta))

    largo1 = 20
    largo2 = 12
    fig, ax = plt.subplots(figsize=(largo1, largo2))
    ax.plot(radio_marte_prom*np.cos(theta),radio_marte_prom*np.sin(theta), color = 'black')
    ax.scatter(x_MPB,rho_MPB, zorder = 100)
    wedge = Wedge((0, 0), radio_marte_prom, -90, 90, facecolor='black', edgecolor='none')
    ax.add_patch(wedge)
    ax.plot(x,y, color='red')
    ax.plot(x_BS,y_BS, color='grey')
    ax.plot(x_min,y_min, color='pink')
    ax.plot(x_max,y_max, color = 'pink')
    ax.set_xlim(-3*radio_marte_prom,2*radio_marte_prom)
    ax.set_ylim(-0.5*radio_marte_prom*largo2/largo1,(4.5*radio_marte_prom)*largo2/largo1)
    plt.grid()
    plt.savefig('temp.png')

graficoDeltaL(50)

def test_cilindricas():
    a = 3  # semieje mayor en x
    b = 1  # semiejes menores en y y z

    # Número de puntos
    N = 50

    # Generamos puntos sobre la superficie de una esfera (parametrización uniforme en la esfera)
    theta = np.random.uniform(0, np.pi / 2, N)        # colatitud: 0 a pi/2 → z > 0
    phi = np.random.uniform(0, np.pi / 2, N)          # longitud: 0 a pi/2 → x > 0

    # Coordenadas sobre la esfera unitaria
    x_s = np.sin(theta) * np.cos(phi)
    y_s = np.sin(theta) * np.sin(phi)
    z_s = np.cos(theta)

    # Escalamos al elipsoide
    x = a * x_s
    y = b * y_s
    z = b * z_s
    L = [x, y, z]
    rho, theta = cilindricas(L)

    fig = plt.figure(figsize=(18,6))
    plt.scatter(x,rho)
    plt.savefig('tempcil2.png')

test_cilindricas()