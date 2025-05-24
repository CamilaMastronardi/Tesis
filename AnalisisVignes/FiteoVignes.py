import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, 'PreprocesamientoDatos'))
plt.style.use("./matplotlibStyles.txt") 
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


def graficoDeltaL(bandsize: int):
    pos_MPB = pos()
    rho_MPB_1, theta_MPB = cilindricas(pos_MPB)
    rho_MPB = rho_MPB_1/radio_marte_prom
    x_MPB = np.array(pos_MPB[0])/radio_marte_prom

    theta = np.linspace(-np.pi, np.pi, 100)
    deltaL = L * bandsize / 100
    r = rVignes(theta, L)/radio_marte_prom
    r_max = rVignes(theta, L + deltaL)/radio_marte_prom
    r_min = rVignes(theta, L - deltaL)/radio_marte_prom
    
    x, y = r * (np.cos(theta), np.sin(theta))
    x_max, y_max = r_max * (np.cos(theta), np.sin(theta))
    x_min, y_min = r_min * (np.cos(theta), np.sin(theta))

    r_BS = BSVignes(theta)/radio_marte_prom
    x_BS, y_BS = r_BS * (np.cos(theta), np.sin(theta))

    largo1 = 20
    largo2 = 12
    fig, ax = plt.subplots(figsize=(largo1, largo2))
    
    ax.plot(np.cos(theta), np.sin(theta), color='black')
    ax.scatter(x_MPB, rho_MPB, zorder=100, color = 'black', s=100, label='Posición MPB entrenamiento')

    # Planeta en negro
    wedge = Wedge((0, 0), 1, 90, -90, facecolor='black', edgecolor='none')
    ax.add_patch(wedge)

    # Relleno entre las curvas rosas (deltaL)
    ax.fill(np.concatenate([x_max, x_min[::-1]]), np.concatenate([y_max, y_min[::-1]]), color='pink',
        alpha=0.4,zorder=0,label='Región de interes')

    ax.plot(x, y, lw=2 ,color='mediumorchid', label='Vignes MPB')
    ax.plot(x_BS, y_BS, lw=2, color='grey', label='Vignes BS')
    ax.plot(x_min, y_min, color='pink')
    ax.plot(x_max, y_max, color='pink')
    ax.legend(loc=[1.01,0.75])
    ax.set_ylabel(r'$\sqrt{y^2+z^2}$')
    ax.set_xlabel(r'$x_{MSO}$')

    ax.set_xlim(-3, 2)
    ax.set_ylim(-0.5 * largo2 / largo1, (4.5) * largo2 / largo1)
    plt.savefig('temp.png', bbox_inches='tight')

#graficoDeltaL(50)

def graficoDeltaL(bandsize: int):
    pos_MPB = pos()
    rho_MPB_1, theta_MPB = cilindricas(pos_MPB)
    rho_MPB = rho_MPB_1/radio_marte_prom
    x_MPB = np.array(pos_MPB[0])/radio_marte_prom

    theta = np.linspace(-np.pi, np.pi, 100)
    deltaL = L * bandsize / 100
    r = rVignes(theta, L)/radio_marte_prom
    r_max = rVignes(theta, L + deltaL)/radio_marte_prom
    r_min = rVignes(theta, L - deltaL)/radio_marte_prom
    
    x, y = r * (np.cos(theta), np.sin(theta))
    x_max, y_max = r_max * (np.cos(theta), np.sin(theta))
    x_min, y_min = r_min * (np.cos(theta), np.sin(theta))

    r_BS = BSVignes(theta)/radio_marte_prom
    x_BS, y_BS = r_BS * (np.cos(theta), np.sin(theta))

    largo1 = 20
    largo2 = 12
    fig, ax = plt.subplots(figsize=(largo1, largo2))
    
    ax.plot(np.cos(theta), np.sin(theta), color='black')
    ax.scatter(x_MPB, rho_MPB, zorder=100, color = 'black', s=100, label='Posición MPB entrenamiento')

    # Planeta en negro
    wedge = Wedge((0, 0), 1, 90, -90, facecolor='black', edgecolor='none')
    ax.add_patch(wedge)

    # Relleno entre las curvas rosas (deltaL)
    ax.fill(np.concatenate([x_max, x_min[::-1]]), np.concatenate([y_max, y_min[::-1]]), color='pink',
        alpha=0.4,zorder=0,label='Región de interes')

    ax.plot(x, y, lw=2 ,color='mediumorchid', label='Vignes MPB')
    ax.plot(x_BS, y_BS, lw=2, color='grey', label='Vignes BS')
    ax.plot(x_min, y_min, color='pink')
    ax.plot(x_max, y_max, color='pink')
    ax.legend(loc=[1.01,0.75])
    ax.set_ylabel(r'$\sqrt{y^2+z^2}$')
    ax.set_xlabel(r'$x_{MSO}$')

    ax.set_xlim(-3, 2)
    ax.set_ylim(-0.5 * largo2 / largo1, (4.5) * largo2 / largo1)
    plt.savefig('temp.png', bbox_inches='tight')

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

#test_cilindricas()

def graficoDeltaL(bandsize_min: int, bandsize_max: int):

    theta = np.linspace(-np.pi, np.pi, 100)
    deltaL_min = L * bandsize_min / 100
    deltaL_max = L * bandsize_max / 100
    r = rVignes(theta, L)/radio_marte_prom
    r_max = rVignes(theta, L + deltaL_max)/radio_marte_prom
    r_min = rVignes(theta, L - deltaL_min)/radio_marte_prom
    
    x, y = r * (np.cos(theta), np.sin(theta))
    x_max, y_max = r_max * (np.cos(theta), np.sin(theta))
    x_min, y_min = r_min * (np.cos(theta), np.sin(theta))

    r_BS = BSVignes(theta)/radio_marte_prom
    x_BS, y_BS = r_BS * (np.cos(theta), np.sin(theta))

    largo1 = 20
    largo2 = 12
    fig, ax = plt.subplots(figsize=(largo1, largo2))
    
    ax.plot(np.cos(theta), np.sin(theta), color='black')
    #ax.scatter(x_MPB, rho_MPB, zorder=100, color = 'black', s=100, label='Posición MPB entrenamiento')

    # Planeta en negro
    wedge = Wedge((0, 0), 1, 90, -90, facecolor='black', edgecolor='none')
    ax.add_patch(wedge)

    # Relleno entre las curvas rosas (deltaL)
    ax.fill(np.concatenate([x_max, x_min[::-1]]), np.concatenate([y_max, y_min[::-1]]), color='pink',
        alpha=0.4,zorder=0,label='Región de interes')

    ax.plot(x, y, lw=2 ,color='mediumorchid', label='Vignes MPB')
    ax.plot(x_BS, y_BS, lw=2, color='grey', label='Vignes BS')
    ax.plot(x_min, y_min, color='pink')
    ax.plot(x_max, y_max, color='pink')
    ax.tick_params(labelsize=30)
    ax.legend(loc=[1.01,0.75])
    ax.set_ylabel(r'$\sqrt{y^2+z^2}$', fontsize = 35)
    ax.set_xlabel(r'$x_{MSO}$', fontsize = 35)

    ax.set_xlim(-3, 2)
    ax.set_ylim(-0.5 * largo2 / largo1, (4.5) * largo2 / largo1)
    plt.savefig('temp.png', bbox_inches='tight')

graficoDeltaL(50, 120)

def graficoOrbita():
    theta = np.linspace(-np.pi, np.pi, 200)
    theta_2 = np.linspace(np.pi/2,2, 100)
    fig, ax = plt.subplots()

    # Círculo que representa Marte
    ax.plot(np.cos(theta), np.sin(theta), color='black')

    # Hemisferio inferior en negro
    wedge = Wedge((0, 0), 1, 90, -90, facecolor='black', edgecolor='none', zorder=100)
    ax.add_patch(wedge)

    # Zona eliminada
    deleted_zone = Wedge((0, 0), 100, 90, -90, facecolor='grey', alpha=0.3, edgecolor='none', zorder=50, label='Región descartada')
    ax.add_patch(deleted_zone)

    # Eje x_MSO (horizontal hacia la derecha, darkblue)
    ax.annotate(
        '', xy=(2, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', linewidth=2, color='mediumorchid'),
        zorder=150
    )
    ax.text(2.1, 0, r'$x_{MSO}$', color='mediumorchid', va='center', ha='left', fontsize=24)

    # Eje z_PC (-25° respecto de la vertical, mediumorchid)
    ang_deg = -25
    ang_rad = np.radians(ang_deg)
    zpc_x = 2.5 * np.sin(ang_rad)
    zpc_y = 2.5 * np.cos(ang_rad)

    ax.annotate(
        '', xy=(zpc_x, zpc_y), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', linewidth=2, color='darkblue'),
        zorder=150
    )
    ax.text(zpc_x * 1.05, zpc_y * 1.05, r'$z_{PC}$', color='darkblue', va='bottom', ha='left', fontsize=24, zorder=1000)

    linea_ang_rad = np.radians(25)
    x1 = - np.cos(linea_ang_rad)
    y1 = - np.sin(linea_ang_rad)
    x2 = np.cos(linea_ang_rad)
    y2 = np.sin(linea_ang_rad)
    ax.plot([x1, x2], [y1, y2], linestyle='--', color='darkblue', linewidth=1.5, zorder=10000)



    # Ángulo en blanco con respecto a la vertical
    ax.plot(1/2*np.cos(theta_2), 1/2*np.sin(theta_2), color='white', zorder=1000)
    ax.text(-0.27, 0.6, r'25°', color='white', zorder=1000, fontsize=16)

    # Limites y estética
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', 'box')
    plt.legend().set_zorder(1000)
    ax.set_xlabel(r'$x_{MSO} (RM)$')
    ax.set_ylabel(r'$\sqrt{y^2+z^2} (RM)$')
    ax.grid(color='black')

    plt.savefig('temp.png', bbox_inches='tight')
    plt.show()

#graficoOrbita()

def graficoHemisferio():
    theta = np.linspace(-np.pi, np.pi, 200)
    theta_2 = np.linspace(np.pi/2,2, 100)
    fig, ax = plt.subplots()

    # Círculo que representa Marte
    ax.plot(np.cos(theta), np.sin(theta), color='black')

    # Hemisferio inferior en negro
    wedge = Wedge((0, 0), 1, 90, -90, facecolor='black', edgecolor='none', zorder=100)
    ax.add_patch(wedge)

    # Zona eliminada
    deleted_zone = Wedge((0, 0), 100, 90, 25, facecolor='grey', alpha=0.3, edgecolor='none', zorder=50, label='Región descartada')
    ax.add_patch(deleted_zone)

    # Eje x_MSO (horizontal hacia la derecha, darkblue)
    ax.annotate(
        '', xy=(2, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', linewidth=2, color='mediumorchid'),
        zorder=150
    )
    ax.text(2.1, 0, r'$x_{MSO}$', color='mediumorchid', va='center', ha='left', fontsize=24)

    # Eje z_PC (-25° respecto de la vertical, mediumorchid)
    ang_deg = -25
    ang_rad = np.radians(ang_deg)
    zpc_x = 2.5 * np.sin(ang_rad)
    zpc_y = 2.5 * np.cos(ang_rad)

    ax.annotate(
        '', xy=(zpc_x, zpc_y), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', linewidth=2, color='darkblue'),
        zorder=150
    )
    ax.text(zpc_x * 1.05, zpc_y * 1.05, r'$z_{PC}$', color='darkblue', va='bottom', ha='left', fontsize=24, zorder=1000)

    linea_ang_rad = np.radians(25)
    x1 = - np.cos(linea_ang_rad)
    y1 = - np.sin(linea_ang_rad)
    x2 = np.cos(linea_ang_rad)
    y2 = np.sin(linea_ang_rad)
    ax.plot([x1, x2], [y1, y2], linestyle='--', color='darkblue', linewidth=1.5, zorder=10000)



    # Ángulo en blanco con respecto a la vertical
    ax.plot(1/2*np.cos(theta_2), 1/2*np.sin(theta_2), color='white', zorder=1000)
    ax.text(-0.27, 0.6, r'25°', color='white', zorder=1000, fontsize=16)

    # Limites y estética
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', 'box')
    plt.legend().set_zorder(1000)
    ax.set_xlabel(r'$x_{MSO} (RM)$')
    ax.set_ylabel(r'$\sqrt{y^2+z^2} (RM)$')
    ax.grid(color='black')

    plt.savefig('temp.png', bbox_inches='tight')
    plt.show()