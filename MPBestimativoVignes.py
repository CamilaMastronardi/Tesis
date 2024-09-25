import sys
import os
import numpy as np
from AcomodarDatosB import acomodarDatos
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("./matplotlibStyles.txt")

YYYY, MM, DD = '2015', '10', '10'
for orbita in range(1,6):
    df = pd.read_csv(f'/app/datos_campo_magnetico_ventana/ventana_{DD}-{MM}-{YYYY}_orbita{orbita}.csv', lineterminator='\n')
    B, Bx, By, Bz, t = df['mod_B'], df['Bx'], df['By'],df['Bz'], df['time']

    dBx = np.gradient(Bx) #np.gradient usa diferencias finitas centrada
    dBy = np.gradient(By)
    dBz = np.gradient(Bz)

    moduloGradiente = np.linalg.vector_norm(np.array((dBx, dBy, dBz)), axis = 0)
    deltaB = moduloGradiente/B
###################
    if orbita==4:
        print(moduloGradiente[1200:1400])
        print(t[1200],t[1400])
###################

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.set_title(f'{YYYY}-{MM}-{DD} orbita {orbita}')
    ax1.plot(t, deltaB)
    ax1.set_xlabel('time (hs)')
    ax1.set_ylabel('|gradB|/|B|') #
    ax2.plot(t, moduloGradiente, color = 'darkviolet')
    ax2.set_xlabel('time (hs)')
    ax2.set_ylabel('|gradB| (nT)')
    ax3.plot(t, B, color = 'darkblue')
    ax3.set_xlabel('time (hs)')
    ax3.set_ylabel('|B| (nT)')
    plt.savefig(f'prueba_{orbita}.jpg')