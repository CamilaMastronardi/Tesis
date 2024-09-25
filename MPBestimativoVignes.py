import sys
import os
import numpy as np
from AcomodarDatosB import acomodarDatos
import pandas as pd
import matplotlib.pyplot as plt
from Filtros import lowpass_filter, highpass_filter 
plt.style.use("./matplotlibStyles.txt")

YYYY, MM, DD = '2015', '01', '04'
for orbita in range(1,6):
    df = pd.read_csv(f'/app/datos_campo_magnetico_ventana/ventana_{DD}-{MM}-{YYYY}_orbita{orbita}.csv', lineterminator='\n')
    B, Bx, By, Bz, t = df['mod_B'], df['Bx'], df['By'],df['Bz'], df['time']

    dBx = np.abs(np.gradient(Bx)) #np.gradient usa diferencias finitas centrada
    dBy = np.abs(np.gradient(By))
    dBz = np.abs(np.gradient(Bz))
    dB = np.abs(np.gradient(B))

    cutoff = 50  # Frecuencia de corte del filtro pasa bajos (Hz)
    order = 3    # Orden del filtro
    fs = 3600
    filtered_grad = lowpass_filter(dB, cutoff, fs, order)

    filtered_B = highpass_filter(B, cutoff, fs, order)

    dB_pd = pd.Series(dB)
    dB_pd_rolling = dB_pd.rolling(10, center=True).mean()
    deltaB = dB_pd.rolling(10, center=True).std().div(dB_pd_rolling)

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.set_title(f'{YYYY}-{MM}-{DD} orbita {orbita}')
    ax1.plot(t, filtered_B)
    ax1.set_xlabel('time (hs)')
    ax1.set_ylabel('') #
    ax2.plot(t, filtered_grad, color = 'darkviolet')
    ax2.set_xlabel('time (hs)')
    ax2.set_ylabel('∇|B| (nT)')
    ax3.plot(t, B, color = 'darkblue')
    ax3.set_xlabel('time (hs)')
    ax3.set_ylabel('|B| (nT)')
    plt.savefig(f'prueba_{orbita}_{cutoff}.jpg')