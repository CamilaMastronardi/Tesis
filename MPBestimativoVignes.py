import sys
import os
import numpy as np
from AcomodarDatosB import acomodarDatos
import pandas as pd
import matplotlib.pyplot as plt
from Filtros import lowpass_filter, highpass_filter 
from PromedioPorVentana import filtrarVentana, n_orbita
plt.style.use("./matplotlibStyles.txt")

def dataframeMPBAOjo(YYYY, MM, DD, orbita):
    df = filtrarVentana(YYYY, MM, DD, orbita)
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

    return t, B, filtered_B, filtered_grad 

def MPBAOjo(YYYY, MM, DD):
    n = n_orbita(YYYY, MM, DD)

    for orbita in range(1,n+1):
        t, B, filtered_B, filtered_grad = dataframeMPBAOjo(YYYY, MM, DD, orbita)

        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        ax1.set_title(f'{YYYY}-{MM}-{DD} orbita {orbita}')
        ax1.plot(t, filtered_B)
        ax1.set_xlabel('time (hs)')
        ax1.set_ylabel('|B|_hfn (nT)') #
        ax2.plot(t, filtered_grad, color = 'darkviolet')
        ax2.set_xlabel('time (hs)')
        ax2.set_ylabel('∇|B|_lf (nT)')
        ax3.plot(t, B, color = 'darkblue')
        ax3.set_xlabel('time (hs)')
        ax3.set_ylabel('|B| (nT)')

        PathFig = '/app/fig_MPB_a_ojo'
        if not os.path.exists(PathFig):
            os.makedirs(PathFig)
        plt.savefig(os.path.join(PathFig, f'{YYYY}-{MM}-{DD}_orbita_{orbita}.jpg'))

if __name__== '__main__' :

  if len(sys.argv) !=2: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 PromedioPorVentana.py YYYY-MM-DD")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
    MPBAOjo(YYYY, MM, DD)
  # Llama a la función para descargar datos de campo magnetico
    print('hecho')  