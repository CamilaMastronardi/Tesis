# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:18:51 2024

@author: cami9
"""

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt
from AcomodarDatosB import acomodarDatos
import pandas as pd
import sys

def filtroPasaBajosB(YYYY,MM,DD, B):

    data = acomodarDatos(YYYY, MM, DD)

    time_vec = list(data.time)
    time_step = time_vec[1]-time_vec[0]
    sig = list(getattr(data, B))

# Transformo la señal
    sig_fft = fftpack.fft(sig)

# Como la transformada es compleja uso el poder
    power = np.abs(sig_fft)**2

# Calculo el espacio de frecuencias
    sample_freq = fftpack.fftfreq(len(sig), d=time_step)

# Plot
    plt.figure(figsize=(6, 5))
    plt.plot(sample_freq, power)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')
    plt.savefig(f'espectro_{DD}_{MM}_{YYYY}')

# Frecuencia de amplitud máxima
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]

# Plot
    plt.figure(figsize=(10, 7))
    plt.plot(sample_freq, power)
    plt.xlabel('frecuencia [1/hs]')
    plt.ylabel('poder')
    axes = plt.axes([0.55, 0.3, 0.3, 0.5])
    plt.title('Frecuencia del pico')
    plt.plot(freqs[:30], power[pos_mask][:30])
    plt.setp(axes, yticks=[])
    plt.savefig(f'espectro_{DD}_{MM}_{YYYY}.jpg')

#Filtro frecuencias mayores a 10 veces la del pico
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) > peak_freq*10] = 0
    filtered_sig = fftpack.ifft(high_freq_fft)
    
#Plot del filtro
    plt.figure(figsize=(10, 7))
    plt.plot(time_vec, sig, label='señal original')
    plt.plot(time_vec, filtered_sig, linewidth=3, label='Señal filtrada')
    plt.xlabel('Tiempo [hs]')
    plt.ylabel(B)
    plt.legend(loc='best')
    plt.savefig(f'señalFiltrada_{DD}_{MM}_{YYYY}.jpg')

    
    return(pd.DataFrame({'time':time_vec,
                         'Bfiltrado': filtered_sig}))


if __name__== '__main__' :

  if len(sys.argv) !=3: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python acomodarDatosB.py YYYY-MM-DD B \n con B = mod_B , Bx , By o Bz")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD

  if len(sys.argv) == 2:
    B = sys.argv[2]
    if B not in ['Bx', 'By', 'Bz', 'mod_B']:
        print('Uso python acomodarDatosB.py YYYY-MM-DD B  \n con B = mod_B , Bx , By o Bz')
        sys.exit(1)
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
  # Llama a la función para acomodar datos
    filtroPasaBajosB(YYYY,MM,DD,B)