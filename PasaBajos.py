# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:36:07 2024

@author: cami9
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import fftpack
from AcomodarDatosB import acomodarDatos
from SepararHemisferios import separarHemisferios
import sys
import os
import pandas as pd

plt.style.use("./matplotlibStyles.txt")

#hago el filtro para la orbita total y despues separo
def pasaBajos(YYYY, MM, DD): # Diseño del filtro pasa bajos
    data = separarHemisferios(YYYY, MM, DD) #es un pd.dataframe
    n_orbitas = max(data['orbita'])

    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs  # Frecuencia de Nyquist
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(data, cutoff, fs, order):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    for orbita in range(1, n_orbitas+1):
        data_orbita = data[data['orbita']==orbita]
        Bx = data_orbita.Bx
        By = data_orbita.By
        Bz = data_orbita.Bz
        B_vector = np.array([Bx,By,Bz]).transpose()
        B = np.zeros(len(B_vector[:,0]))
        for i in range(len(B_vector[:,0])):
            B_f = np.linalg.norm(B_vector[i])
            B[i] = B_f
        time = data_orbita.time
    # Parámetros del filtro
        cutoff = 200  # Frecuencia de corte del filtro pasa bajos (Hz)
        order = 5    # Orden del filtro
        fs = 3600

    # Aplicar el filtro a la señal
        filtered_Bx = lowpass_filter(Bx, cutoff, fs, order)
        filtered_By = lowpass_filter(By, cutoff, fs, order)
        filtered_Bz = lowpass_filter(Bz, cutoff, fs, order)

        filtered_B_vector = np.array([filtered_Bx,filtered_By,filtered_Bz]).transpose()
        filtered_B = np.zeros(len(filtered_B_vector[:,0]))
        for i in range(len(filtered_B_vector[:,0])):
            B_f = np.linalg.norm(filtered_B_vector[i])
            filtered_B[i] = B_f

        Path = f'/app/datos_campo_magnetico_pasabajos'
        if not os.path.exists(Path):
            os.makedirs(Path)

        # Transformo la señal filtrada y sin filtrar para comparar
        sig_filtrada = list(filtered_B)
        sig_fft_filtrada = np.fft.fft(sig_filtrada)
        sig_fft = np.fft.fft(B)
        power_filtrado = np.abs(sig_fft_filtrada)**2
        power = np.abs(sig_fft)**2
    # Calculo el espacio de frecuencias
        time_step = 1/fs
        sample_freq = fftpack.fftfreq(len(B), d=time_step)
        
        archivoDestino = os.path.join(Path, f"pasabajos_{DD}-{MM}-{YYYY}_orbita{orbita}.csv")
        data_filtered = pd.DataFrame({'time': time, 'mod_B': filtered_B, 'Bx': filtered_Bx, 'By': filtered_By, 'Bz': filtered_Bz, 'fft_filtered': power_filtrado, 'fft': power, 'freq':sample_freq})
        data_filtered.to_csv(archivoDestino)

    return(n_orbitas)



def graficadora(DD, MM, YYYY, n_orbitas):
  Path = f'/app/datos_campo_magnetico_pasabajos'
  data = separarHemisferios(YYYY, MM, DD) #es un pd.dataframe
  for orbita in range(1, n_orbitas+1):
    archivoDestino = os.path.join(Path, f"pasabajos_{DD}-{MM}-{YYYY}_orbita{orbita}.csv")
    data_cruda = data[data['orbita']==orbita]
    B_cruda = data_cruda.mod_B
    r_sat = data_cruda.r_sat
    df = pd.read_csv(archivoDestino)
    t, B, Bx, By, Bz, fft, fft_filtered, freq = df.time, df.mod_B, df.Bx, df.By, df.Bz, df.fft, df.fft_filtered, df.freq
    
    PathFig = '/app/datos_campo_magnetico_pasabajos/Ploteos'
    if not os.path.exists(PathFig):
      os.makedirs(PathFig)

    fig, ax1 = plt.subplots(1, 1, figsize=(20, 14))
    ax1.set_title(f'{DD}-{MM}-{YYYY}')
    ax1.plot(freq, fft_filtered,'-' ,label='filtrada', zorder=1)
    ax1.plot(freq, fft,'-', label='cruda', zorder=0, alpha=0.5)
    ax1.set_ylim(0, 1000000)
    ax1.set_xlabel('frecuencia [1/hs]')
    ax1.set_ylabel('power')
    plt.legend()
    ax1.grid()
    plt.savefig(f'/app/datos_campo_magnetico_pasabajos/Ploteos/{YYYY}_{MM}_{DD}_transformada_{orbita}')
    
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 14))
    ax1.set_title(f'{DD}-{MM}-{YYYY}')
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()
    ax1.plot(t, B, lw=2, zorder=1)
    ax1.plot(t, B_cruda, alpha=0.5, zorder=0)
    ax2.plot(t, r_sat, '-', alpha = 0.7)
    ax1.set_xlabel('tiempo [hs]')
    ax1.set_ylabel('|B| [nT]')
    ax2.set_ylabel('altura [Km]')
    ax1.grid()
    ax3.plot(t, Bx, label='B en X')
    ax3.plot(t, By, label='B en Y')
    ax3.plot(t, Bz, label='B en Z')
    ax3.legend()
    ax4.plot(t, r_sat, '-', alpha = 0.7)
    ax3.set_xlabel('tiempo [hs]')
    ax3.set_ylabel('Campo Magnetico [nT]')
    ax4.set_ylabel('altura [Km]')
    ax3.grid()

    plt.savefig(f'/app/datos_campo_magnetico_pasabajos/Ploteos/{YYYY}_{MM}_{DD}_pasabajos_{orbita}')


if __name__== '__main__' :

  if len(sys.argv) !=2: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 PasaBajos.py YYYY-MM-DD")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
    n_orbita = pasaBajos(YYYY,MM,DD)
    graficadora(DD, MM, YYYY, n_orbita)
  # Llama a la función para descargar datos de campo magnetico
    print('hecho')
