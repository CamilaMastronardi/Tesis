# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:36:07 2024

@author: cami9
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from AcomodarDatosB import acomodarDatos
import sys

def pasaBajos(YYYY, MM, DD):
    data = acomodarDatos(YYYY, MM, DD) #es un pd.dataframe
    signal = data.Bx
    By = data.By
    Bz = data.Bz
    B_vector = np.array([signal,By,Bz]).transpose()
    B = np.zeros(len(B_vector[:,0]))
    for i in range(len(B_vector[:,0])):
        B_f = np.linalg.norm(B_vector[i])
        B[i] = B_f
    time = data.time

    # Diseño del filtro pasa bajos
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs  # Frecuencia de Nyquist
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    # Parámetros del filtro
    cutoff = 10  # Frecuencia de corte del filtro pasa bajos (Hz)
    order = 4    # Orden del filtro
    fs = 9000

    # Aplicar el filtro a la señal
    filtered_signal = lowpass_filter(signal, cutoff, fs, order)

    # Graficar la señal original y la filtrada
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal, label='Señal Original', alpha=0.7)
    plt.plot(time, filtered_signal, label='Señal Filtrada', color='red', linewidth=2)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.title('Filtro Pasa Bajos')
    plt.legend()
    plt.grid()
    plt.savefig('prueba.png')

if __name__== '__main__' :

  if len(sys.argv) !=2: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 PromedioPorVentana.py YYYY-MM-DD")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
    pasaBajos(YYYY,MM,DD)
  # Llama a la función para descargar datos de campo magnetico
    print('hecho')

