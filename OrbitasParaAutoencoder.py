import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Filtros import lowpass_filter, highpass_filter 
from PromedioPorVentana import filtrarVentana, n_orbita
from CorteVignes import filtradoVignes

plt.style.use("./matplotlibStyles.txt")

def guardadoDataframeAutoencoder(YYYY, MM, DD, orbita):
    df = filtradoVignes(YYYY, MM, DD, orbita)
    
    PathDataAutoencoder = '/app/Autoencoder'

    if not os.path.exists(PathDataAutoencoder):
        os.makedirs(PathDataAutoencoder)
    
    archivoDestino = os.path.join(PathDataAutoencoder, f"Autoencoder_{DD}-{MM}-{YYYY}_{orbita}.csv")
    
    with open(archivoDestino, "w") as archivo:
        archivo.write(df)

    return df


def graficoParaSepararCasos(YYYY, MM, DD, orbita):
    df = guardadoDataframeAutoencoder(YYYY, MM, DD, orbita)
    B, Bx, By, Bz, t, posX, posY, posZ = df['mod_B'], df['Bx'], df['By'], df['Bz'], df['time'], df['posX'], df['posY'], df['posZ']
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.set_title(f'{YYYY}-{MM}-{DD} orbita {orbita}')
    ax1.plot(t, B)
    ax1_twin = plt.twiny(ax1)
    ax1_twin.plot(t, posX)
    ax1_twin.plot(t, posY)
    ax1_twin.plot(t, posZ)
    ax1.set_xlabel('time (hs)')
    ax1.set_ylabel('B (nT)')
    ax1_twin.set_ylabel('posición (m)')
    
    ax2.plot(t, Bx, label = '$B_x$')
    ax2.plot(t, By, label = '$B_y$')
    ax2.plot(t, Bz, label = '$B_z$')
    ax2.set_xlabel('time (hs)')
    ax2.set_ylabel('$campo magnetico$ (nT)')
    
    PathFig = '/app/Autoencoder/FigurasParaSepararCasos'

    if not os.path.exists(PathFig):
        os.makedirs(PathFig)

    plt.savefig(os.path.join(PathFig, f'Autoencoder_{YYYY}-{MM}-{DD}_orbita_{orbita}.jpg'))

if __name__== '__main__' :

  if len(sys.argv) !=3: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 PromedioPorVentana.py YYYY-MM-DD nºorbita")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 3:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    orbita = sys.argv[2] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
    graficoParaSepararCasos(YYYY, MM, DD, orbita)
  # Llama a la función para descargar datos de campo magnetico
    print('hecho')  