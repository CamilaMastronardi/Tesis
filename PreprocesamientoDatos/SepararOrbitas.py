import pandas as pd
import sys
import matplotlib.pyplot as plt
from AcomodarDatosB import acomodarDatos
import numpy as np
import os

plt.style.use("matplotlibStyles.txt")

def separarOrbitas(YYYY: str, MM: str, DD: str) -> list[pd.DataFrame, pd.DataFrame]:

    df = acomodarDatos(YYYY, MM, DD)
    latitud = pd.read_csv(f'/app/DatosCrudos/datos_campo_magnetico_crudos_pc/z_{DD}-{MM}-{YYYY}_pc.csv', header=None, lineterminator='\n')[:-1]
    # Identifico cambios de signo en posX
    df['cambio'] = (df['posX'] > 0) & (df['posX'].shift(1) <= 0)
    df['orbita'] = df['cambio'].cumsum() # Creo columna para las órbitas 

    # Filtro filas donde posX es positivo
    df_positivas = df[df['posX'] > 0]
    latitud_x_positivas = latitud[df['posX'] > 0]
    df_orbitas_positivas = df_positivas.dropna()
    
    # Elimino la columna auxiliar 'cambio'
    df_orbitas_positivas = df_orbitas_positivas.drop(columns=['cambio'])
    
    return df_orbitas_positivas, latitud_x_positivas

def graficarOrbitas(YYYY: str, MM: str, DD: str):
    df_latitud_positivas, latitud_x_positivas = separarOrbitas(YYYY, MM, DD)
    rm = 3389.5 # Radio de Marte en km
    n_orbitas = df_latitud_positivas['orbita'].max() # Obtengo el número de órbitas
    orbita = 4
    df_latitud_positivas_orbita = df_latitud_positivas[df_latitud_positivas['orbita'] == orbita].rolling(15).mean().dropna()
    fig, ax = plt.subplots(figsize=(24, 8))
    ax.plot(df_latitud_positivas_orbita['time'], df_latitud_positivas_orbita['mod_B'], label='Campo Magnético')
    ax2 = ax.twinx()
    ax2.plot(df_latitud_positivas_orbita['time'], df_latitud_positivas_orbita['r_sat']/rm+1, label= 'Altura', color = 'mediumorchid')
    
    # Pintar la región donde t < 17.6
    ax.axvspan(17.01, 17.64, alpha=0.1, color = 'palevioletred')
    ax.text(17.2, 25, 'Viento Solar', fontsize= 28)
    ax.axvspan(17.7, 17.95, alpha=0.1, color = 'blue')
    ax.text(17.72, 25, 'Magnetofunda', fontsize= 28)
    ax.vlines(x=17.666, ymin=0, ymax=27, color='deeppink', linestyle='--', lw = 1.2, label='Bow Shock')
    ax.vlines(x=17.97, ymin=0, ymax=27, color='teal', linestyle='--', lw=1.2, label='MPB')
    
    ax.set_xlim((17,18.4))
    ax.set_ylim((0, 27))
    ax.set_xlabel('Tiempo (hs)')
    ax.set_ylabel('Campo Magnético (nT)')
    ax2.set_ylabel('Altura (rm)', color='mediumorchid')
    ax2.tick_params(colors='mediumorchid', which = 'major')
    ax.legend(loc='center left')
    plt.show()
    # Guardar la figura
    fig.savefig(f'orbitas_{YYYY}_{MM}_{DD}_{orbita}.png', bbox_inches='tight')

if __name__ == '__main__':

    if len(sys.argv) != 2:  # Verifica que se haya ingresado un parámetro después del nombre del programa (argv[0])
        print("Uso: python separarOrbitas.py YYYY-MM-DD")
        sys.exit(1)  # Sale del programa
    
    if len(sys.argv) == 2:
        fecha = sys.argv[1]  # Usa el argumento indicado para ejecutar el programa
        YYYY, MM, DD = fecha.split('-')
        graficarOrbitas(YYYY, MM, DD)