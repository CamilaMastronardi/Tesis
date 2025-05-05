import pandas as pd
import sys
import matplotlib.pyplot as plt
from SepararOrbitas import separarOrbitas
import numpy as np

plt.style.use("./matplotlibStyles.txt")

def separarHemisferios(YYYY: str, MM: str, DD: str) -> pd.DataFrame:
    #defino funcion para eliminar hemisferios espacios 
    LIMITE_ESPACIADO_TIEMPO = 50/3600
    def tieneEspaciadoTemporal(data_frame):
        data_frame['espaciado_t'] = (data_frame['time'] - data_frame['time'].shift(1) >= LIMITE_ESPACIADO_TIEMPO)
        if data_frame['espaciado_t'].any(): 
            return False
        else: 
            return True 

    df, latitud = separarOrbitas(YYYY, MM, DD)
    
    # Identifico cambios de signo en z de pc
    latitud['cambio'] = ((latitud > 0) & (latitud.shift(1) <= 0)) | ((latitud < 0) & (latitud.shift(1) >= 0)) 
    df['hemisferio'] = latitud['cambio'].cumsum() # Creo columna para hemisferios

    # Filtro filas donde la latitud es mayor al ecuador
    df_latitud_positivas = df[(latitud[0]>0) | (df['posX'] > 1000)]

    df_latitud_positivas = df_latitud_positivas.groupby('orbita').filter(tieneEspaciadoTemporal)

    return df_latitud_positivas

def graficarHemisferios(YYYY: str, MM: str, DD: str):
    df_latitud_positivas = separarHemisferios(YYYY, MM, DD)
    # Graficar los datos separados por hemisferios
    fig, ax = plt.subplots(figsize=(24, 7))
    ax.plot(df_latitud_positivas['time'], df_latitud_positivas['mod_B'], label='Campo Magnético', color='blue')
    ax.set_xlabel('Tiempo (hs)')
    ax.set_ylabel('Campo Magnético (nT)')
    ax.set_ylim(0,50)
    plt.legend()
    plt.show()
    # Guardar la figura
    fig.savefig(f'hemisferios_{YYYY}_{MM}_{DD}.png')

if __name__ == '__main__':

    if len(sys.argv) != 2:  # Verifica que se haya ingresado un parámetro después del nombre del programa (argv[0])
        print("Uso: python SepararHemisferios.py YYYY-MM-DD")
        sys.exit(1)  # Sale del programa
    
    if len(sys.argv) == 2:
        fecha = sys.argv[1]  # Usa el argumento indicado para ejecutar el programa
        YYYY, MM, DD = fecha.split('-')
        graficarHemisferios(YYYY, MM, DD)