import pandas as pd
import sys
import matplotlib.pyplot as plt
from AcomodarDatosB import acomodarDatos
from SepararOrbitas import separarOrbitas
import numpy as np

plt.style.use("./matplotlibStyles.txt")

def separarHemisferios(YYYY, MM, DD):
    #defino funcion para eliminar hemisferios espacios 
    LIMITE_ESPACIADO_TIEMPO = 50/3600
    def tieneEspaciadoTemporal(data_frame):
        data_frame['espaciado_t'] = (data_frame['time'] - data_frame['time'].shift(1) >= LIMITE_ESPACIADO_TIEMPO)
        if data_frame['espaciado_t'].any(): 
            print(data_frame['espaciado_t'])
            return False
        else: 
            return True 

    df, latitud = separarOrbitas(YYYY, MM, DD)
    
    # Identifico cambios de signo en z de pc
    latitud['cambio'] = ((latitud > 0) & (latitud.shift(1) <= 0)) | ((latitud < 0) & (latitud.shift(1) >= 0)) 
    df['hemisferio'] = latitud['cambio'].cumsum() # Creo columna para hemisferios

    # Filtro filas donde la latitud es positivo
    df_latitud_positivas = df[(latitud[0]>0) | ((latitud[0]>-3500) & (df['posX'] > 1000))]

    df_latitud_positivas = df_latitud_positivas.groupby('orbita').filter(tieneEspaciadoTemporal)

    return df_latitud_positivas

if __name__ == '__main__':

    if len(sys.argv) != 2:  # Verifica que se haya ingresado un parámetro después del nombre del programa (argv[0])
        print("Uso: python separarOrbitas.py YYYY-MM-DD")
        sys.exit(1)  # Sale del programa
    
    if len(sys.argv) == 2:
        fecha = sys.argv[1]  # Usa el argumento indicado para ejecutar el programa
        YYYY, MM, DD = fecha.split('-')
        df_latitud_positivas = separarHemisferios(YYYY, MM, DD)