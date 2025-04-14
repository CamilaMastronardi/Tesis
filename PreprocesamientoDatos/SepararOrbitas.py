import pandas as pd
import sys
import matplotlib.pyplot as plt
from AcomodarDatosB import acomodarDatos
import numpy as np
import os

#plt.style.use("matplotlibStyles.txt")

def separarOrbitas(YYYY: str, MM: str, DD: str) -> (pd.DataFrame, pd.DataFrame):

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

if __name__ == '__main__':

    if len(sys.argv) != 2:  # Verifica que se haya ingresado un parámetro después del nombre del programa (argv[0])
        print("Uso: python separarOrbitas.py YYYY-MM-DD")
        sys.exit(1)  # Sale del programa
    
    if len(sys.argv) == 2:
        fecha = sys.argv[1]  # Usa el argumento indicado para ejecutar el programa
        YYYY, MM, DD = fecha.split('-')
        df_orbitas_positivas = separarOrbitas(YYYY, MM, DD)