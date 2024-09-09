import pandas as pd
import sys
import matplotlib.pyplot as plt
from AcomodarDatosB import acomodarDatos
from SepararOrbitas import separarOrbitas
plt.style.use("./matplotlibStyles.txt")

def separarHemisferios(YYYY, MM, DD):

    df, latitud = separarOrbitas(YYYY, MM, DD)
    
    # Identifico cambios de signo en z de pc
    latitud['cambio'] = ((latitud > 0) & (latitud.shift(1) <= 0)) | ((latitud < 0) & (latitud.shift(1) >= 0)) 
    df['hemisferio'] = latitud['cambio'].cumsum() # Creo columna para hemisferios

    # Filtro filas donde la latitud es positivo
    df_latitud_positivas = df[latitud['cambio']==True]
    
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(latitud[0], '.',label='latitud')
    ax1.plot(latitud[0][latitud['cambio']==True],'o', color='darkblue')
    ax2.plot(df['mod_B'], label='modulo de B')
    ax2.plot(df['mod_B'][latitud['cambio']==True],'o', color='darkblue')
    plt.legend()
    plt.savefig('prueba.jpg')
    
    return df_latitud_positivas

if __name__ == '__main__':

    if len(sys.argv) != 2:  # Verifica que se haya ingresado un parámetro después del nombre del programa (argv[0])
        print("Uso: python separarOrbitas.py YYYY-MM-DD")
        sys.exit(1)  # Sale del programa
    
    if len(sys.argv) == 2:
        fecha = sys.argv[1]  # Usa el argumento indicado para ejecutar el programa
        YYYY, MM, DD = fecha.split('-')
        df_latitud_positivas = separarHemisferios(YYYY, MM, DD)