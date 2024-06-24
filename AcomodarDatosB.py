# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

col_names = ["año","nro_día","hora","minuto", "segundo", "microsegundo", "dia decimal", "Bx", "By", "Bz", "rangoB", "posX", "posY", "posZ", "motorX", "motorY", "motorZ", "rango_motor"]

def acomodarDatos(YYYY,MM,DD):
    
    '''
    df = pd.read_csv(f'/home/camila/Escritorio/Tesis/datos_campo_magnetico/datos_{DD}-{MM}-{YYYY}.csv', 
                     sep='\s+',skiprows=149, header=None, lineterminator='\n', names = col_names, 
                     usecols=['año','nro_día', 'hora', 'minuto', 'segundo', 'Bx', 'By', 'Bz', 'rangoB'])
    '''

    df = pd.read_csv(f'C:/Escritorio/Tesis/datos_campo_magnetico/datos_{DD}-{MM}-{YYYY}.csv', 
                     header=None, lineterminator='\n', sep='\s+',skiprows=149, names = col_names, 
                     usecols=['año','nro_día', 'hora', 'minuto', 'segundo', 'Bx', 'By', 'Bz', 'rangoB'])
    
    mes = round(df.nro_día/30) + 1
    dia = df.nro_día - (mes-1)
    
    hora = df.hora
    minuto = df.minuto
    seg = df.segundo
    
    time = hora + minuto/60 + seg/3600
    
    B_vector = np.array([df.Bx,df.By,df.Bz]).transpose()
    B_norm = np.zeros(len(B_vector[:,0]))
    for i in range(len(B_vector[:,0])):
        B = np.linalg.norm(B_vector[i])
        B_norm [i] = B
        
    return(pd.DataFrame({'time': time, 'mod_B': B_norm, 'Bx': df.Bx, 'By':df.By, 'Bz':df.Bz})[:-1])

if __name__== '__main__' :

  if len(sys.argv) !=2: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python acomodarDatosB.py YYYY-MM-DD")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
  # Llama a la función para acomodar datos
    acomodarDatos(YYYY,MM,DD)