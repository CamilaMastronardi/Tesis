# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import requests
from datetime import datetime

col_names = ["año","nro_día","hora","minuto", "segundo", "milisegundo", "dia decimal", "Bx", "By", "Bz", "rangoB", "posX", "posY", "posZ", "motorX", "motorY", "motorZ", "rango_motor"]

def day_of_year(DD, MM, YYYY):
    date = datetime(YYYY, MM, DD)
    day_of_year = date.timetuple().tm_yday
    return f"{day_of_year:03d}"

def compararDatos(YYYY,MM,DD):
    
    DOY = str(day_of_year(int(DD), int(MM), int(YYYY)))
    
    #IAFE
    df_maven = pd.read_csv(f'/home/camila/Escritorio/Camila/datos_campo_magnetico/datos_pds/mvn_mag_l2_{YYYY}{DOY}ss1s_{YYYY}{MM}{DD}_v01_r01.csv', 
                     sep='\s+',skiprows=149, header=None, lineterminator='\n', names = col_names, 
                      usecols=['año','nro_día', 'hora', 'minuto', 'segundo', 'Bx', 'By', 'Bz', 'rangoB', 'posX', 'posY', 'posZ'],low_memory=False)
    
    df_pds = pd.read_csv(f'/home/camila/Escritorio/Camila/datos_campo_magnetico/datos_{DD}-{MM}-{YYYY}.csv', 
                     sep='\s+',skiprows=149, header=None, lineterminator='\n', names = col_names, 
                      usecols=['año','nro_día', 'hora', 'minuto', 'segundo', 'Bx', 'By', 'Bz', 'rangoB', 'posX', 'posY', 'posZ'],low_memory=False)
    if df_maven.equals(df_pds):
    	print(f'para la fecha {DD}-{MM}-{YYYY} los datos son iguales')
    else:
    	print(f'para la fecha {DD}-{MM}-{YYYY} los datos NO son iguales')

if __name__== '__main__' :

  if len(sys.argv) !=2: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python acomodarDatosB.py YYYY-MM-DD")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
  # Llama a la función para acomodar datos
    compararDatos(YYYY,MM,DD)
