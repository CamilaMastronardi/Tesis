#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:43:38 2024

@author: camila
"""
import sys
import os
import requests
from datetime import datetime

def day_of_year(DD, MM, YYYY):
    date = datetime(YYYY, MM, DD)
    day_of_year = date.timetuple().tm_yday
    return f"{day_of_year:03d}"

def descargarDatosCampo(YYYY,MM,DD): #Define la URL segun la fecha ingresada

    DOY = str(day_of_year(int(DD), int(MM), int(YYYY)))
    url = f"https://lasp.colorado.edu/maven/sdc/public/data/sci/mag/l2/{YYYY}/{MM}/mvn_mag_l2_{YYYY}{DOY}ss1s_{YYYY}{MM}{DD}_v01_r01.sts" 
    #esto despues lo tengo que adaptar porque estaba colapsada la pagina

# Realizar una solicitud GET a una URL
    response = requests.get(url)
    response.raise_for_status() #se fija que onda el status, por ejemplo si es 404 (todo mal), 403 (sin permisos), 200 (todo ok)

#Creo carpeta para la bajada de los datos
    #datosCampoPath = "/Escritorio/Tesis/datos_campo_magnetico" #ESTE ES EL QUE USO EN MI PC
    #datosCampoPath = "/home/camila/Escritorio/Camila/datos_campo_magnetico"#ESTE ES EL QUE USO EN EL IAFE
    datosCampoPath = '/app/datos_campo_magnetico'
    if not os.path.exists(datosCampoPath):
      os.makedirs(datosCampoPath)

#Crea u archivo para meter los datos que salen de la API
    archivoDestino = os.path.join(datosCampoPath, f"datos_{DD}-{MM}-{YYYY}.csv")
    #por ultimo escribe en el archivo lo que sale de la API
    with open(archivoDestino, "w") as archivo:
        archivo.write(response.text)

if __name__== '__main__' :

  if len(sys.argv) !=2: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python DescargaDatosB.py YYYY-MM-DD")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
  # Llama a la función para descargar datos de campo magnetico
    descargarDatosCampo(YYYY,MM,DD)
    print('hecho')
