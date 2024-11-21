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
from datetime import timedelta

def day_of_year(DD, MM, YYYY):
    date = datetime(YYYY, MM, DD)
    day_of_year = date.timetuple().tm_yday
    return f"{day_of_year:03d}"

def descargarDatosCampo(YYYY,MM,DD): #Define la URL segun la fecha ingresada

    DOY = str(day_of_year(int(DD), int(MM), int(YYYY)))
    url = f"https://lasp.colorado.edu/maven/sdc/public/data/sci/mag/l2/{YYYY}/{MM}/mvn_mag_l2_{YYYY}{DOY}ss1s_{YYYY}{MM}{DD}_v01_r01.sts" 
    url_pc = f"https://lasp.colorado.edu/maven/sdc/public/data/sci/mag/l2/{YYYY}/{MM}/mvn_mag_l2_{YYYY}{DOY}pc1s_{YYYY}{MM}{DD}_v01_r01.sts" 
    #esto despues lo tengo que adaptar porque estaba colapsada la pagina

# Realizar una solicitud GET a una URL
    response = requests.get(url)
    response.raise_for_status() #se fija que onda el status, por ejemplo si es 404 (todo mal), 403 (sin permisos), 200 (todo ok)
    lines = response.text.splitlines()

    response_pc = requests.get(url_pc)
    response_pc.raise_for_status() #se fija que onda el status, por ejemplo si es 404 (todo mal), 403 (sin permisos), 200 (todo ok)
    lines_pc = response_pc.text.splitlines()

#Creo carpeta para la bajada de los datos
    datosCampoPath = '/app/DatosCrudos/datos_campo_magnetico_crudos'
    datosCampoPath_pc = '/app/DatosCrudos/datos_campo_magnetico_crudos_pc'
    if not os.path.exists(datosCampoPath):
      os.makedirs(datosCampoPath)
    if not os.path.exists(datosCampoPath_pc):
      os.makedirs(datosCampoPath_pc)

#Crea u archivo para meter los datos que salen de la API
    archivoDestino = os.path.join(datosCampoPath, f"datos_{DD}-{MM}-{YYYY}.csv")
    archivoDestino_pc = os.path.join(datosCampoPath_pc, f"z_{DD}-{MM}-{YYYY}_pc.csv")
    
#por ultimo escribe en el archivo lo que sale de la API
    with open(archivoDestino, "w") as archivo:
      for line in lines:
        if line.strip() and line.startswith('  ' + YYYY):
            archivo.write(line + '\n')

    
#guardo solo la coordenada z en coordenadas centradas en el planeta
    with open(archivoDestino_pc, "w") as archivo:
      for line in lines_pc:
        if line.strip() and line.startswith('  ' + YYYY):
            columns = line.split()
            archivo.write(columns[13] + '\n')

if __name__== '__main__' :

  if len(sys.argv) not in (2,3): #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 DescargaDatosB.py YYYY_i-MM_i-DD_i YYYY_f-MM_f-DD_f (fecha_inicio fecha_fin, fecha_fin es opcional)")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
  # Llama a la función para descargar datos de campo magnetico
    descargarDatosCampo(YYYY,MM,DD)
    print('hecho')
  elif len(sys.argv) == 3:
    parametro_inicial = sys.argv[1]
    parametro_final = sys.argv[2]
    YYYY_i, MM_i, DD_i = (int(f) for f in parametro_inicial.split('-'))
    YYYY_f, MM_f, DD_f = (int(f) for f in parametro_final.split('-'))
    fecha_inicial = datetime(YYYY_i,MM_i,DD_i).date()
    fecha_final = datetime(YYYY_f,MM_f,DD_f).date()
    print(f"Descargando datos en rango {fecha_inicial} - {fecha_final}")
    fecha_actual = fecha_inicial
    count = 1
    while fecha_actual <= fecha_final:
      print(f"Descargando dia numero {count} de {fecha_final-fecha_inicial}")
      descargarDatosCampo(fecha_actual.strftime('%Y'),
                          fecha_actual.strftime('%m'),
                          fecha_actual.strftime('%d'))
      fecha_actual += timedelta(days=1)
      count += 1