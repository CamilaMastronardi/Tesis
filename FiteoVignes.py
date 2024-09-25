import sys
import os
import numpy as np
from AcomodarDatosB import acomodarDatos
from scipy.optimize import curve_fit


def r_fiteoVignes(theta, L): #Importa los datos acomodados de campo magnetico y hace el ajuste
    epsilon = 1
    return L/(1-epsilon*np.cos(theta))

def intervaloVignes(r_MPB_estimativo):
    popt, pcov = curve_fit(r_MPB_estimativo['r'],r_MPB_estimativo['theta'], r_fiteoVignes) #hago un ajuste por cuadrados minimos
    L = popt 
    L_err = np.sqrt(np.linalg(pcov))
    deltaL = 0
    intervaloL = np.array([L-deltaL, L+deltaL])
    return(intervaloL)

def filtradoVignes(intervaloL, YYYY, MM, DD): 
    def r_min(theta): 
        epsilon = 1
        L = intervaloL[0]
        return L/(1-epsilon*np.cos(theta))
    def r_max(theta): 
        epsilon = 1
        L = intervaloL[1]
        return L/(1-epsilon*np.cos(theta))
    indice
    for i in range(len(r_MPB_estimativo)): 
        


if __name__== '__main__' :

  if len(sys.argv) !=2: #se fija que se haya ingresado un parametro despues del nombre del programa (argv[0])
        print("Uso: python3 DescargaDatosB.py YYYY-MM-DD")
        sys.exit(1) #sale del programa
    # Pide al usuario que ingrese la fecha en formato YYYY-MM-DD
    
  if len(sys.argv) == 2:
    fecha = sys.argv[1] #Usa el argumento indicado para ejecutar el programa
    YYYY, MM, DD = fecha.split('-')
  # Llama a la función para descargar datos de campo magnetico
    descargarDatosCampo(YYYY,MM,DD)
    print('hecho')
