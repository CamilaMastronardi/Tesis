# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:18:51 2024

@author: cami9
"""

import numpy as np
from matplotlib import pyplot as plt
from AcomodarDatosB import acomodarDatos
import pandas as pd
import sys

data = acomodarDatos(2014, 12, 25) #es un pd.dataframe

#le aplico un promedio por ventanas a cada coordenada del campo magnetico 

Bx_filtrado = data['Bx'].rolling(1000, center = True).sum()
By_filtrado = data['By'].rolling(1000, center = True).sum()
Bz_filtrado = data['Bz'].rolling(1000, center = True).sum()
time_filtrado = data['time'].rolling(1000, center = True).sum()


if __name__ == '__main__':
    plt.plot(time_filtrado, Bx_filtrado)
    plt.plot(time_filtrado, By_filtrado)
    plt.plot(time_filtrado, Bz_filtrado)
    plt.savefig('prueba3.jpg')
