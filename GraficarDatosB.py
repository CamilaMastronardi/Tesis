# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:18:51 2024

@author: cami9
"""

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt
from AcomodarDatosB import acomodarDatos
import pandas as pd
import sys

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.labelsize'] = 22

YYYY = '2017'
MM = '11'
DD = '24'

hra = (12*60 + 15)/60

data = acomodarDatos(YYYY, MM, DD)[:-1]

t = data.time
Bx = data.Bx
By = data.By
Bz = data.Bz
B = data.mod_B
r_sat = data.r_sat

# Plot campo con r superpuesto
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 14))
ax1.set_title(f'{DD}-{MM}-{YYYY}')
ax2 = ax1.twinx()
ax4 = ax3.twinx()
ax1.plot(t, B, color = 'aquamarine')
ax2.plot(t, r_sat, '-', color = 'purple', alpha = 0.7)
ax1.set_xlabel('tiempo [hs]')
ax1.set_ylabel('|B| [N]')
ax1.set_ylim((0,50))
ax2.set_ylabel('altura [Km]')
ax1.grid()
ax3.plot(t, B, color = 'aquamarine')
ax4.plot(t, r_sat, '-', color = 'purple', alpha = 0.7)
ax3.set_xlim(hra - 0.5, hra + 0.5)
ax3.set_ylim((0,50))
ax3.set_xlabel('tiempo [hs]')
ax3.set_ylabel('|B| [N]')
ax4.set_ylabel('altura [Km]')
ax3.grid()
plt.savefig(f'/app/PrimerosGraficosB/B_{DD}_{MM}_{YYYY}')


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))
ax1.set_title(f'{DD}-{MM}-{YYYY}')
ax1.plot(t, Bx, color = 'aquamarine', label = 'B_x')
ax1.plot(t, By, color = 'plum', label = 'B_y')
ax1.plot(t, Bz, color = 'skyblue', label = 'B_z')
ax1.set_ylim(-30,40)
ax1.set_xlabel('tiempo [hs]')
ax1.set_ylabel('Componente de campo [N]')
ax1.grid()
ax1.legend(loc='best')
ax2.plot(t, Bx, color = 'aquamarine')
ax2.plot(t, By, color = 'plum')
ax2.plot(t, Bz, color = 'skyblue')
ax2.set_xlim(hra - 0.5, hra + 0.5)
ax2.set_ylim((-35,35))
ax2.set_xlabel('tiempo [hs]')
ax2.set_ylabel('|B| [N]')
ax2.grid()
plt.savefig(f'/app/PrimerosGraficosB/Bvector_{DD}_{MM}_{YYYY}')