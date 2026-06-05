import pandas as pd
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from PreprocesamientoDatos.AcomodarDatosB import acomodarDatos

def matriz_varianza(B: list)-> np.matrix:
  """
  Computes the variance matrix M of the given list of arrays B.
  Each array in B is expected to be the component of a vector 
  (e.g., Bx, By, Bz) and should have the same length.
  
  Arguments
  ---------
  B : List of arrays with equal length

  Returns
  -------
  Matrix of variance: M_\mu\nu = <B_\mu B_\nu>-<B_\mu><B_\nu>
  """
  M = np.zeros((3,3))
  for i in range(3):
    for j in range(3):
      M[i,j] = np.mean(B[i]*B[j])-np.mean(B[i])*np.mean(B[j])
  return M

def mva_b(YYYY: str, MM: str, DD: str, t_min: float, t_max: float) -> np.matrix:
    """
    Computes the variance matrix M of the magnetic field components 
    Bx, By, Bz for a given date (YYYY, MM, DD) and time interval 
    (t_min, t_max).

    Arguments
    ---------
    YYYY : str
        Year of the data to be analyzed (e.g., '2014')
    MM : str
        Month of the data to be analyzed (e.g., '12')
    DD : str
        Day of the data to be analyzed (e.g., '25')
    t_min : float
        Minimum time (in hours) for the analysis (e.g., 0)
    t_max : float
        Maximum time (in hours) for the analysis (e.g., 24)    

    Returns
    ------- 
    Matrix of variance M for the magnetic field components Bx, By, Bz 
    and the dataframe with the data used for the analysis (filtered by time interval)
    """
    df = acomodarDatos(YYYY, MM, DD, res=0)
    df_MVA = df.loc[(df.time >= t_min) & (df.time <= t_max)]
    B_vec = [df_MVA.Bx, df_MVA.By, df_MVA.Bz]
    M_var = matriz_varianza(B_vec)
    return M_var, df_MVA  

def MinVarianceSpace(YYYY: str, MM: str, DD: str, t_min: float, t_max: float):
    """
    Computes the B vector in the space of variance

    Arguments
    ---------
    YYYY : str
        Year of the data to be analyzed (e.g., '2014')
    MM : str
        Month of the data to be analyzed (e.g., '12')
    DD : str
        Day of the data to be analyzed (e.g., '25')
    t_min : float
        Minimum time (in hours) for the analysis (e.g., 0)
    t_max : float
        Maximum time (in hours) for the analysis (e.g., 24)    

    returns
    -------
    B vector in the space of variance, and the eigenvalues and eigenvectors of the variance matrix
    """
    MVAmatrix, df = mva_b(YYYY, MM, DD, t_min, t_max)
    autoval, autovec = np.linalg.eigh(MVAmatrix)
    idx = np.argsort(autoval)[::-1]

    autoval = autoval[idx]
    autovec = autovec[:, idx]

    e1 = autovec[:, 0]
    e2 = autovec[:, 1]
    e3 = autovec[:, 2]

    # matriz de cambio de base (columnas = nueva base)
    A = np.column_stack((e1, e2, e3))

    # inversa
    A_inv = np.linalg.inv(A)
    B = df[["Bx", "By", "Bz"]].values

    B_new = B @ A_inv.T

    df_new = df.copy()

    df_new[["B1", "B2", "B3"]] = B_new

    return df_new, autoval, autovec

def  thickness_mpb(YYYY: str, MM: str, DD: str, t_min: float, t_max: float,
                    t_in_min: float, t_in_max: float, t_out_min: float, t_out_max: float):
    """
    Computes the thickness of the MPB using the variance matrix and the B vector in the space of variance

    Arguments
    ---------
    YYYY : str
        Year of the data to be analyzed (e.g., '2014')
    MM : str
        Month of the data to be analyzed (e.g., '12')
    DD : str
        Day of the data to be analyzed (e.g., '25')
    t_min : float
        Minimum time (in hours) for the analysis (e.g., 0)
    t_max : float
        Maximum time (in hours) for the analysis (e.g., 24)    

    returns
    -------
    Thickness of the MPB in kilometers
    """   
    df_new, autoval, autovec = MinVarianceSpace(YYYY, MM, DD, t_min, t_max)
    df = acomodarDatos(YYYY, MM, DD, res=0)
    df_out=df.loc[(df.time >= t_out_min) & (df.time <= t_out_max)]
    df_in=df.loc[(df.time >= t_in_min) & (df.time <= t_in_max)]

    normal = autovec[:,2]
    if normal[0] < 0:
        normal = -normal
    print(f"Normal vector to the MPB: {normal}")

    x_out = (df_out['posX'].iloc[-1]-df_out['posX'].iloc[0])
    y_out = (df_out['posY'].iloc[-1]-df_out['posY'].iloc[0])
    z_out = (df_out['posZ'].iloc[-1]-df_out['posZ'].iloc[0])
    r_vec_out = np.array([x_out, y_out, z_out])
    
    x_in = (df_in['posX'].iloc[-1]-df_in['posX'].iloc[0])
    y_in = (df_in['posY'].iloc[-1]-df_in['posY'].iloc[0])
    z_in = (df_in['posZ'].iloc[-1]-df_in['posZ'].iloc[0])
    r_vec_in = np.array([x_in, y_in, z_in])

    theta_out = np.arccos(np.dot(r_vec_out/np.linalg.norm(r_vec_out), normal/np.linalg.norm(normal)))*180/np.pi
    theta_in = np.arccos(np.dot(r_vec_in/np.linalg.norm(r_vec_in), normal/np.linalg.norm(normal)))*180/np.pi

    theta_kB = np.degrees(
    np.arccos(
        np.dot(Bm, normal)
        /(np.linalg.norm(Bm)*np.linalg.norm(normal))
    )
    )

    if theta_kB > 90:
        theta_kB = 180 - theta_kB
    
    print(f"Theta (in): {theta_in}")
    print(f"Theta (out): {theta_out}")
    print(f"Theta (kB): {theta_kB}")
    min_thickness = np.abs(np.dot(r_vec_in, normal)) #en km
    max_thickness = np.abs(np.dot(r_vec_out, normal))
    return min_thickness, max_thickness

if __name__== '__main__' :
  t_min_list = [12.675, 18.2258333333, 19.31666667,13.07388889, 5.271388889, 12.25, 9.81943277778]
  t_max_list = [12.684444, 18.235, 19.32388889, 13.08055556, 5.278611111, 12.265,  9.81948472222]
  t_out_min_list = [12.6624, 18.2166666667, 19.29877, 13.0664, 5.258838, 12.20975, 9.7333]
  t_out_max_list = [12.6938, 18.2475, 19.3414, 13.0824, 5.290666, 12.30619, 9.99986]
  t_in_min_list = [12.6764, 18.2202777778, 19.31378, 13.07095, 5.268676, 12.23922, 9.8116]
  t_in_max_list = [12.689, 18.235, 19.32459, 13.076, 5.276777, 12.26399, 9.89378]
  
  fecha = [['2015', '10', '10'], ['2016', '03', '16'],['2015', '10', '12'], ['2016', '03', '31'], ['2016', '04', '05'], ['2017','11','24'], ['2014', '12', '25']]
  i = -2
  YYYY, MM, DD = fecha[i][0], fecha[i][1], fecha[i][2]

  t_in_max = t_in_max_list[i]
  t_in_min = t_in_min_list[i]
  t_out_max = t_out_max_list[i]
  t_out_min = t_out_min_list[i]
  t_min = t_min_list[i]
  t_max = t_max_list[i]
  
  min_thickness, max_thickness = thickness_mpb(YYYY, MM, DD, t_min, t_max, t_in_min, t_in_max, t_out_min, t_out_max)

  print(f"Minimum thickness of the MPB: {min_thickness} km")
  print(f"Maximum thickness of the MPB: {max_thickness} km")

  df_new, autoval, autovec = MinVarianceSpace(YYYY, MM, DD, t_min, t_max)
  print(f"lambdas: {autoval}")
  print(f"lambda2/lambda3: {autoval[1]/autoval[2]}")

  fig, axs = plt.subplots(1,2,figsize=(14,10))
  axs[0].set_aspect('equal')
  axs[1].set_aspect('equal')
  axs[0].plot(df_new['B3'], df_new['B1'], '-')
  axs[1].plot(df_new['B2'], df_new['B1'], '-')
  plt.savefig("temp_MVA.png", dpi=300)
