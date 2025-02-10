import pandas as pd
import os
import sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, 'PreprocesamientoDatos'))

from DescargaDatosB import descargarDatosCampo

def cargarTrainingData(group: str): #group es el nombre de la pestaña de datos de gabi que quiera usar

    path = f'/app/MAVEN_MPB_Data_{group}.csv'
    col_names = ['date','BS time','MPB_time','ThetaBn (deg) (Cyril)','Pdyn_proton (Halekas)','','beta_proton (Halekas)']
    df = pd.read_csv(path, skiprows=1, header=None,sep=',' ,lineterminator='\n', names = col_names, usecols=['date','MPB_time'])
    dates = df.date.str.split('-')

    df.MPB_time = df.MPB_time.str.split(':').apply(lambda t: int(t[0])+int(t[1])/60+int(t[2])/3600)

    for date in dates:
        descargarDatosCampo(date[0], date[1], date[2])

    date_MPB = pd.DataFrame(dates.tolist(), columns = ['YYYY', 'MM', 'DD'])
    date_MPB['MPB_time'] = df.MPB_time

    return date_MPB

if __name__ == '__main__':
    if len(sys.argv)==2:
        path = f'/app/MAVEN_MPB_Data_{sys.argv[1]}.csv'
        if os.path.exists(path):
            cargarTrainingData(sys.argv[1])
            print(f'Datos para entrenamiento de {sys.argv[1]} descargado')
        else: 
            print(f'No se encuentra el archivo de datos de MPB')
    else: 
        print(f'Uso: python3 DescargaTrainingData.py Grupo_de_entrenamiento')