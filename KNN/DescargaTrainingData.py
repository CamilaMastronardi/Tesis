import pandas as pd
import os
import sys
from requests.exceptions import HTTPError

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, 'PreprocesamientoDatos'))

from DescargaDatosB import descargarDatosCampo

def cargarTrainingData(groups: list[str]): #group es el nombre de la pestaña de datos de gabi que quiera usar
    date_tot = pd.DataFrame(columns = ['YYYY', 'MM', 'DD', 'MPB_time'])
    tot_drop = 0
    for group in groups:
        path = f'/app/Training_Data/MAVEN_MPB_Data_{group}.csv'
        col_names = ['date','MPB_time']
        df = pd.read_csv(path, skiprows=1, header=None, sep=',' ,lineterminator='\n', usecols=[0,2]).dropna()
        df.columns = col_names
        dates = df.date.str.split('-')
        path_cache_not_avaiable = f'/app/PreprocesamientoDatos/cache/not_avaiable_date.csv'
        dates_not_avaiable = pd.read_csv(path_cache_not_avaiable, skiprows=0, header=None, sep=',', lineterminator='\n', names = ['YYYY', 'MM', 'DD'] ,dtype=int)
        df.MPB_time = df.MPB_time.str.split(':').apply(lambda t: int(t[0])+int(t[1])/60+int(t[2])/3600)

        idxs_to_drop = []
        for idx, date in dates.items():
            if ((int(date[0])==dates_not_avaiable["YYYY"]) & (int(date[1]) == dates_not_avaiable["MM"]) & (int(date[2]) == dates_not_avaiable["DD"])).any():
                idxs_to_drop.append(idx) 
                continue
            
            else:
                try:
                    descargarDatosCampo(date[0], date[1], date[2])
                except HTTPError as e:
                    if e.response.status_code == 404:
                        print(f'Link no disponible para {date}')
                    else:
                        raise

        tot_drop = tot_drop + len(idxs_to_drop)

        dates.drop(index = idxs_to_drop, inplace = True)

        date_MPB = pd.DataFrame(dates.tolist(), columns = ['YYYY', 'MM', 'DD'])
        date_MPB['MPB_time'] = df.MPB_time
        date_tot = pd.concat([date_tot, date_MPB.dropna()])
        
    print(f'Ignoradas {tot_drop} por cache')
    return date_tot.reset_index()

def cargarData(file_name: str):
    date_tot = pd.DataFrame(columns = ['YYYY', 'MM', 'DD'])
    tot_drop = 0

    path = f'/app/data_a_clasificar/{file_name}.csv'
    col_names = ['date']
    df = pd.read_csv(path, skiprows=1, header=None, sep=',' ,lineterminator='\n').dropna()
    df.columns = col_names
    dates = df.date.str.split('-')
    path_cache_not_avaiable = f'/app/PreprocesamientoDatos/cache/not_avaiable_date.csv'
    dates_not_avaiable = pd.read_csv(path_cache_not_avaiable, skiprows=0, header=None, sep=',', lineterminator='\n', names = ['YYYY', 'MM', 'DD'] ,dtype=int)
    idxs_to_drop = []
    for idx, date in dates.items():
        if ((int(date[0])==dates_not_avaiable["YYYY"]) & (int(date[1]) == dates_not_avaiable["MM"]) & (int(date[2]) == dates_not_avaiable["DD"])).any():
            idxs_to_drop.append(idx) 
            continue
        
        else:
            try:
                descargarDatosCampo(date[0], date[1], date[2])
            except HTTPError as e:
                if e.response.status_code == 404:
                    print(f'Link no disponible para {date}')
                else:
                    raise

        tot_drop = tot_drop + len(idxs_to_drop)

        dates.drop(index = idxs_to_drop, inplace = True)

        date_to_classified = pd.DataFrame(dates.tolist(), columns = ['YYYY', 'MM', 'DD'])

    print(f'Ignoradas {tot_drop} por cache')
    return date_to_classified.reset_index()

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