import os
import pandas as pd

# Función para dividir datos en intervalos con solapamiento
def segmentar(data, window_size=60, overlap=30):

    step = window_size - overlap
    splits = []
    for i in range(0, len(data) - window_size + 1, step):
        splits.append(data[i:i + window_size])
    return splits

# Función principal
def dataClasificada(folder_path, MPB_time, window_size=60, overlap=30):

    normales = {}  # Diccionario para almacenar datos normales
    anormales = {}  # Diccionario para almacenar datos anormales
    
    # Lista de todos los archivos en la carpeta
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    MPB_time = os.path.join(MPB_path, f)
    
    for i,file in zip(len(files),files):

        file_path = os.path.join(folder_path, file)
        try:
            data = pd.read_csv(file_path)
            
            # Clasificar los datos
            if data[i]==MPB_time[i]:
                # Si contiene valores anormales, se agrega al diccionario de anormales
                splits = segmentar(data.values, window_size=window_size, overlap=overlap)
                anormales[file] = [pd.DataFrame(split, columns=data.columns) for split in splits]
            else:
                # Si no contiene valores anormales, se agrega al diccionario de normales
                splits = segmentar(data.values, window_size=window_size, overlap=overlap)
                normales[file] = [pd.DataFrame(split, columns=data.columns) for split in splits]
        
        except Exception as e:
            print(f"Error al procesar {file}: {e}")
    
    return normales, anormales