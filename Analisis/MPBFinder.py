# Librerias basicas para manejo de datos
import numpy as np
import pandas as pd
import os

#para la visualización
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.patches as mpatches
plt.style.use("./matplotlibStyles.txt")

# Librerias basicas para manejo de datos
# Carpeta donde está este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
archivo = os.path.join(BASE_DIR,'../KNN/Clasificador/Prueba/Gabi_3clusters_analysis.csv')

def string_to_array(string: str):
    '''Convierte una cadena con formato de array (ej: "[1 2 3]") a un array de numpy.'''
    string = string.strip("[]")
    return np.fromstring(string, sep=' ')

columnas_arrays = ["B", "time", "posX", "posY", "posZ"]
df = pd.read_csv(archivo, index_col=0)
for col in columnas_arrays:
    df[col] = df[col].apply(string_to_array)


def detectar_mpb(predicciones: list, n_consecutivos: int = 2):
    '''Detecta el índice donde ocurre la transición entre clases 1
      y 2 (MPB) en una lista de predicciones.'''
    pred = np.array(predicciones)
    for i in range(len(pred) - 2*n_consecutivos + 1):

        bloque_1 = pred[i:i+n_consecutivos]
        bloque_2 = pred[i+n_consecutivos:i+2*n_consecutivos]

        # 1 -> 2
        if (
            np.all(bloque_1 == 1)
            and
            np.all(bloque_2 == 2)
        ):

            return i + n_consecutivos

        # 2 -> 1
        if (
            np.all(bloque_1 == 2)
            and
            np.all(bloque_2 == 1)
        ):

            return i + n_consecutivos

    return None

if __name__ == "__main__":

    colores = {
        0: "tab:blue",
        1: "tab:orange",
        2: "tab:green",
        3: "tab:red"
    }

    grupos_orbita = df.groupby("orbita")

    for orbita, df_orbita in grupos_orbita:

        plt.figure(figsize=(10,6))

        labels_usados = set()

        # =========================
        # Plot de las ventanas
        # =========================

        for _, fila in df_orbita.iterrows():

            time = fila["time"]
            B = fila["B"]
            pred = fila["pred"]

            color = colores.get(pred, "black")

            if pred not in labels_usados:

                plt.plot(
                    time,
                    B,
                    color=color,
                    alpha=0.6,
                    label=f"Clase {pred}"
                )

                labels_usados.add(pred)

            else:

                plt.plot(
                    time,
                    B,
                    color=color,
                    alpha=0.6
                )

        # =========================
        # Detectar MPB
        # =========================

        preds = df_orbita["pred"].values

        indice_mpb = detectar_mpb(preds)

        if indice_mpb is not None:

            fila_mpb = df_orbita.iloc[indice_mpb]

            time_array = fila_mpb["time"]

            # tiempo central de la ventana
            time_mpb = time_array[len(time_array)//2]

            plt.axvline(
                x=time_mpb,
                color="black",
                linestyle="--",
                linewidth=2,
                label="MPB"
            )

            print(f"Órbita {orbita}: MPB encontrada en t={time_mpb}")

        else:

            print(f"Órbita {orbita}: MPB no encontrada")

        plt.xlabel("Time")
        plt.ylabel("|B|")

        plt.title(f"Órbita {orbita}")

        plt.legend()

        plt.grid(True)

        nombre = f"orbita_{orbita}.png"

        plt.savefig(nombre, dpi=300, bbox_inches='tight')

        plt.close()

        print(f"Guardado: {nombre}")