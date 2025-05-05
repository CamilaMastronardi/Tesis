from KNN import DTW, completeData
from PromedioPorVentana import filtrarVentana
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numba import njit
import seaborn as sns
import matplotlib.gridspec as gridspec

def distance_matrix(data_1: pd.DataFrame, data_2: pd.DataFrame):
    tot_dist = []
    for i in data_1:
        dist_i = []
        for j in data_2:
            distance = np.abs(i - j)/np.sqrt(i**2+j**2)
            dist_i.append(distance)
        tot_dist.append(dist_i)
    return np.matrix(tot_dist)

def cost_matrix(distance_matrix):
    N, M = distance_matrix.shape  
    cost = np.full((N, M), np.inf)
    cost[0, 0] = distance_matrix[0, 0]
    
    for i in range(1, N):
        cost[i, 0] = cost[i-1, 0] + distance_matrix[i,0]
    
    for j in range(1, M):
        cost[0, j] = cost[0, j-1] + distance_matrix[0,j]
    
    for i in range(1, N):
        for j in range(1, M):
            cost[i, j] = min(cost[i-1, j-1], cost[i, j-1], cost[i-1, j]) + distance_matrix[i,j]
    return cost

def camino_de_menor_costo(cost_matrix):
    N, M = cost_matrix.shape
    i, j = N-1, M-1
    path_i = [i]
    path_j = [j]
    
    while i > 0 or j > 0:
        p1 = cost_matrix[i-1,j]
        p2 = cost_matrix[i,j-1]
        p3 = cost_matrix[i-1,j-1]
        if p1 < p2 and p1 < p3:
            i += -1
        elif p2 < p1 and p2 < p3:
            j += -1
        else:   
            i += -1
            j += -1
        path_i.append(i)
        path_j.append(j)

    return path_i, path_j


def plot_dtw_matrix_with_series(data_1, data_2, dtw_matrix, path_i, path_j):
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3,  # ahora hay 3 columnas: serie izquierda, heatmap, colorbar
                           width_ratios=[1, 5, 0.2], height_ratios=[4, 1],
                           wspace=0.05, hspace=0.05)

    # Serie izquierda (vertical)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.plot(data_1, np.arange(len(data_1)), color='mediumorchid')
    ax_left.invert_xaxis()
    ax_left.invert_yaxis()
    ax_left.set_title('2016-01-05', rotation=90, y = 0.5, x = 0)
    ax_left.axis('off')

    # Heatmap central (sin colorbar)
    ax_main = fig.add_subplot(gs[0, 1])
    heatmap = ax_main.imshow(dtw_matrix, aspect='auto', origin='upper', cmap="mako")
    ax_main.plot(path_j, path_i, color='red', linewidth=2)  # Traza la ruta de menor costo
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # Serie inferior (horizontal), perfectamente alineada con el heatmap
    ax_bottom = fig.add_subplot(gs[1, 1])
    ax_bottom.plot(np.arange(len(data_2)), data_2, color='turquoise')
    ax_bottom.set_xlim(ax_main.get_xlim())  # Alineación horizontal
    ax_bottom.invert_yaxis()
    ax_bottom.set_title('2018-10-31', y = -0.01)
    ax_bottom.axis('off')

    # Colorbar en su propio eje
    ax_cbar = fig.add_subplot(gs[0, 2])
    plt.colorbar(heatmap, cax=ax_cbar)

    plt.savefig("dtw_aligned_heatmap.png", bbox_inches='tight')
    plt.close()
    
def plot_aligned(data_1, data_2, path_i, path_j):
    fig, ax = plt.subplots(figsize=(22, 7))
    ax.plot(data_1, color='mediumorchid', lw = 2, label = '|B| para 2016-01-05')

    offset = (max(data_1) - min(data_2)) + 5  # ajustar separación
    data_2_offset = [y - offset for y in data_2]
    ax.plot(data_2_offset, color='turquoise', lw=2, label = '|B| para 2018-10-31')

    # Líneas de conexión
    path_i_cortado = path_i[::100]
    path_j_cortado = path_j[::100]
    for (i, j) in zip(path_i_cortado, path_j_cortado):
        ax.plot([i, j], [data_1[i], data_2_offset[j]], color='gray', lw=0.5)

    ax.legend(loc='upper left')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.title("Mapeo DTW", loc='center', y = 0.9)
    plt.savefig("dtw_aligned_series.png", bbox_inches='tight')
    plt.show()
    
dataB_1  = filtrarVentana('2016','01','05',1)['mod_B'].to_list()[4000::]
dataB_2 = filtrarVentana('2018','10','31',4)['mod_B'].to_list()[3000::]
plt.plot(dataB_1, label='2016', color='mediumorchid')
plt.plot(dataB_2, label='2018', color='turquoise')
plt.legend()
plt.savefig('temp.png', bbox_inches='tight')
plt.close()

M = distance_matrix(dataB_1, dataB_2)
costM = cost_matrix(M)
path_i, path_j = camino_de_menor_costo(costM)
plot_dtw_matrix_with_series(dataB_1, dataB_2, M, path_i, path_j)
plot_aligned(dataB_1, dataB_2, path_i, path_j)
