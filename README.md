# Tesis

## Para la ejecución del buscador de MPB

### 1. Instalar Docker 

### 2. Levantar el enviroment en Docker
```
./start_docker.sh
```
Para esto tiene que ser ejecutable el archivo. Si no lo es, puede que haga falta cambiar la configuración de permisos (**una única vez**):
```
chmod +x start_docker.sh
```

### 3. Ejecutar la descarga de datos cuya MPB se quiere identificar
```
python3 PreprocesamientoDatos/DescargaDatosB.py 2014-12-25
```
Se puede reemplazar por otro año-mes-día o pueden colocarse dos fechas para descargar datos en ese periodo (YYYY_i-MM_i-DD_i YYYY_f-MM_f-DD_f). 

Esta linea descarga los datos en coordenadas Solar System (ss) de la pagina de MAVEN (https://lasp.colorado.edu/maven/sdc/public/data/sci/mag/l2/) correspondiente al dia o periodo indicado. Los cuales se bajan con el nombre de datos_{DD}-{MM}-{YYYY}.csv en la carpeta datos_campo_magnetico_crudos. Se descarga tambien la coordenada z en un sistema Planet Centered (pc) bajo el nombre z_{DD}-{MM}-{YYYY}.csv en la carpeta datos_campo_magnetico_crudos_pc, para posteriores analisis. 

### 4. Hacer un archivo fechas.txt con las fechas a analizar

Debe estar en el formato:
```
YYYY_1, MM_1, DD_1 \n
YYYY_2, MM_2, DD_2 \n
       ...
YYYY_n, MM_n, DD_n
```
 
### 5. Ejecutar el algoritmo de KNN para las fechas seleccionadas
```
python3 KNN/IdentificaciónDeMPB.py fechas.txt
```
Este comando ejecuta todo el preprocesamiento de datos adémas del KNN. El preprocesamiento consiste en acomodar los datos crudos descargados de la página de MAVEN, eliminando las filas cuyo contenido son especificaciones tecnicas de MAVEN. Se pasan las tres columnas de dato temporal a una sola correspondiente a la hora con unidades decimales. Este paso puede ejecutarse de manera aislada (ver sección siguiente) utilizando **AcomodarDatosB.py**.

El siguiente paso que se realiza para la determinación de las zonas con MPB es separar las orbitas, distinguiendo así cuándo el satélite se encuentra en el lado de día y descartando las orbitas correspondientes al lado de noche. Tambien se separan los hemisferios sur y norte, dejando solo aquellos datos correspondientes al hemisferio norte para poder despreciar los aportes del campo magnetico cortical. Es decir que, solo quedan aquellos tramos donde el **satelite se encuentra en el lado de día y en el hemisferio norte**. Ambos pasos pueden ejecutarse de manera aislada con los codigos **SepararOrbitas.py** y **SepararHemisferios.py**.

Mediante el codigo **PromedioPorVentanas.py** se hace un rolling window para promediar los datos de campo magnetico de a 10 y disminuir así su ruido. 
Como último paso del preprocesamiento, se hace un corte de los datos al rededor del fit dado por Vignes et. al () utilizando **CorteVignes.py**. 
