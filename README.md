# Tesis

### 1. Instalar Docker 

### 2. Levantar el enviroment en Docker
```
./start_docker.sh
```
Para esto tiene que ser ejecutable el archivo. Si no lo es, puede que haga falta cambiar la configuración de permisos (**una única vez**):
```
chmod +x start_docker.sh
```



### 3. Ejecutar la descarga de datos
```
python3 DescargaDatosB.py 2014-12-25
```
Se puede reemplazar por otro año-mes-día o pueden colocarse dos fechas para descargar datos en ese periodo (YYYY_i-MM_i-DD_i YYYY_f-MM_f-DD_f). 

Esta linea descarga los datos en coordenadas Solar System (ss) de la pagina de MAVEN (https://lasp.colorado.edu/maven/sdc/public/data/sci/mag/l2/) correspondiente al dia o periodo indicado. Los cuales se bajan con el nombre de datos_{DD}-{MM}-{YYYY}.csv en la carpeta datos_campo_magnetico_crudos. Se descarga tambien la coordenada z en un sistema Planet Centered (pc) bajo el nombre z_{DD}-{MM}-{YYYY}.csv en la carpeta datos_campo_magnetico_crudos_pc, para posteriores analisis. 
 
### 4. Ejecutar pre-procesamiento de los datos
```
python3 OrbitasParaAutoencoder.py 2014-12-25 1
```
Se puede reemplazar por otro año-mes-día nº de orbita. 

Esta linea ejecuta una serie de codigos de modo de ir preprocesando los datos para la posterior ejecución del autoencoder. Lo primero que hace es utilizar AcomodarDatos.py para acomodar las columnas de los datos descargados. Luego, utilizando SepararHemisferios.py y SepararOrbitas.py se realiza una separación por orbitas y hemisferios seleccionando solo aquellos tramos donde el **satelite se encuentra en el lado de día y en el hemisferio norte**. Mediante el codigo PromedioPorVentanas se hace un Rolling Window para promediar los datos de campo magnetico de a 10. Luego se hace un corte de los datos al rededor del fit dado por Vignes et. al () utilizando CorteVignes.py. 
