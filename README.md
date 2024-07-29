# Tesis

### 1. Crear un environment virtual

``` python -m venv venv
source venv/bin/activate
```

### 2. Instalar librerias
```
pip install -r requirements.txt
```

### 3. Configurar donde se guardan los datos de la descarga
Editar la linea 31 de DescargarDatosB.py para setear la variable datosCampoPath en el directorio en el está instalado el repo y después la carpeta "datos_campo_magnetico"

> Solucion: configurar en docker

### 4. Ejecutar la descarga de datos
```
python3 DescargaDatosB.py 2014-12-25
```
Se puede reemplazar por otro año-mes-día

### 5.a Configurar donde se leen los datos del acomodo de datos
Editar la linea 29 de AcomodarDatosB.py para configurar el parámetro de `pd.read_csv` al directorio que usaste en el paso 3 antes de donde dice "datos_campo_magnetico"

> Solucion: configurar en docker


### 6. Configurar la ruta de archivos en GraficarDatosB.py
Entrar a **DOS** lineas random (no una) y poner un coso hardcodeado

xd

> Solucion: configurar en docker

### 7. Graficar los datos
```
python3 GraficarDatosB.py 
```
