# Tesis

### 1. Instalar Docker 

### 2. Levantar el enviroment en Docker
```
docker compose up -d
docker compose exec tesis /bin/bash
```

### 3. Ejecutar la descarga de datos
```
python3 DescargaDatosB.py 2014-12-25
```
Se puede reemplazar por otro año-mes-día

### 4. Ejecutar los filtros de los datos
```
python3 PromedioPorVentana.py 2014-12-25
python3 PasaBajos.py 2014-12-25
```
Se puede reemplazar por otro año-mes-día
