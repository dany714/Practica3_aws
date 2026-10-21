# Práctica 3: Limpieza de datos

Tratamiento de un dataset de 10 registros con valores faltantes y anómalos.
Utiliza IQR para detectar ingresos atípicos y KNNImputer para completar los datos.
Compara k=2, k=3 y k=5 con imputación por media y mediana.

## Ejecución

En PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe practica3.py todo
```

Las etapas se pueden ejecutar por separado con `inicial`, `iqr`, `knn`,
`comparar` y `graficos`.

## Resultados

La carpeta `salidas/` contiene los datasets original, preparado e imputados,
la comparación de métodos, los límites IQR y el boxplot antes y después.
Las edades mayores de 100 y los ingresos fuera de los límites IQR se sustituyen
por NaN antes de imputar. Todos los métodos conservan los valores válidos.

KNN usa pesos uniformes y las escalas originales del ejemplo. Cambiar k modifica
el vecindario y puede cambiar las estimaciones. Las edades imputadas se mantienen
decimales para mostrar el resultado del algoritmo.
