# Clasificador de imágenes

Proyecto de procesamiento digital de imágenes que combina características visuales clásicas, características extraídas con ResNet50, un clasificador SVM y detección de personas con YOLOv8. Las clases del conjunto de datos son `animales`, `comida`, `fiestas`, `paisajes` y `personas`.

## Estructura

- `code\utils_features.py`: preprocesamiento y extracción de color, forma, textura y características profundas con ResNet50.
- `code\procesos1y2.py`: recorre `datasetres\raw` y genera `datasetres\features\features_dataset_fase3_labeled.csv`.
- `code\entrena.py`: entrena y evalúa un pipeline StandardScaler + PCA + SVM y lo guarda como modelo joblib.
- `code\clasifica.py`: aplica el SVM y una regla basada en las detecciones de personas de YOLO; guarda un reporte y copia imágenes organizadas en `output`.
- `datasetres\raw\<clase>\`: imágenes de entrada, ordenadas por clase para generar el CSV de entrenamiento.
- `datasetres\testi\`: imágenes que el script de clasificación intenta localizar por nombre.
- `weights\yolov8n.pt`: pesos de YOLOv8n.
- `models\svm_pipeline.joblib`: pipeline SVM preentrenado (si se descarga correctamente con Git LFS).

## Requisitos

- Python instalado y disponible como `python` o `py` en PowerShell.
- Git LFS para descargar los CSV y el modelo SVM, que se almacenan mediante Git LFS.
- Conexión a Internet la primera vez que se carga ResNet50, para obtener sus pesos de ImageNet.

Desde la carpeta del repositorio, crea y activa un entorno virtual en PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install ultralytics matplotlib
```

Si PowerShell no permite activar el entorno, habilita la ejecución solo para la sesión actual y vuelve a activarlo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

En CMD, el comando de activación es:

```cmd
.venv\Scripts\activate.bat
```

Descarga los archivos grandes del repositorio desde su raíz:

```powershell
git lfs install
git lfs pull
```

Comprueba que los CSV y `models\svm_pipeline.joblib` sean archivos de datos reales y no archivos de texto que comiencen con `version https://git-lfs.github.com/spec/v1`.

## Ejecución

Los scripts usan rutas relativas que comienzan con `Proyecto-PDI\...`. Para que esas rutas funcionen, ejecuta los comandos desde la carpeta **que contiene** el proyecto y asegúrate de que la carpeta del proyecto se llame `Proyecto-PDI`. Por ejemplo, si el repositorio está en `C:\trabajo\Proyecto-PDI`, activa primero el entorno virtual desde la carpeta del proyecto y luego sitúate en su carpeta contenedora:

```powershell
Set-Location C:\trabajo\Proyecto-PDI
.\.venv\Scripts\Activate.ps1
Set-Location ..
```

### 1. Generar características

```powershell
python .\Proyecto-PDI\code\procesos1y2.py
```

Lee las imágenes de `datasetres\raw\<clase>\` y crea el CSV etiquetado. Las carpetas dentro de `raw` se usan como etiquetas. ResNet50 genera un vector de 2048 características por imagen, además de las características de color, forma y textura.

### 2. Entrenar el SVM

```powershell
python .\Proyecto-PDI\code\entrena.py --features .\Proyecto-PDI\datasetres\features\features_dataset_fase3_labeled.csv --out .\Proyecto-PDI\models\svm_pipeline.joblib
```

El entrenamiento usa una partición estratificada de 80/20 y una búsqueda de parámetros por validación cruzada. Para omitir la búsqueda y entrenar más rápido:

```powershell
python .\Proyecto-PDI\code\entrena.py --features .\Proyecto-PDI\datasetres\features\features_dataset_fase3_labeled.csv --out .\Proyecto-PDI\models\svm_pipeline.joblib --no-grid
```

### 3. Clasificar y organizar imágenes

```powershell
python .\Proyecto-PDI\code\clasifica.py
```

El script guarda `output\reporte_final.csv`, copia las imágenes en subcarpetas de `output` según la etiqueta final y genera `output\resumen.png`. YOLO solo detecta personas: la regla actual llama `fiestas` a una imagen con más de dos personas, `personas` si encuentra una o dos y, si no detecta ninguna, conserva la etiqueta del SVM.

## Limitaciones conocidas

- Las rutas de los scripts esperan una carpeta llamada `Proyecto-PDI` y deben ejecutarse desde la carpeta que la contiene; cambia las constantes de ruta en los scripts si usas otro nombre o ubicación.
- `clasifica.py` busca en `datasetres\testi` los nombres de archivo del CSV de características. Cada nombre del CSV debe existir en esa carpeta; si falta, el script no puede localizar la imagen. El conjunto incluido puede no tener todos los nombres coincidentes.
- `clasifica.py` vuelve a predecir sobre las filas del CSV de características. Si ese CSV también se usó para entrenar, sus resultados no son una evaluación independiente del modelo.
- La regla de YOLO basada en el número de personas es una heurística, no un detector de fiestas. No debe interpretarse como una clasificación fiable de escenas.
- El CSV empleado para evaluar el SVM debe provenir de datos de prueba separados de los usados en el entrenamiento para obtener una estimación imparcial del desempeño.
- `requirements.txt` no incluye `ultralytics` ni `matplotlib`, necesarios para `clasifica.py`; por eso se instalan por separado arriba.
- Si ResNet50 no puede cargarse, `utils_features.py` sustituye sus características por ceros. Revisa el mensaje de error y la conectividad si ocurre.

Al terminar, sal del entorno virtual con:

```powershell
deactivate
```
