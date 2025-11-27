Proyecto-PDI
================

Este repositorio contiene un script principal (`Codigo.py`) para preprocesamiento y extracción de características de imágenes. El objetivo del proyecto es construir un clasificador/organizador de imágenes por tipo de escena.

Requisitos
---------

Instala las dependencias (recomendado en un entorno virtual):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Uso
---

Extraer características de una carpeta de imágenes:

```powershell
python .\Proyecto-PDI\Codigo.py --input "C:\ruta\a\imagenes" --output ".\output_features"
```

Notas
----
- `Codigo.py` realiza preprocesamiento (resizing, denoise) y extrae features de color, forma (Canny+HOG), textura (LBP) y deep features (MobileNetV2 si está disponible).
- Si MobileNetV2 no está disponible, las deep features se rellenan con ceros para mantener longitud fija en el vector de características.

Próximos pasos sugeridos
-----------------------
- Añadir un módulo para entrenar un clasificador (p. ej. `sklearn.RandomForestClassifier`) con un dataset etiquetado.
- Añadir un script de organización que mueva las imágenes predichas a subcarpetas por etiqueta.
