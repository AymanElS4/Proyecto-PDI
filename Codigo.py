import cv2
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
from typing import Optional
from skimage.feature import hog, local_binary_pattern
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet

# --- 1. CONFIGURACIÓN INICIAL ---
# Dimensiones estandarizadas para el preprocesamiento (Módulo 1)
TARGET_SIZE = (640, 640)
CATEGORIES = ['paisaje', 'retrato', 'evento', 'documento'] # Ejemplo de categorías

# Inicializar el modelo pre-entrenado para la extracción de Deep Features (Módulo 2)
# Usaremos MobileNetV2, ligero y rápido. Excluimos la capa de clasificación (include_top=False).
DEEP_FEAT_DIM = 1280  # Dimensión esperada para MobileNetV2 (pooling='avg')
try:
    BASE_MODEL = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    # Intentar ajustar la dimensión real del embedding si está disponible
    try:
        DEEP_FEAT_DIM = int(BASE_MODEL.output_shape[-1])
    except Exception:
        pass
    print("Modelo MobileNetV2 cargado para Deep Features. Dimensión:", DEEP_FEAT_DIM)
except Exception as e:
    print("ADVERTENCIA: No se pudo cargar MobileNetV2. Las Deep Features se reemplazarán por ceros. Motivo:", e)
    BASE_MODEL = None

# ----------------------------------------------------------------------
# --- FASE 1: IMPLEMENTACIÓN DEL MÓDULO 1 (Carga y Preprocesamiento) ---
# ----------------------------------------------------------------------

def preprocess_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Módulo 1: Carga y preprocesamiento de una imagen.
    - Lectura (validación de integridad).
    - Conversión a RGB (si no lo es).
    - Redimensionamiento (TARGET_SIZE).
    - Normalización (a rango 0-1).
    - Aplicación de filtro de suavizado (bilateral para mantener bordes).
    """
    try:
        # 1. Lectura de la imagen (cv2.IMREAD_COLOR para 3 canales)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Error: No se pudo leer la imagen {image_path.name}. Saltando.")
            return None
        
        # 2. Conversión a RGB (OpenCV lee en BGR por defecto)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Redimensionamiento
        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # 4. Normalización de valores de intensidad (a rango 0-1)
        img_normalized = img_resized.astype("float32") / 255.0
        
        # 5. Aplicación de filtro de suavizado (Filtro Bilateral para reducción de ruido)
        # Los parámetros (d, sigmaColor, sigmaSpace) deben ajustarse si es necesario.
        img_denoised = cv2.bilateralFilter(
            (img_normalized * 255).astype(np.uint8),  # Bilateral requiere uint8
            d=9, sigmaColor=75, sigmaSpace=75
        )
        img_denoised = img_denoised.astype("float32") / 255.0 # Volver a normalizar
        
        return img_denoised

    except Exception as e:
        print(f"Error procesando {image_path.name}: {e}")
        return None

# ----------------------------------------------------------------------
# --- FASE 2: IMPLEMENTACIÓN DEL MÓDULO 2 (Extracción de Características) ---
# ----------------------------------------------------------------------

def extract_color_features(img: np.ndarray) -> np.ndarray:
    """
    Módulo 2.1: Extracción de características de Color (Histogramas RGB y HSV).
    """
    # Asegurarse de trabajar sobre uint8 en rango 0-255 para OpenCV
    uint8_img = (img * 255).astype(np.uint8)

    # 1. Histograma RGB (concatenado para los 3 canales)
    hist_rgb = []
    for i in range(3):  # 0: R, 1: G, 2: B
        hist = cv2.calcHist([uint8_img], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        hist_rgb.extend(hist.flatten())

    # Convertir a HSV para histograma complementario de Color
    img_hsv = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV)
    hist_hsv = []
    # Solo se extraen H (Hue/Tono) y S (Saturation/Saturación) para el color.
    for i in range(2): # 0: H, 1: S (V/Valor se omite a menudo)
        if i == 0:
            hist = cv2.calcHist([img_hsv], [i], None, [180], [0, 180])
        else:
            hist = cv2.calcHist([img_hsv], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        hist_hsv.extend(hist.flatten())

    return np.array(hist_rgb + hist_hsv)

def extract_edge_and_shape_features(img: np.ndarray) -> np.ndarray:
    """
    Módulo 2.2: Extracción de características de Bordes y Forma (Canny y HOG).
    """
    # Trabajar sobre uint8
    uint8_img = (img * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

    # 1. Detección de Bordes Canny (Medida de la densidad/complejidad de bordes)
    edges = cv2.Canny(img_gray, 100, 200)
    canny_feature = np.sum(edges) / (TARGET_SIZE[0] * TARGET_SIZE[1]) # Densidad de bordes

    # 2. Histograma de Gradientes Orientados (HOG)
    # Evitar parámetros que dependen de versiones de skimage
    hog_features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        visualize=False
    )

    return np.concatenate(([canny_feature], hog_features))

def extract_texture_features(img: np.ndarray) -> np.ndarray:
    """
    Módulo 2.3: Extracción de características de Textura (LBP).
    """
    # Trabajar sobre uint8 y escala de grises
    uint8_img = (img * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

    # Local Binary Pattern (LBP)
    # P: número de puntos vecinos (8), R: radio (1)
    P = 8
    lbp = local_binary_pattern(img_gray, P=P, R=1, method='uniform')

    # Calcular el histograma de LBP (uniform LBP produce P+2 posibles valores)
    bins = P + 2
    (hist, _) = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))

    # Normalizar el histograma
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist

def extract_deep_features(img: np.ndarray) -> np.ndarray:
    """
    Módulo 2.4: Extracción de Features Profundas (Embeddings de CNN).
    NOTA: Esto requiere el modelo MobileNetV2 cargado.
    """
    # Siempre devolver un vector de longitud fija: DEEP_FEAT_DIM
    if BASE_MODEL is None:
        return np.zeros(DEEP_FEAT_DIM, dtype=np.float32)

    try:
        # La imagen ya está en TARGET_SIZE y float32.
        # Expandir la dimensión para el batch (1, H, W, 3)
        img_batch = np.expand_dims(img * 255.0, axis=0).astype(np.float32) # Keras espera 0-255

        # Preprocesamiento específico del modelo (ej. centrado, escalado)
        img_processed = preprocess_mobilenet(img_batch)

        # Extracción del embedding
        features = BASE_MODEL.predict(img_processed, verbose=0)
        feats = features.flatten()

        # Ajustar tamaño (padding o recorte) para garantizar longitud fija
        if feats.size < DEEP_FEAT_DIM:
            feats = np.pad(feats, (0, DEEP_FEAT_DIM - feats.size), mode='constant')
        elif feats.size > DEEP_FEAT_DIM:
            feats = feats[:DEEP_FEAT_DIM]

        return feats
    except Exception as e:
        print(f"Error extrayendo Deep Features: {e}")
        return np.zeros(DEEP_FEAT_DIM, dtype=np.float32)


# --- FUNCIÓN PRINCIPAL DE EJECUCIÓN ---


def run_sprints(input_dir: str, output_dir: str):
    """
    Ejecuta las fases 1 y 2 del proyecto.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Crear carpetas de salida si no existen
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Buscar imágenes con extensiones comunes (recursivo)
    image_paths = [p for p in input_path.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not image_paths:
        print(f"Error: No se encontraron imágenes en la carpeta de entrada: {input_dir}")
        return

    all_features = []
    
    # --- FASE 1 Y 2 ITERACIÓN ---
    for i, image_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Procesando: {image_path.name}")
        
        # === FASE 1: PREPROCESAMIENTO ===
        processed_img = preprocess_image(image_path)
        
        if processed_img is None:
            continue
            
        # Opcional: Guardar la imagen preprocesada (Entregable Módulo 1)
        # Asegurarse de convertir de float32 a uint8 y de RGB a BGR para cv2.imwrite
        # output_filename = output_path / f"pre_{image_path.stem}.png"
        # cv2.imwrite(str(output_filename), cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # === FASE 2: EXTRACCIÓN DE CARACTERÍSTICAS ===
        
        # 1. Color
        color_feats = extract_color_features(processed_img)
        # 2. Borde/Forma
        edge_shape_feats = extract_edge_and_shape_features(processed_img)
        # 3. Textura
        texture_feats = extract_texture_features(processed_img)
        # 4. Profundas (si el modelo cargó o rellenadas con ceros)
        deep_feats = extract_deep_features(processed_img)

        # Concatenar todos los vectores de características
        # Se asegura que si deep_feats es None o vacío, la concatenación sigue funcionando
        all_vector = np.concatenate([
            color_feats,
            edge_shape_feats,
            texture_feats,
            deep_feats
        ])

        # Crear un diccionario para la fila del DataFrame
        feature_dict = {
            'filename': image_path.name,
            'path': str(image_path),
            'feature_vector': all_vector
        }
        all_features.append(feature_dict)
        
    # --- FIN DE FASE 2: GENERACIÓN DEL ENTREGABLE ---
    
    # Creamos un DataFrame para estructurar los resultados (Entregable Módulo 2)
    df_features = pd.DataFrame(all_features)
    
    # Asegurarse que todos los vectores tengan la misma longitud (padding si es necesario)
    vectors = list(df_features['feature_vector'].values)
    lengths = [v.size for v in vectors]
    max_len = max(lengths) if lengths else 0
    normalized = [np.pad(v, (0, max_len - v.size), mode='constant') if v.size < max_len else v[:max_len] for v in vectors]
    feature_matrix = np.vstack(normalized)
    feature_names = [f'feat_{i+1}' for i in range(feature_matrix.shape[1])]
    
    df_final = pd.DataFrame(feature_matrix, columns=feature_names)
    df_final.insert(0, 'filename', df_features['filename'])
    
    # Guardar el DataFrame
    output_csv = output_path / "extracted_features.csv"
    df_final.to_csv(output_csv, index=False)
    
    print("\n--- FASES 1 Y 2 COMPLETADAS ---")
    print(f"Imágenes procesadas: {len(df_final)}")
    print(f"Dimensión del vector de características (por imagen): {feature_matrix.shape[1]}")
    print(f"ENTREGABLE FASE 2 guardado en: {output_csv}")


if __name__ == '__main__':
    # Configuración de los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Script para Preprocesamiento y Extracción de Características de Imágenes (Fases 1 y 2 del Proyecto).")
    parser.add_argument('--input', type=str, required=True, help="Ruta a la carpeta que contiene las imágenes de entrada.")
    parser.add_argument('--output', type=str, default='./output_features', help="Ruta a la carpeta donde se guardarán las características extraídas.")
    
    args = parser.parse_args()
    
    # Ejemplo de uso: python nombre_del_script.py --input C:/Users/Usuario/Fotos --output C:/Proyecto/Data
    run_sprints(args.input, args.output)