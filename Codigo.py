import cv2
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
from typing import Optional

# Librerías para características de textura y forma
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

# Librerías de Deep Learning Cambiado a ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet

# --- NUEVAS LIBRERÍAS FASE 3 ---
from ultralytics import YOLO  # Para YOLOv8
from sklearn.svm import SVC   # Sugerido en Avance #2 para clasificación final
import joblib

# CONFIGURACIÓN INICIAL

# Según Avance #2 - Módulo 1: "redimensionamiento se planean 640x640 píxeles"
TARGET_SIZE = (640, 640) 

# Inicializar modelo pre-entrenado (Módulo 2)
# "Se obtendrán características claves utilizando ResNet"
DEEP_FEAT_DIM = 2048  # ResNet50 con pooling='avg' devuelve un vector de 2048
try:
    # include_top=False elimina la capa de clasificación final, nos quedamos con los "features"
    BASE_MODEL = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    print(f"Modelo ResNet50 cargado. Dimensión del vector: {DEEP_FEAT_DIM}")
except Exception as e:
    print(f"ADVERTENCIA: No se pudo cargar ResNet50. Error: {e}")
    BASE_MODEL = None

# ----------------------------------------------------------------------
# --- FASE 1: IMPLEMENTACIÓN DEL MÓDULO 1 (Carga y Preprocesamiento) ---
# ----------------------------------------------------------------------

def preprocess_image(image_path: Path) -> Optional[np.ndarray]:
    
    #Cumple con Módulo 1:
    # Lectura y verificación[cite: 16].
    # Conversión a RGB y Redimensionamiento (640x640)[cite: 17].
    # Filtro de suavizado (Bilateral)[cite: 18]
    
    try:
        # 1. Lectura
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        
        # 2. Estandarización: RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Redimensionamiento a 640x640
        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # 4. Normalización (0-1)
        img_normalized = img_resized.astype("float32") / 255.0
        
        # 5. Filtro de suavizado (Bilateral para reducir ruido preservando bordes)
        # Se convierte temporalmente a uint8 porque bilateralFilter lo requiere/funciona mejor
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        img_denoised = cv2.bilateralFilter(img_uint8, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Retornar normalizado float32
        return img_denoised.astype("float32") / 255.0

    except Exception as e:
        print(f"Error procesando {image_path.name}: {e}")
        return None

# ----------------------------------------------------------------------
# --- FASE 2: IMPLEMENTACIÓN DEL MÓDULO 2 (Extracción de Características) ---
# ----------------------------------------------------------------------

def extract_color_features(img: np.ndarray) -> np.ndarray:

    #Módulo 2: "Se calcularán histogramas de color en los espacios RGB y HSV".
    
    uint8_img = (img * 255).astype(np.uint8)

    # Histograma RGB
    hist_rgb = []
    for i in range(3):
        hist = cv2.calcHist([uint8_img], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        hist_rgb.extend(hist.flatten())

    # Histograma HSV
    img_hsv = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV)
    hist_hsv = []
    # Usamos Hue (Tono) y Saturation (Saturación)
    for i in range(2): 
        bins = 180 if i == 0 else 256
        range_vals = [0, 180] if i == 0 else [0, 256]
        hist = cv2.calcHist([img_hsv], [i], None, [bins], range_vals)
        cv2.normalize(hist, hist)
        hist_hsv.extend(hist.flatten())

    return np.array(hist_rgb + hist_hsv)

def extract_edge_and_shape_features(img: np.ndarray) -> np.ndarray:

    #Módulo 2: "se extraerán bordes mediante el operador de Canny y descriptores de forma con HOG".
    
    uint8_img = (img * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

    # Canny (Densidad de bordes)
    edges = cv2.Canny(img_gray, 100, 200)
    canny_feature = np.sum(edges) / (TARGET_SIZE[0] * TARGET_SIZE[1])

    # HOG (Histogram of Oriented Gradients)
    hog_features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(32, 32), # Ajustado para 640x640 para no generar un vector gigante
        cells_per_block=(2, 2),
        transform_sqrt=True,
        visualize=False
    )

    return np.concatenate(([canny_feature], hog_features))

def extract_texture_features(img: np.ndarray) -> np.ndarray:
    
    #Módulo 2: "se caracterizarán las texturas a través de LBP y GLCM".
    
    uint8_img = (img * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

    # 1. LBP (Local Binary Patterns)
    P, R = 8, 1
    lbp = local_binary_pattern(img_gray, P=P, R=R, method='uniform')
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=P+2, range=(0, P+2))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)

    # 2. GLCM (Gray Level Co-occurrence Matrix) - AGREGADO NUEVO
    # Calculamos contraste y energía como descriptores clave
    # 'levels' debe coincidir con el rango de la imagen (0-255)
    glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    glcm_features = np.array([contrast, energy, homogeneity, correlation])

    return np.concatenate([hist_lbp, glcm_features])

def extract_deep_features(img: np.ndarray) -> np.ndarray:
    
    #Módulo 2: "Se obtendrán características claves utilizando redes como ResNet".
    
    if BASE_MODEL is None:
        return np.zeros(DEEP_FEAT_DIM, dtype=np.float32)

    try:
        # Preparamos la imagen para Keras (batch size 1)
        img_batch = np.expand_dims(img * 255.0, axis=0) # ResNet espera valores tipo imagen sin normalizar a 0-1 previo al preprocess
        
        # Preprocesamiento específico de ResNet (Zero-center, etc.)
        img_processed = preprocess_resnet(img_batch)

        # Predicción (Extracción del vector)
        features = BASE_MODEL.predict(img_processed, verbose=0)
        return features.flatten() # Debe ser longitud 2048
        
    except Exception as e:
        print(f"Error extrayendo Deep Features: {e}")
        return np.zeros(DEEP_FEAT_DIM, dtype=np.float32)


# --- YOLO ---
try:
    YOLO_MODEL = YOLO("yolov8n.pt")
    print("YOLOv8 cargado correctamente.")
except Exception as e:
    print("No se pudo cargar YOLO:", e)
    YOLO_MODEL = None

def extract_yolo_person_features(img_path: Path) -> np.ndarray:
    """
    Usa YOLOv8 para contar personas (Clase 0 en COCO).
    Retorna un pequeño vector con el conteo y la confianza máxima.
    """
    if YOLO_MODEL is None: return np.array([0, 0])
    
    # classes=[0] filtra para que YOLO solo busque personas
    results = YOLO_MODEL.predict(str(img_path), classes=[0], verbose=False)
    
    num_persons = len(results[0].boxes)
    max_conf = 0
    if num_persons > 0:
        max_conf = results[0].boxes.conf.cpu().numpy().max()
    
    # Retornamos el conteo y la confianza como características
    return np.array([num_persons, max_conf])

# --- FUNCIÓN PARA DECISIÓN SEMÁNTICA (Módulo 3.3) ---
def classify_scene_logic(prediction_label, v_yolo):
    """
    Aplica las reglas de decisión combinadas mencionadas en tu Avance #2.
    """
    num_persons = v_yolo[0]
    
    # Regla: "Si se detectan muchos rostros/personas, es Retrato o Fiesta"
    if num_persons >= 1 and num_persons <= 2:
        return "Personas/Retrato"
    elif num_persons > 2:
        return "Fiestas"
    
    return prediction_label # Si no hay personas, confía en el clasificador (Paisajes, Comida, etc.)

# ----------------------------------------------------------------------
# --- ORQUESTADOR ACTUALIZADO ---
# ----------------------------------------------------------------------

def run_pipeline(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [p for p in input_path.rglob('*') if p.suffix.lower() in valid_extensions]
    
    all_data = []
    print(f"Iniciando Fase 3 para {len(image_paths)} imágenes...")

    for img_path in image_paths:
        processed_img = preprocess_image(img_path)
        if processed_img is None: continue

        # 1. Características PDI Clásicas (Fase 2)
        f_color = extract_color_features(processed_img)
        f_edge = extract_edge_and_shape_features(processed_img)
        f_texture = extract_texture_features(processed_img)
        
        # 2. Características Deep (ResNet50)
        f_deep = extract_deep_features(processed_img)

        # 3. Características de Objetos (YOLOv8 - SOLO PERSONAS)
        f_yolo = extract_yolo_person_features(img_path)

        # FUSIÓN DE VECTORES (El "Vector Maestro" para el SVM)
        full_vector = np.concatenate([f_color, f_edge, f_texture, f_deep, f_yolo])
        all_data.append(full_vector)
        print(f"Vector generado para {img_path.name}. Personas detectadas: {int(f_yolo[0])}")

    # Guardar Dataset de Características
    if all_data:
        df = pd.DataFrame(all_data)
        df.insert(0, 'filename', [p.name for p in image_paths])
        df.to_csv(output_path / "features_dataset_fase3.csv", index=False)
        print(f"\nProceso completado. Vector de dimensión: {df.shape[1]-1}")
        print("Siguiente paso: Entrenar el clasificador con este CSV.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='./output')
    args = parser.parse_args()
    run_pipeline(args.input, args.output)

# --- FUNCIÓN PRINCIPAL DE EJECUCIÓN (ORQUESTADOR) ---
"""
def run_pipeline(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extensiones válidas mencionadas en PDF (JPG, PNG, BMP) [cite: 39]
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [p for p in input_path.rglob('*') if p.suffix.lower() in valid_extensions]
    
    if not image_paths:
        print(f"No se encontraron imágenes en {input_dir}")
        return

    all_data = []

    print(f"Iniciando procesamiento de {len(image_paths)} imágenes...")
    
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Procesando: {img_path.name}")
        
        # --- FASE 1: Preprocesamiento ---
        processed_img = preprocess_image(img_path)
        if processed_img is None: continue

        # --- FASE 2: Extracción ---
        # 1. Vectores clásicos
        feat_color = extract_color_features(processed_img)
        feat_edge = extract_edge_and_shape_features(processed_img)
        feat_texture = extract_texture_features(processed_img) # Ahora incluye GLCM
        
        # 2. Vectores profundos (Deep Learning)
        feat_deep = extract_deep_features(processed_img)

        # Concatenación final
        full_vector = np.concatenate([feat_color, feat_edge, feat_texture, feat_deep])
        
        all_data.append(full_vector)
    
    # Guardar resultados
    if all_data:
        # Crear DataFrame
        df = pd.DataFrame(all_data)
        # Añadir nombres de archivo como primera columna
        df.insert(0, 'filename', [p.name for p in image_paths])
        
        csv_path = output_path / "features_dataset.csv"
        df.to_csv(csv_path, index=False)
        print("\n--- PROCESO FINALIZADO ---")
        print(f"Archivo de características generado en: {csv_path}")
        print(f"Dimensiones del vector por imagen: {df.shape[1] - 1}")
    else:
        print("No se generaron datos.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PDI Proyecto: Fases 1 y 2")
    parser.add_argument('--input', type=str, required=True, help="Carpeta de imágenes")
    parser.add_argument('--output', type=str, default='./output', help="Carpeta de salida")
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output)
"""
