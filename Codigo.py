import os
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# --------------------------
# CONFIGURACIÓN GENERAL
# --------------------------
TARGET_SIZE = (640, 640)   # Tamaño uniforme
ALLOWED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")

def load_image(path):
    """Carga una imagen usando Pillow y la convierte a RGB."""
    try:
        img = Image.open(path)
        img.verify()  # Verifica integridad
        img = Image.open(path)  # Reabrir para procesar
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def preprocess_image(pil_img):
    """Estandariza, convierte, normaliza y suaviza una imagen."""
    # PIL → NumPy (OpenCV usa NumPy)
    img = np.array(pil_img)

    # RGB → BGR (OpenCV usa BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Redimensionar
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Suavizado (GAUSSIANO)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Normalización 0–1
    img = img.astype("float32") / 255.0

    return img


def save_image(img, output_path):
    """Guarda imagen normalizada en formato JPG."""
    # Convertir de 0–1 a 0–255
    img_uint8 = (img * 255).astype("uint8")
    cv2.imwrite(output_path, img_uint8)


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total = 0
    processed = 0
    corrupted = 0

    for filename in os.listdir(input_folder):
        total += 1
        ext = os.path.splitext(filename)[1].lower()

        # Verificar formato
        if ext not in ALLOWED_FORMATS:
            continue

        full_path = os.path.join(input_folder, filename)

        # Carga e integridad
        img = load_image(full_path)
        if img is None:
            print(f"[CORRUPTA] {filename} descartada.")
            corrupted += 1
            continue

        # Procesamiento
        processed_img = preprocess_image(img)

        # Guardado
        output_path = os.path.join(output_folder, filename)
        save_image(processed_img, output_path)

        processed += 1
        print(f"[OK] {filename} procesada.")

    # Reporte (opcional)
    print("\n========= REPORTE DEL MÓDULO 1 =========")
    print(f"Total de imágenes:     {total}")
    print(f"Procesadas correctamente: {processed}")
    print(f"Corruptas/invalidas:    {corrupted}")
    print(f"Guardadas en: {output_folder}")
    print("==========================================")


if __name__ == "__main__":
    INPUT = "input_images"
    OUTPUT = "output_images"

    process_folder(INPUT, OUTPUT)