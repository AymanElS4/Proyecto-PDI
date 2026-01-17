import pandas as pd
from pathlib import Path
import numpy as np
from utils_features import (
    preprocess_image, extract_color, extract_shape,
    extract_texture, extract_deep
)

DATA_DIR = Path("Proyecto-PDI/datasetres/raw")
OUT_CSV = Path("Proyecto-PDI/datasetres/features/features_dataset_fase3_labeled.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def run():
    rows=[]
    for cls_dir in DATA_DIR.iterdir():
        if not cls_dir.is_dir(): continue
        label = cls_dir.name.lower()
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() not in [".jpg",".jpeg",".png",".bmp"]:
                continue
            img = preprocess_image(img_path)
            if img is None: continue
            feats = np.concatenate([
                extract_color(img),
                extract_shape(img),
                extract_texture(img),
                extract_deep(img)
            ])
            rows.append([img_path.name] + feats.tolist() + [label]) #ya se verifico que si está al final

    if not rows:
        print("No se encontraron imágenes.")
        return

    n_feats = len(rows[0]) - 2
    cols = ["filename"] + [f"f{i}" for i in range(n_feats)] + ["label"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(OUT_CSV, index=False)
    print("CSV creado:", OUT_CSV)

if __name__=="__main__":
    run()
