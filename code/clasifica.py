# classify_fase3_4.py
import pandas as pd, joblib, shutil
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

MODEL = Path("models/svm_model.joblib")
DATA = Path("data/features/features_dataset_fase3_labeled.csv")
RAW = Path("data/raw")
OUT = Path("output")
WEIGHTS = Path("weights/yolov8n.pt")

model = joblib.load(MODEL)
yolo = YOLO(str(WEIGHTS))

df = pd.read_csv(DATA)
X = df.drop(columns=["filename","label"]).values
svm_pred = model.predict(X)
svm_conf = model.predict_proba(X).max(axis=1)

final=[]
for fname, base in zip(df["filename"], svm_pred):
    img_path = next(RAW.rglob(fname))
    r = yolo.predict(str(img_path), classes=[0], verbose=False)
    persons = len(r[0].boxes)
    if persons > 2:
        final.append("fiestas")
    elif persons >= 1:
        final.append("personas")
    else:
        final.append(base)

df["svm_label"] = svm_pred
df["svm_conf"] = svm_conf
df["final_label"] = final

OUT.mkdir(exist_ok=True)
df.to_csv(OUT/"reporte_final.csv", index=False)

for _, r in df.iterrows():
    d = OUT / r["final_label"]
    d.mkdir(exist_ok=True)
    src = next(RAW.rglob(r["filename"]))
    shutil.copy2(src, d/r["filename"])

df["final_label"].value_counts().plot(kind="bar", title="Resumen por clase")
plt.tight_layout()
plt.savefig(OUT/"resumen.png")
print("Proceso terminado.")
