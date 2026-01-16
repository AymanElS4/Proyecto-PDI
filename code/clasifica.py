# classify_and_organize.py
import pandas as pd, joblib, shutil
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

model=joblib.load("models/svm_model.joblib")
yolo=YOLO("weights/yolov8n.pt")

df=pd.read_csv("data/features/features_dataset_fase3.csv")
X=df.drop(columns=["filename"]).values
svm_pred=model.predict(X)
svm_conf=model.predict_proba(X).max(axis=1)

final=[]
for fname,base in zip(df["filename"],svm_pred):
    r=yolo.predict(f"data/raw/{fname}",classes=[0],verbose=False)
    persons=len(r[0].boxes)
    if persons>2: final.append("fiestas")
    elif persons>=1: final.append("personas")
    else: final.append(base)

df["svm_label"]=svm_pred
df["svm_conf"]=svm_conf
df["final_label"]=final
df.to_csv("output/reporte_final.csv",index=False)

out=Path("output"); out.mkdir(exist_ok=True)
for _,r in df.iterrows():
    d=out/r["final_label"]; d.mkdir(exist_ok=True)
    shutil.copy2(f"data/raw/{r['filename']}", d/r["filename"])

df["final_label"].value_counts().plot(kind="bar",title="Resumen por clase")
plt.savefig("output/resumen.png")
print("Clasificación y organización terminadas.")
