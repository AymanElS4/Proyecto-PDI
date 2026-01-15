# train_phase3.py
"""
Entrena un clasificador (SVM) usando el CSV de features etiquetado.
Salida: svm_pipeline.joblib
Uso: python train_phase3.py --features ./output/features_dataset_fase3_labeled.csv --out ./models/svm_pipeline.joblib
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_features_csv(path: Path):
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        raise ValueError("El CSV debe tener una columna 'label' con las etiquetas.")
    X = df.drop(columns=['filename', 'label'], errors='ignore').values
    y = df['label'].values
    return X, y

def build_and_train(X, y, out_path: Path, do_grid=True):
    # Pipeline: scaler -> PCA (retener 0.98 varianza) -> SVC(probability)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.98, svd_solver='full')),
        ('svc', SVC(kernel='rbf', probability=True))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if do_grid:
        param_grid = {
            'svc__C': [0.1, 1, 5],
            'svc__gamma': ['scale', 0.01, 0.001]
        }
        gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        print("Mejores params:", gs.best_params_)
        model = best
    else:
        pipe.fit(X_train, y_train)
        model = pipe

    ypred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, ypred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, ypred))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Modelo guardado en: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='CSV con features y columna label')
    parser.add_argument('--out', default='./models/svm_pipeline.joblib', help='Ruta de salida del modelo')
    parser.add_argument('--no-grid', action='store_true', help='No usar GridSearchCV')
    args = parser.parse_args()

    X, y = load_features_csv(Path(args.features))
    build_and_train(X, y, Path(args.out), do_grid=not args.no_grid)

if __name__ == '__main__':
    main()
