from flask import Flask, jsonify, send_from_directory
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(str(PROJECT_ROOT), "index.html")


@app.get("/api/status")
def status():
    model_path = PROJECT_ROOT / "Proyecto-PDI-main" / "models" / "svm_pipeline.joblib"
    features_path = (
        PROJECT_ROOT
        / "Proyecto-PDI-main"
        / "datasetres"
        / "features"
        / "features_dataset_fase3_labeled.csv"
    )

    def is_git_lfs_pointer(path: Path) -> bool:
        try:
            return path.read_text(encoding="utf-8", errors="ignore").startswith(
                "version https://git-lfs.github.com/spec/v1"
            )
        except OSError:
            return True

    model_ready = model_path.is_file() and not is_git_lfs_pointer(model_path)
    features_ready = features_path.is_file() and not is_git_lfs_pointer(features_path)

    return jsonify(
        {
            "status": "ready",
            "model_ready": model_ready,
            "features_ready": features_ready,
            "message": (
                "Aplicación web desplegable. Ejecuta Git LFS para habilitar "
                "la clasificación completa con SVM y YOLO."
                if not model_ready or not features_ready
                else "Modelo y características disponibles."
            ),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
