import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature # Import tambahan
import os
import warnings

# Filter warning agar output lebih bersih
warnings.filterwarnings("ignore")

def load_data(path):
    """Memuat data bersih."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File data tidak ditemukan di: {path}")
    return pd.read_csv(path)

def main():
    print("Memulai run 'Basic' (Dataset: Heart Disease)...")

    # 1. Autolog
    # Kita nonaktifkan log_models di autolog agar tidak bentrok dengan manual log kita
    mlflow.sklearn.autolog(log_models=False)

    # 2. Load Data
    data_path = '.././namadataset_preprocessing/heart_processed.csv'
    
    try:
        df = load_data(data_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 3. Split Data
    target_col = 'target'
    if target_col not in df.columns:
        print(f"Error: Kolom target '{target_col}' tidak ditemukan.")
        return

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. MLflow Run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # 5. Train Model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # 6. Evaluasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Akurasi: {acc}")
        
        # Log metrik manual (jaga-jaga jika autolog skip)
        mlflow.log_metric("accuracy", acc)

        # 7. Simpan Model (FIX WARNINGS)
        print("Menyimpan model ke MLflow...")
        
        # Buat Signature (Deskripsi Input/Output) agar tidak muncul warning signature
        signature = infer_signature(X_train, preds)
        
        # Buat Input Example (Contoh data)
        input_example = X_train.iloc[[0]]

        # Log Model dengan parameter lengkap
        # Menggunakan keyword argument 'artifact_path' untuk menghindari warning deprecated
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

    print("\nRun selesai!")
    print(f"Silakan cek Artifacts di MLflow UI untuk Run ID: {run_id}")

if __name__ == "__main__":
    main()