import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import warnings
import os

# --- PENTING: JANGAN SET TRACKING URI DI SINI ---
# 'mlflow.set_tracking_uri' dihapus agar workflow CI
# dapat menggunakan URI dari environment variables (GitHub Secrets)

# Abaikan peringatan agar output bersih
warnings.filterwarnings('ignore')

# Path data (relative to the root of the Workflow-CI repository)
DATA_PATH = 'namadataset_preprocessing/wine_processed.csv'

def load_data(path):
    """Memuat data bersih."""
    print(f"Memuat data dari: {path}")
    if not os.path.exists(path):
        print(f"Error: File data tidak ditemukan di {path}")
        print("Pastikan file 'wine_processed.csv' ada di folder 'namadataset_preprocessing'.")
        return None
    return pd.read_csv(path)

def main():
    print("Memulai run 'CI Workflow' dengan manual log dan tuning...")
    
    # 1. Memuat dataset
    df = load_data(DATA_PATH)
    if df is None:
        return # Hentikan eksekusi jika data tidak ditemukan

    # 2. Memisahkan fitur (X) dan target (y)
    X = df.drop('quality_category', axis=1)
    y = df['quality_category']
    
    # 3. Membagi data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Memulai MLflow run
    # MLflow akan otomatis mengambil URI dari environment variables
    # (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # 5. Menerapkan Hyperparameter Tuning
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
        
        base_model = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(estimator=base_model, 
                                   param_grid=param_grid, 
                                   cv=5, 
                                   scoring='accuracy', 
                                   n_jobs=-1)
        
        print("Menjalankan GridSearchCV...")
        grid_search.fit(X_train, y_train)
        
        # Dapatkan model dan parameter terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best Parameters: {best_params}")
        
        # 6. Log Parameter Secara Manual
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("tuning_cv_folds", 5)
        
        # 7. Evaluasi model terbaik
        print("Mengevaluasi model terbaik...")
        preds = best_model.predict(X_test)
        
        # Hitung metrik (termasuk metrik tambahan dari Kriteria 2 Advance)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, preds) # Metrik tambahan

        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print(f"ROC-AUC: {roc_auc}")

        # 8. Log Metrik Secara Manual
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc) # Metrik tambahan
        mlflow.log_metric("best_cv_score", grid_search.best_score_) # Metrik tambahan

        # 9. Log Model Secara Manual
        # Ini akan menyimpan model sebagai artefak 'model'
        print("Menyimpan model ke MLflow...")
        mlflow.sklearn.log_model(best_model, "model")

    print("\nRun 'CI Workflow' selesai.")
    print(f"Cek run di DagsHub dengan ID: {run_id}")

if __name__ == "__main__":
    main()