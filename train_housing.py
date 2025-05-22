import os
import warnings
import argparse
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def plot_predictions_vs_actual(actual, pred, filename="predictions_vs_actual.png"):
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, pred, alpha=0.5, edgecolors='w', linewidth=0.5)
    min_val = min(actual.min(), pred.min())
    max_val = max(actual.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.xlabel('Actual Values ($100k)')
    plt.ylabel('Predicted Values ($100k)')
    plt.title('Actual vs. Predicted Housing Prices')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLflow Tracking URI: {mlflow_tracking_uri}")

    experiment_name = "California Housing Fresh Experiment V2"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Создан новый эксперимент с ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Используется существующий эксперимент с ID: {experiment_id}")

    run_name_parts = [
        "RFR",
        f"est={args.n_estimators}",
        f"depth={args.max_depth if args.max_depth is not None else 'None'}",
        f"rs={args.random_state}"
    ]
    run_name = "_".join(run_name_parts)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"Начало MLflow Run ID: {run_id}")

        housing = fetch_california_housing(as_frame=True)
        X, y = housing.data, housing.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        scaler_filename = "standard_scaler.joblib"
        joblib.dump(scaler, scaler_filename)
        mlflow.log_artifact(scaler_filename, "preprocessor")
        print(f"Скейлер сохранен и залогирован: {scaler_filename}")
        if os.path.exists(scaler_filename):
            os.remove(scaler_filename)

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth if args.max_depth is not None else "None")
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("scaler", "StandardScaler")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        input_example = pd.DataFrame(X_train_scaled[:5], columns=housing.feature_names)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random-forest-model",
            input_example=input_example,
            signature=mlflow.models.infer_signature(X_train_scaled, y_pred)
        )
        print("Модель залогирована в MLflow.")

        plot_file = plot_predictions_vs_actual(y_test, y_pred)
        mlflow.log_artifact(plot_file, "plots")
        print(f"График предсказаний залогирован: {plot_file}")
        if os.path.exists(plot_file):
            os.remove(plot_file)

        mlflow.set_tag("project_phase", "development")
        mlflow.set_tag("task_type", "regression")
        print(f"MLflow Run ID: {run_id} завершен.")
