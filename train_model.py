from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "Bengaluru_House_Data.csv"
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
RESULTS_PATH = BASE_DIR / "model_results.txt"

def convert_sqft(value):
    """Convert values such as '1200' or '1000-1200' to a numeric area."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    numbers = re.findall(r"\d+\.?\d*", text)
    if not numbers:
        return np.nan
    values = [float(x) for x in numbers]
    return sum(values) / len(values)

def prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place Bengaluru_House_Data.csv inside the dataset folder."
        )

    data = pd.read_csv(DATA_PATH)
    required = {"location", "total_sqft", "bath", "price"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data["total_sqft"] = data["total_sqft"].apply(convert_sqft)
    if "bhk" in data.columns:
        data["bhk"] = pd.to_numeric(data["bhk"], errors="coerce")
    elif "size" in data.columns:
        data["bhk"] = pd.to_numeric(
            data["size"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce"
        )
    else:
        raise ValueError("Dataset must contain either 'bhk' or 'size' column.")

    for col in ["bath", "price"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["location"] = data["location"].astype(str).str.strip()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["location", "total_sqft", "bath", "bhk", "price"])
    data = data[
        (data["location"] != "") &
        (data["total_sqft"] > 0) &
        (data["bath"] > 0) &
        (data["bhk"] > 0) &
        (data["price"] > 0)
    ].copy()

    if len(data) < 10:
        raise ValueError("Dataset has too few valid rows. Please use at least 10 records.")

    return data

def build_pipeline(model):
    numeric_features = ["total_sqft", "bath", "bhk"]
    categorical_features = ["location"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("location", categorical_transformer, categorical_features)
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

def main():
    data = prepare_data()
    X = data[["location", "total_sqft", "bath", "bhk"]]
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": build_pipeline(LinearRegression()),
        "Random Forest": build_pipeline(
            RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        )
    }

    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        results.append((name, mae, rmse, r2))
        trained_models[name] = model

    # Save Random Forest as the default application model.
    joblib.dump(trained_models["Random Forest"], MODEL_PATH)

    report = [
        "HOUSE PRICE PREDICTION - MODEL EVALUATION",
        "Target unit: lakhs (as represented in the dataset)",
        f"Total cleaned records: {len(data)}",
        f"Training records: {len(X_train)}",
        f"Testing records: {len(X_test)}",
        "",
        "Model | MAE | RMSE | R2"
    ]
    for name, mae, rmse, r2 in results:
        report.append(f"{name} | {mae:.3f} | {rmse:.3f} | {r2:.3f}")

    RESULTS_PATH.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved evaluation report to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
