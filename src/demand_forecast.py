"""
demand_forecast.py
Food Donation Demand Forecasting & Expiry Hazard Predictor
Machine Learning Extension for the Frugal Food Redistribution Platform (BSCS FYP 2024).

DATASET PROVENANCE NOTE:
Per project specifications, this module is trained on documented synthetic food recovery data,
simulating food category, ambient temperature, transit duration, and shelter capacity.

Author: Farid Masood Khan
Roadmap Project 4 — Frugal + ML Extension
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib


def generate_documented_synthetic_dataset(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate documented synthetic records representing food donation surplus:
    - food_category: Cooked Meals, Fresh Produce, Bakery, Dairy, Packaged Pantry
    - quantity_kg: 5.0 to 120.0 kg
    - ambient_temp_c: 15 to 42 C (warm climate factors)
    - hours_since_prep: 1 to 24 hours
    - shelter_distance_km: 1.0 to 25.0 km
    - day_of_week: Monday through Sunday
    Targets:
    - demand_meals_needed: Continuous demand from partner shelter
    - spoils_before_delivery: Binary risk indicator
    """
    rng = np.random.RandomState(seed)
    categories = ["Cooked Meals", "Fresh Produce", "Bakery", "Dairy", "Packaged Pantry"]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    food_category = rng.choice(categories, size=n_samples, p=[0.35, 0.25, 0.15, 0.10, 0.15])
    day_of_week = rng.choice(days, size=n_samples)
    quantity_kg = rng.uniform(5.0, 100.0, size=n_samples)
    ambient_temp_c = rng.uniform(18.0, 40.0, size=n_samples)
    hours_since_prep = rng.uniform(1.0, 18.0, size=n_samples)
    shelter_distance_km = rng.uniform(1.0, 20.0, size=n_samples)

    # Synthetic demand function (higher on weekends and for cooked meals)
    base_demand = quantity_kg * 2.2 + (ambient_temp_c * 0.5)
    weekend_boost = np.isin(day_of_week, ["Sat", "Sun"]) * 25.0
    demand_meals_needed = np.maximum(10.0, base_demand + weekend_boost + rng.normal(0, 10, n_samples))

    # Spoilage hazard function (high heat + high hours for cooked meals/dairy)
    spoilage_score = (
        (ambient_temp_c - 20) * 0.08
        + (hours_since_prep) * 0.15
        + (shelter_distance_km * 0.05)
        + (np.isin(food_category, ["Cooked Meals", "Dairy"]) * 1.2)
        + rng.normal(0, 0.4, n_samples)
    )
    spoils_before_delivery = (spoilage_score > 2.5).astype(int)

    df = pd.DataFrame({
        "food_category": food_category,
        "day_of_week": day_of_week,
        "quantity_kg": quantity_kg,
        "ambient_temp_c": ambient_temp_c,
        "hours_since_prep": hours_since_prep,
        "shelter_distance_km": shelter_distance_km,
        "demand_meals_needed": demand_meals_needed,
        "spoils_before_delivery": spoils_before_delivery,
    })
    return df


def train_frugal_models():
    print("==================================================")
    print("  FRUGAL ML EXTENSION — DEMAND & EXPIRY PIPELINE  ")
    print("==================================================")

    df = generate_documented_synthetic_dataset()
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "synthetic_frugal_food_data.csv", index=False)
    print(f"[Frugal ML] Generated & saved documented synthetic dataset (N={len(df)})")

    features = ["food_category", "day_of_week", "quantity_kg", "ambient_temp_c", "hours_since_prep", "shelter_distance_km"]
    X = df[features]
    y_demand = df["demand_meals_needed"]
    y_spoil = df["spoils_before_delivery"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["quantity_kg", "ambient_temp_c", "hours_since_prep", "shelter_distance_km"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["food_category", "day_of_week"]),
        ]
    )

    X_train, X_test, y_dem_tr, y_dem_te, y_sp_tr, y_sp_te = train_test_split(
        X, y_demand, y_spoil, test_size=0.2, random_state=42
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # 1. Demand Regressor
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_trans, y_dem_tr)
    dem_preds = reg.predict(X_test_trans)
    r2 = float(r2_score(y_dem_te, dem_preds))
    rmse = float(np.sqrt(mean_squared_error(y_dem_te, dem_preds)))
    print(f"[Frugal ML] Demand Regressor | R2: {r2:.4f} | RMSE: {rmse:.2f} meals")

    # 2. Spoilage Classifier
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(X_train_trans, y_sp_tr)
    sp_preds = clf.predict(X_test_trans)
    acc = float(accuracy_score(y_sp_te, sp_preds))
    print(f"[Frugal ML] Spoilage Hazard Classifier | Accuracy: {acc:.2%}")

    # Save
    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, out_dir / "frugal_preprocessor.joblib")
    joblib.dump(reg, out_dir / "demand_regressor.joblib")
    joblib.dump(clf, out_dir / "spoilage_classifier.joblib")

    metrics = {
        "demand_r2": r2,
        "demand_rmse": rmse,
        "spoilage_accuracy": acc,
        "data_provenance": "Documented synthetic benchmark (clearly labelled as synthetic)",
    }
    with open(out_dir / "frugal_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Frugal ML] Saved trained models & metrics to {out_dir}")


if __name__ == "__main__":
    train_frugal_models()
