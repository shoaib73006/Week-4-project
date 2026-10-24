"""
Week 4 Task: Predictive Modeling and Optimization in Logistics Systems
Author: [Your Name]

Run:
    pip install -r requirements.txt
    python main.py

This project simulates logistics data, trains and evaluates regression models,
selects a model using cross-validation, tunes a Random Forest, and creates
prediction/optimization outputs and visualizations.
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
N_ROWS = 3000
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def simulate_logistics_data(n_rows: int = N_ROWS, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a reproducible hypothetical logistics dataset."""
    rng = np.random.default_rng(random_state)

    distance_km = rng.uniform(5, 1200, n_rows)
    traffic_level = rng.choice(["Low", "Medium", "High"], n_rows, p=[0.30, 0.45, 0.25])
    weather = rng.choice(["Clear", "Rain", "Storm"], n_rows, p=[0.65, 0.27, 0.08])
    shipment_weight_kg = rng.uniform(0.5, 1500, n_rows)
    vehicle_type = rng.choice(["Bike", "Van", "Truck"], n_rows, p=[0.15, 0.45, 0.40])
    warehouse_load_pct = rng.uniform(20, 100, n_rows)
    driver_experience_years = rng.integers(0, 21, n_rows)
    priority = rng.choice(["Standard", "Express"], n_rows, p=[0.75, 0.25])
    stops_count = rng.integers(1, 9, n_rows)

    traffic_penalty = pd.Series(traffic_level).map({"Low": 0.5, "Medium": 2.5, "High": 6.0}).to_numpy()
    weather_penalty = pd.Series(weather).map({"Clear": 0.0, "Rain": 2.5, "Storm": 7.0}).to_numpy()
    vehicle_penalty = pd.Series(vehicle_type).map({"Bike": 0.8, "Van": 0.4, "Truck": 1.2}).to_numpy()
    express_adjustment = np.where(priority == "Express", -1.2, 0.0)
    experience_adjustment = np.maximum(0, 8 - driver_experience_years) * 0.12
    warehouse_penalty = np.maximum(0, warehouse_load_pct - 70) * 0.045

    # A simplified simulated relationship, not a real-world guarantee.
    delivery_time_hours = (
        1.0
        + distance_km / 55
        + traffic_penalty
        + weather_penalty
        + shipment_weight_kg / 900
        + vehicle_penalty
        + warehouse_penalty
        + experience_adjustment
        + stops_count * 0.35
        + express_adjustment
        + rng.normal(0, 1.4, n_rows)
    )
    delivery_time_hours = np.clip(delivery_time_hours, 0.5, None)

    # Estimated operating cost for optimization analysis.
    cost = (
        40
        + distance_km * 1.8
        + shipment_weight_kg * 0.08
        + stops_count * 12
        + np.where(traffic_level == "High", 70, np.where(traffic_level == "Medium", 30, 10))
        + np.where(weather == "Storm", 90, np.where(weather == "Rain", 25, 0))
        + np.where(vehicle_type == "Truck", 60, np.where(vehicle_type == "Van", 25, 10))
        + rng.normal(0, 25, n_rows)
    )
    cost = np.clip(cost, 20, None)

    df = pd.DataFrame({
        "distance_km": distance_km.round(2),
        "traffic_level": traffic_level,
        "weather": weather,
        "shipment_weight_kg": shipment_weight_kg.round(2),
        "vehicle_type": vehicle_type,
        "warehouse_load_pct": warehouse_load_pct.round(2),
        "driver_experience_years": driver_experience_years,
        "priority": priority,
        "stops_count": stops_count,
        "estimated_shipping_cost": cost.round(2),
        "delivery_time_hours": delivery_time_hours.round(2),
    })

    # Add a small amount of missing data to demonstrate preprocessing.
    for col in ["traffic_level", "weather", "shipment_weight_kg"]:
        idx = rng.choice(df.index, size=max(1, int(n_rows * 0.01)), replace=False)
        df.loc[idx, col] = np.nan

    return df


def build_pipeline(numeric_features, categorical_features, model):
    """Preprocess numeric/categorical columns and fit the selected estimator."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Fit model and return its test metrics and predictions."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return {
        "model": name,
        "MAE_hours": mean_absolute_error(y_test, predictions),
        "RMSE_hours": rmse,
        "R2": r2_score(y_test, predictions),
    }, predictions


def save_plots(df, metrics_df, y_test, predictions, best_model):
    """Save evaluation and exploratory charts."""
    # Target distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["delivery_time_hours"], bins=35)
    plt.title("Distribution of Simulated Delivery Times")
    plt.xlabel("Delivery time (hours)")
    plt.ylabel("Number of shipments")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "delivery_time_distribution.png", dpi=150)
    plt.close()

    # Model comparison
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["model"], metrics_df["MAE_hours"])
    plt.title("Model Comparison: Mean Absolute Error")
    plt.ylabel("MAE (hours; lower is better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison_mae.png", dpi=150)
    plt.close()

    # Actual vs predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.45)
    low = min(float(np.min(y_test)), float(np.min(predictions)))
    high = max(float(np.max(y_test)), float(np.max(predictions)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title(f"Actual vs Predicted Delivery Time ({best_model})")
    plt.xlabel("Actual delivery time (hours)")
    plt.ylabel("Predicted delivery time (hours)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


def create_optimization_recommendations(df, model, feature_columns):
    """
    Score all shipments and create a simple priority list.
    This is decision support, not a route optimizer: it does not solve vehicle
    routing constraints or guarantee a cost saving.
    """
    scored = df.copy()
    scored["predicted_delivery_time_hours"] = model.predict(scored[feature_columns])
    scored["delay_risk_band"] = pd.cut(
        scored["predicted_delivery_time_hours"],
        bins=[-np.inf, 8, 16, np.inf],
        labels=["Lower", "Medium", "Higher"],
    )

    # Illustrative operational rules based on the simulated dataset.
    def recommendation(row):
        actions = []
        if row["traffic_level"] == "High":
            actions.append("Consider an off-peak dispatch window")
        if row["weather"] == "Storm":
            actions.append("Review weather buffer and customer ETA")
        if row["warehouse_load_pct"] >= 85:
            actions.append("Balance warehouse workload or add a picking shift")
        if row["stops_count"] >= 6:
            actions.append("Review stop sequence and route consolidation")
        if row["predicted_delivery_time_hours"] >= 16:
            actions.append("Flag for dispatcher review and proactive customer update")
        if not actions:
            actions.append("Monitor shipment; no special rule triggered")
        return "; ".join(actions)

    scored["optimization_recommendation"] = scored.apply(recommendation, axis=1)
    scored = scored.sort_values("predicted_delivery_time_hours", ascending=False)
    scored.to_csv(OUTPUT_DIR / "shipment_predictions_and_recommendations.csv", index=False)

    summary = (
        scored.groupby("delay_risk_band", observed=False)
        .agg(
            shipment_count=("predicted_delivery_time_hours", "size"),
            avg_predicted_delivery_hours=("predicted_delivery_time_hours", "mean"),
            avg_estimated_shipping_cost=("estimated_shipping_cost", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "risk_band_summary.csv", index=False)
    return scored, summary


def main():
    print("=" * 72)
    print("WEEK 4: PREDICTIVE MODELING AND OPTIMIZATION IN LOGISTICS")
    print("=" * 72)

    # 1. Data simulation and export
    df = simulate_logistics_data()
    df.to_csv(OUTPUT_DIR / "simulated_logistics_data.csv", index=False)
    print(f"\nDataset shape: {df.shape}")
    print("\nSample rows:")
    print(df.head().to_string(index=False))
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    target = "delivery_time_hours"
    numeric_features = [
        "distance_km",
        "shipment_weight_kg",
        "warehouse_load_pct",
        "driver_experience_years",
        "stops_count",
        "estimated_shipping_cost",
    ]
    categorical_features = ["traffic_level", "weather", "vehicle_type", "priority"]
    feature_columns = numeric_features + categorical_features

    X = df[feature_columns]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # 2. Compare baseline candidate models on the same holdout split.
    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = []
    fitted = {}
    predictions_by_model = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(numeric_features, categorical_features, estimator)
        metrics, predictions = evaluate_model(
            name, pipeline, X_train, X_test, y_train, y_test
        )
        results.append(metrics)
        fitted[name] = pipeline
        predictions_by_model[name] = predictions

    metrics_df = pd.DataFrame(results).sort_values("MAE_hours")
    metrics_df.to_csv(OUTPUT_DIR / "model_evaluation_metrics.csv", index=False)
    print("\nHoldout test metrics (lower MAE/RMSE is better; higher R² is better):")
    print(metrics_df.round(3).to_string(index=False))

    # 3. Cross-validation on training data; scoring neg MAE means larger is better.
    cv_scores = cross_val_score(
        fitted["Random Forest"],
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    print(f"\nRandom Forest 5-fold CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f} hours")

    # 4. Tune Random Forest using training data only.
    base_pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    )
    parameter_distributions = {
        "model__n_estimators": [100, 150, 200, 300],
        "model__max_depth": [None, 8, 12, 16, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [0.7, 1.0],
    }
    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=parameter_distributions,
        n_iter=12,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_
    tuned_predictions = tuned_model.predict(X_test)
    tuned_metrics = {
        "model": "Tuned Random Forest",
        "MAE_hours": mean_absolute_error(y_test, tuned_predictions),
        "RMSE_hours": np.sqrt(mean_squared_error(y_test, tuned_predictions)),
        "R2": r2_score(y_test, tuned_predictions),
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([tuned_metrics])], ignore_index=True)
    metrics_df = metrics_df.sort_values("MAE_hours")
    metrics_df.to_csv(OUTPUT_DIR / "model_evaluation_metrics.csv", index=False)
    print("\nTuned Random Forest best parameters:")
    print(search.best_params_)
    print(f"Best CV MAE: {-search.best_score_:.3f} hours")
    print("\nFinal evaluation including tuned model:")
    print(metrics_df.round(3).to_string(index=False))

    # Save final model, test predictions, and plots.
    joblib.dump(tuned_model, OUTPUT_DIR / "best_delivery_time_model.joblib")
    prediction_output = X_test.copy()
    prediction_output["actual_delivery_time_hours"] = y_test.to_numpy()
    prediction_output["predicted_delivery_time_hours"] = tuned_predictions
    prediction_output["absolute_error_hours"] = np.abs(
        prediction_output["actual_delivery_time_hours"]
        - prediction_output["predicted_delivery_time_hours"]
    )
    prediction_output.to_csv(OUTPUT_DIR / "test_set_predictions.csv", index=False)

    save_plots(df, metrics_df, y_test, tuned_predictions, "Tuned Random Forest")

    # Generate actionable rule-based suggestions using the tuned model.
    scored, risk_summary = create_optimization_recommendations(df, tuned_model, feature_columns)
    print("\nRisk band summary:")
    print(risk_summary.round(2).to_string(index=False))
    print("\nTop 10 shipments for review:")
    print(
        scored[
            ["predicted_delivery_time_hours", "traffic_level", "weather",
             "warehouse_load_pct", "stops_count", "optimization_recommendation"]
        ].head(10).to_string(index=False)
    )

    # Save a small summary for easy report writing.
    best_row = metrics_df.sort_values("MAE_hours").iloc[0]
    report_summary = pd.DataFrame([{
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "best_model_by_test_MAE": best_row["model"],
        "best_test_MAE_hours": best_row["MAE_hours"],
        "best_test_RMSE_hours": best_row["RMSE_hours"],
        "best_test_R2": best_row["R2"],
        "tuned_random_forest_cv_MAE_hours": -search.best_score_,
    }])
    report_summary.to_csv(OUTPUT_DIR / "project_summary.csv", index=False)

    print(f"\nCompleted. All output files are in: {OUTPUT_DIR}")
    print("Important: the data is simulated. Replace it with real historical data before operational use.")


if __name__ == "__main__":
    main()
