# Week 4: Predictive Modeling and Optimization in Logistics Systems

## Objective
Predict shipment delivery time (hours) from logistics features, compare regression models, tune a Random Forest, evaluate it, and produce operational recommendations.

## Setup (Windows / VS Code)
1. Open this folder in VS Code.
2. Open Terminal → New Terminal.
3. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
4. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run `python -m pip install -r requirements.txt` without activation, or use Command Prompt and `.venv\Scripts\activate.bat`.
5. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Run:
   ```powershell
   python main.py
   ```

## Outputs
The `outputs` folder is created automatically and includes:
- `simulated_logistics_data.csv`
- `model_evaluation_metrics.csv`
- `test_set_predictions.csv`
- `shipment_predictions_and_recommendations.csv`
- `risk_band_summary.csv`
- `project_summary.csv`
- `best_delivery_time_model.joblib`
- PNG charts for target distribution, model comparison, and actual-vs-predicted results.

## Important
The dataset is simulated and the estimated cost field is illustrative. Report the metrics produced by your own run; do not invent results. This model is a learning prototype and should not be used for live dispatch decisions without real data, time-aware validation, monitoring, and business review.
