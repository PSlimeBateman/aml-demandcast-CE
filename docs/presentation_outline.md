# Project 1 — DemandCast: Presentation Outline

## 1. Problem
* **What is DemandCast:** A predictive machine learning model that forecasts the hourly demand for NYC Yellow Taxis across different zones.
* **Why Taxi Operations Care:** By accurately predicting where and when riders will need taxis, operations managers can preemptively direct drivers to high-demand zones. This reduces driver idle time, decreases passenger wait times, and maximizes total fleet revenue.

## 2. Data & Features
* **Dataset Used:** NYC TLC Yellow Taxi Trip Records (January 2024).
* **Features Engineered:** We processed raw trips into hourly demand blocks, adding temporal features (`hour`, `day_of_week`, `month`), conditional flags (`is_weekend`, `is_rush_hour`), and rolling lag features (`demand_lag_1h`, `demand_lag_24h`, `demand_lag_168h`) to capture momentum and seasonality.
* **Most Important EDA Finding:** The immense value of temporal context. The historical demand from exactly one week prior (`demand_lag_168h`) combined with time-of-day features cleanly revealed recurring human behavioral patterns (e.g., commute spikes vs. weekend nightlife).

## 3. Model
* **Models Tried:** Linear Regression (Baseline), Random Forest Regressor, and Gradient Boosting Regressor.
* **The Winner:** Random Forest Regressor (Tuned via Optuna with 15 trials).
* **Best Validation MAE (in Plain Terms):** Our tuned model achieved a validation MAE of 8.73 rides and a completely sealed Test MAE of 7.75. In plain terms: **On average, our demand forecast is off by about 8 rides per hour in any given zone. For operations, this means if we schedule drivers based on this forecast, we will typically have about 8 too many or 8 too few drivers waiting in a zone.**

## 4. Demo
* **Live Streamlit App:**
    * Explain the user sidebar inputs and how operations would use them to query a specific zone, day, and time.
    * Walk through the UI, demonstrating the prominent `st.metric()` predicted ride counter.
    * Briefly show the bar chart visualization validating the historical curve of our demand forecasts (e.g., highlighting peak commuting hours).

## 5. Reflection
* **One thing that surprised me:** How prone tree-based models (like Random Forest) can be to overfitting on temporal data if you don't restrict their depth (`max_depth`) and minimum leaf samples (`min_samples_leaf`), and how powerfully Optuna navigated that trade-off.
* **One thing I would do differently:** Instead of using fixed lags (1 hour ago, 24 hours ago), I would like to try and engineer rolling window metrics (e.g., "average of the last 6 hours") to give the model a smoother, more robust sense of recent demand momentum. I think that would be interesting to see how it affects the accuracy.
