import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_URI = "models:/DemandCast/Production"

# Must match exactly with train.py
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "is_rush_hour",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
]

# Set page config
st.set_page_config(page_title="DemandCast Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Caching Data and Model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the Production model from MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        return model
    except Exception as e:
        st.error(f"Error loading model from MLflow: {e}")
        st.info("Make sure the MLflow tracking server is running at http://localhost:5000 and a 'Production' model exists.")
        return None

@st.cache_data
def load_data():
    """Load the features.parquet for visualization."""
    data_path = Path(__file__).parent.parent / "data" / "features.parquet"
    if data_path.exists():
        return pd.read_parquet(data_path)
    return None

# Load resources
model = load_model()
df = load_data()

# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
st.sidebar.title("DemandCast Input Parameters")

# Sidebar inputs requested by assignment
pickup_zone = st.sidebar.number_input("Pickup Zone (ID)", min_value=1, max_value=265, value=1)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_of_week = st.sidebar.selectbox("Day of Week", range(7), index=2, format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
is_weekend = st.sidebar.checkbox("Is Weekend?", value=(day_of_week >= 5))

st.sidebar.markdown("---")
st.sidebar.markdown("**Additional Auto-Calculated Features**")
# Infer month and rush hour for the model
month = 1  # Assuming January model
is_rush_hour = 1 if (not is_weekend and ((7 <= hour <= 9) or (16 <= hour <= 18))) else 0
st.sidebar.write(f"Month: {month}")
st.sidebar.write(f"Is Rush Hour: {bool(is_rush_hour)}")

# Defaults for lag features just to make the model run smoothly from interactive inputs
st.sidebar.markdown("---")
st.sidebar.markdown("**Recent History Logs (Lags)**")
demand_lag_1h = st.sidebar.number_input("Demand 1 hour ago", min_value=0, value=20)
demand_lag_24h = st.sidebar.number_input("Demand 24 hours ago", min_value=0, value=25)
demand_lag_168h = st.sidebar.number_input("Demand 1 week ago", min_value=0, value=22)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🚖 DemandCast: NYC Taxi Demand Predictor")
st.markdown("Predict the expected hourly demand for Yellow Taxis in NYC given a specific zone and time context.")

if model:
    # Build prediction dataframe
    input_data = {
        "hour": [hour],
        "day_of_week": [day_of_week],
        "is_weekend": [int(is_weekend)],
        "month": [month],
        "is_rush_hour": [is_rush_hour],
        "demand_lag_1h": [demand_lag_1h],
        "demand_lag_24h": [demand_lag_24h],
        "demand_lag_168h": [demand_lag_168h],
    }
    
    input_df = pd.DataFrame(input_data)[FEATURE_COLS]
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Display massive metric
    st.metric(label=f"Predicted Demand (Zone {pickup_zone} at {hour}:00)", value=f"{int(round(prediction))} rides")
    
    # Operations Note
    st.info("""
    **Interpreter / Operations Note:**
    On average, our demand forecast is off by about 8.8 rides per hour in any given zone. 
    For operations, this means if we schedule drivers based on this forecast, we will typically have about 9 too many or 9 too few drivers waiting in a zone.
    """)
    
    st.divider()
    
    # Data visualization
    if df is not None:
        st.subheader("Historical Demand by Hour of Day")
        
        # Calculate average demand by hour
        hourly_demand = df.groupby("hour")["demand"].mean().reset_index()
        
        # Bar chart
        st.bar_chart(data=hourly_demand.set_index("hour"))
    else:
        st.warning("Could not load data/features.parquet to show visualizations.")
else:
    st.error("Model unavailable. Please start MLflow and ensure a Production model is registered.")
