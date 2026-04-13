"""
features.py — Feature engineering for DemandCast
=================================================
This module contains feature engineering logic for the NYC taxi demand
forecasting pipeline.
"""

import pandas as pd
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Feature column contract
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    'PULocationID', 
    'hour',             # Fixed to match docstring (was hour_of_day)
    'day_of_week', 
    'is_weekend', 
    'month', 
    'is_rush_hour',
    'demand_lag_1h',
    'demand_lag_24h',
    'demand_lag_168h'
]

# ---------------------------------------------------------------------------
# 1. clean_data
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw trip-level rows before feature engineering."""
    mask = (
        (df['trip_distance'] > 0) & (df['trip_distance'] <= 100) &  
        (df['fare_amount'] > 0) & (df['fare_amount'] <= 500) &      
        (df['passenger_count'] > 0) & (df['passenger_count'] <= 6) 
    )
    return df.loc[mask].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. create_temporal_features
# ---------------------------------------------------------------------------
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from the tpep_pickup_datetime column."""
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    
    # Create the datetime bucket key
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('h')
    
    # Extract temporal features (matching docstring exactly)
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['tpep_pickup_datetime'].dt.month
    
    df['is_rush_hour'] = (
        df['hour'].isin([7, 8, 17, 18]) & (df['day_of_week'] < 5)
    ).astype(int)
    
    return df

# ---------------------------------------------------------------------------
# 3. aggregate_to_hourly_demand
# ---------------------------------------------------------------------------
def aggregate_to_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual trips into hourly demand counts per pickup zone."""
    # Per your note, we include the other temporal columns in the groupby 
    # so they aren't lost during aggregation.
    group_cols = [
        'PULocationID', 
        'pickup_hour', 
        'hour', 
        'day_of_week', 
        'is_weekend', 
        'month', 
        'is_rush_hour'
    ]
    
    hourly = df.groupby(group_cols).size().reset_index(name='demand')
    
    return hourly

# ---------------------------------------------------------------------------
# 4. add_lag_features
# ---------------------------------------------------------------------------
# AI PROMPT USED FOR IMPLEMENTATION:
# "Write a pandas function to add 1-hour, 24-hour, and 168-hour lag features for a target column. 
# Make sure to sort temporally and group by the zone_col so that lag features don't bleed across different geographic zones."

def add_lag_features(
    df: pd.DataFrame,
    zone_col: str = "PULocationID",
    target_col: str = "demand",
) -> pd.DataFrame:
    """Add lagged demand features, computed separately for each zone."""
    # FIXED: Must sort by the actual datetime column (pickup_hour) to step back in time correctly
    df = df.sort_values([zone_col, 'pickup_hour']).copy()
    
    df['demand_lag_1h'] = df.groupby(zone_col)[target_col].shift(1)
    df['demand_lag_24h'] = df.groupby(zone_col)[target_col].shift(24)
    df['demand_lag_168h'] = df.groupby(zone_col)[target_col].shift(168)
    
    return df