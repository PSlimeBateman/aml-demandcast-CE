import pandas as pd
from pathlib import Path
import sys

# Add src to the path so we can import our custom module
sys.path.append(str(Path(__file__).parent))
from src.features import (
    clean_data, 
    create_temporal_features, 
    aggregate_to_hourly_demand, 
    add_lag_features
)

def main():
    print("Loading raw data...")
    # Ensuring we target the exact parquet file from your week 1 dataset
    data_path = Path("data/yellow_tripdata_2025-01.parquet")
    output_path = Path("data/features.parquet")
    
    if not data_path.exists():
        print(f"Error: Could not find {data_path}. Ensure you are running this from the project root.")
        return

    df = pd.read_parquet(data_path)
    initial_rows = len(df)
    
    print("Cleaning data...")
    df = clean_data(df)
    print(f"  Removed {initial_rows - len(df)} anomalous rows.")
    
    print("Extracting temporal features...")
    df = create_temporal_features(df)
    
    print("Aggregating to hourly demand...")
    hourly_df = aggregate_to_hourly_demand(df)
    print(f"  Aggregated into {len(hourly_df)} hourly zone records.")
    
    print("Adding lag features...")
    features_df = add_lag_features(hourly_df)
    
    # Drop ONLY the rows with NaNs in the lag columns (caused by the 168h lag on the first week)
    features_df = features_df.dropna(subset=['demand_lag_168h']).reset_index(drop=True)
    
    print(f"Saving {len(features_df)} model-ready rows to {output_path}...")
    features_df.to_parquet(output_path, index=False)
    print("Pipeline complete!")

if __name__ == "__main__":
    main()