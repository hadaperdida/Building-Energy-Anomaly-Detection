import pandas as pd
import numpy as np

# Cleaning section
def clean_weather_data(df):
    
    # 1. Linear interpolation for small gaps in temp/wind
    cols_to_interpolate = ['air_temperature', 'dew_temperature', 'wind_speed', 'wind_direction']
    for col in cols_to_interpolate:
        df[col] = df.groupby('site_id')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
        
    # 2. Fill missing precipitation with 0, and cloud coverage with site median
    df['precip_depth_1_hr'] = df['precip_depth_1_hr'].fillna(0)
    df['cloud_coverage'] = df.groupby('site_id')['cloud_coverage'].transform(lambda x: x.fillna(x.median()))
    df['sea_level_pressure'] = df.groupby('site_id')['sea_level_pressure'].transform(lambda x: x.fillna(x.median()))
    
    # Fill any remaining NaNs with the global median to ensure no ML models crash
    df = df.fillna(df.median(numeric_only=True))
    
    return df

def clean_metadata(df):
  
    # Fill gaps in building metadata 
    # If an Office is missing its floor_count, guess based on other Offices.
    df['year_built'] = df.groupby('primary_use')['year_built'].transform(lambda x: x.fillna(x.median()))
    df['floor_count'] = df.groupby('primary_use')['floor_count'].transform(lambda x: x.fillna(x.median()))
    
    # Fill any stragglers with overall medians
    df['year_built'] = df['year_built'].fillna(df['year_built'].median())
    df['floor_count'] = df['floor_count'].fillna(df['floor_count'].median())
    
    return df

def drop_zero_streaks(df):
   
    # Remove streaks where a meter reads 0 for more than 48 hours
    # Sort carefully to ensure time is perfectly sequential per building and meter
    df = df.sort_values(['building_id', 'meter', 'timestamp']).reset_index(drop=True)
    
    is_zero = df['meter_reading'] == 0
    zero_groups = is_zero.ne(is_zero.shift()).cumsum()
    streak_counts = df.groupby(['building_id', 'meter', zero_groups])['meter_reading'].transform('size')
    
    # Keep the row if it's not a zero / it's been zero for shorter than 48 hours
    mask = (~is_zero) | (is_zero & (streak_counts <= 48))
    
    cleaned_df = df[mask].copy()
    print(f"Dropped {len(df) - len(cleaned_df)} rows of faulty 'zero' readings.")
    
    return cleaned_df

def feature_engineering(df):

    # Extract time and thermodynaic features    
    # Temporal Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = np.where(df['day_of_week'] >= 5, 1, 0)
    
    # 2. Thermodynamic Features
    base_temp = 18.0    # Using 18°C as the baseline
    df['heating_degree_hours'] = np.maximum(base_temp - df['air_temperature'], 0)
    df['cooling_degree_hours'] = np.maximum(df['air_temperature'] - base_temp, 0)
    
    return df

if __name__ == "__main__":
    # print("Loading data...")
    
    # Loads the data exactly where 'ingest.py' left off
    master_df = pd.read_pickle("ashrae_master_p1.pkl")
    # print(f"Data loaded successfully. Shape: {master_df.shape}")
    
    # 2. Pass the data through the cleaning pipeline
    # print("\nStarting Processing...")
    cleaned_df = clean_weather_data(master_df)
    cleaned_df = clean_metadata(cleaned_df)
    cleaned_df = drop_zero_streaks(cleaned_df)
    final_df = feature_engineering(cleaned_df)
    
    # print("\nProcessing Complete. final_df shape:", final_df.shape)
    
    # 3. Save the finalized, clean data for Phase 3 (Machine Learning)
    # print("Saving processed data to pickle file...")
    final_df.to_pickle("ashrae_cleaned_p2.pkl")
    print("Saved as 'ashrae_cleaned_p2.pkl'")