import pandas as pd
import numpy as np

def reduce_mem_usage(df): # Iterates through all columns of a dataframe and modifies the data type to reduce memory usage.
    
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Initial memory usage: {start_mem:.2f} MB")
    
    for col in df.columns:
        # Process  numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Check if it's an integer
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            
            # Or a float lol
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Optimized memory usage: {end_mem:.2f} MB")
    return df

def align_weather_timezones(weather_df): # Converts weather timestamps from UTC to local site time

    # All UTF offsets known in the dataset
    site_timezone_offsets = {
        0: -4, 1: 0, 2: -7, 3: -4, 4: -8, 5: 0, 6: -4, 7: -4, 
        8: -4, 9: -5, 10: -7, 11: -4, 12: 0, 13: -5, 14: -4, 15: -4
    }

    # Applying the offset mapping
    offset_map = weather_df['site_id'].map(site_timezone_offsets)

    # Converting from UTC to local time
    weather_df['timestamp'] = weather_df['timestamp'] + pd.to_timedelta(offset_map, unit='h')
    return weather_df

def correct_site_zero_units(df): # Converts electric meter readings from kBTU to kWh.
    
    # mask locates rows where both the site and meter are 0
    mask = (df['site_id'] == 0) & (df['meter'] == 0)
    
    # Changing by multiplying by the conversion factor
    df.loc[mask, 'meter_reading'] = df.loc[mask, 'meter_reading'] * 0.293071
    
    return df

def load_and_merge_data(data_dir="."): # Loads, optimizes, and merges the datasets.

    # Load the bulding metadata
    building_df = pd.read_csv(f"{data_dir}/building_metadata.csv") 
    building_df = reduce_mem_usage(building_df)
    
    # Load the weather data
    weather_df = pd.read_csv(f"{data_dir}/weather_train.csv", parse_dates=['timestamp'])
    weather_df = align_weather_timezones(weather_df)
    weather_df = reduce_mem_usage(weather_df)
    
    # Load the training data
    train_df = pd.read_csv(f"{data_dir}/train.csv", parse_dates=['timestamp'])
    train_df = reduce_mem_usage(train_df)
    
    # Merge the loaded datasets
    # Link building_id in train to building_id in metadata
    master_df = train_df.merge(building_df, on='building_id', how='left')
    
    # Apply the unit correction
    master_df = correct_site_zero_units(master_df)
    
    # Link site_id and timestamp in master to weather data
    master_df = master_df.merge(weather_df, on=['site_id', 'timestamp'], how='left')
    
    print("Final Master DataFrame Shape:", master_df.shape)
    
    return master_df

if __name__ == "__main__":
    df = load_and_merge_data(data_dir=".")
    
    # Export a small, clean slice for visual analysis
    sample_df = df[df['building_id'].isin([0, 50, 100, 150])].copy()
    sample_df.to_csv("eda_sample_buildings.csv", index=False)
    print("Sample exported to 'eda_sample_buildings.csv'")

    df.to_pickle("ashrae_master_p1.pkl")
    print("Saved as 'ashrae_master_p1.pkl'.")