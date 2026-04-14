import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def inject_synthetic_anomalies(df, contamination_rate=0.01):
   
    # Inject anomalies artificially into the data to see if the model works 
    df = df.copy()

    # Column to track ground truth
    # 0 = normal, 1 = true anomaly
    df['is_true_anomaly'] = 0

    # Randomly select indices to become anomalies
    n_anomalies = int(len(df) * contamination_rate)
    anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)

    # Inject the anomaly~ Multiply the normal energy usage by a random spike factor
    spike_factors = np.random.uniform(3.0, 5.0, size=n_anomalies)
    new_spiked_values = df.loc[anomaly_indices, 'meter_reading'] * spike_factors

    df.loc[anomaly_indices, 'meter_reading'] = new_spiked_values.astype(df['meter_reading'].dtype)
    
    # Mark them in our ground truth column
    df.loc[anomaly_indices, 'is_true_anomaly'] = 1

    # print(f"Injected {n_anomalies} fake anomalies for testing.")
    
    return df

def train_rival_models(df):
    
    # Trains an Isolation Forest to find anomalous energy readings based on weather and time
    # Select the features the model is allowed to learn from
    features = [
        'meter_reading', 
        'air_temperature', 
        'hour', 
        'day_of_week', 
        'heating_degree_hours', 
        'cooling_degree_hours'
    ]

    X = df[features].fillna(0)

    best_recall = 0
    best_contamination = 0
    best_iso_predictions = None

    # Test from 1% to 7% for a wider range of anomalies
    contamination_rates = [0.01, 0.02, 0.04, 0.05, 0.06, 0.07]
    
    print("Model 1 - Isolation Forest")

    for cont in contamination_rates:
        model = IsolationForest(n_estimators=200, contamination=cont, random_state=42)
        preds = model.fit_predict(X)
        
        # Convert predictions (-1/1) to match the ground truth (1/0)
        binary_preds = np.where(preds == -1, 1, 0)
        
        # Calculate recall for this specific loop
        true_positives = len(df[(df['is_true_anomaly'] == 1) & (binary_preds == 1)])
        actual_anomalies = len(df[df['is_true_anomaly'] == 1])
        recall = true_positives / actual_anomalies if actual_anomalies > 0 else 0
        
        print(f"Testing Contamination {cont*100:04.1f}% -> Recall: {recall * 100:.2f}%")
        
        if recall > best_recall:
            best_recall = recall
            best_contamination = cont
            best_iso_predictions = binary_preds

    print(f"\nThe Best Contamination Score was attained using: {best_contamination*100}%")
    
    # Save the winning Isolation Forest predictions
    df['iso_prediction'] = best_iso_predictions

    print("\nModel 2: Local Outlier Factor (LOF)")
    print(f"Using the winning contamination rate ({best_contamination*100}%)")
    
    best_lof_recall = 0
    best_n_neighbors = 20
    best_lof_predictions = None
    
    # Tuning how wide LOF casts its "local" net
    neighbor_options = [10, 20, 30, 40, 50]
    
    for neighbors in neighbor_options:
        lof = LocalOutlierFactor(n_neighbors=neighbors, contamination=best_contamination)
        lof_preds = lof.fit_predict(X)
        binary_lof_preds = np.where(lof_preds == -1, 1, 0)
        
        lof_tp = len(df[(df['is_true_anomaly'] == 1) & (binary_lof_preds == 1)])
        actual_anomalies = len(df[df['is_true_anomaly'] == 1])
        lof_recall = lof_tp / actual_anomalies if actual_anomalies > 0 else 0
        
        print(f"Testing n_neighbors={neighbors:03} -> Recall: {lof_recall * 100:.2f}%")
        
        # We use >= so if the score ties, it picks the larger, more robust neighborhood
        if lof_recall >= best_lof_recall:
            best_lof_recall = lof_recall
            best_n_neighbors = neighbors
            best_lof_predictions = binary_lof_preds
            
    print(f"\nThe Best LOF n_neighbors was: {best_n_neighbors}")
    df['lof_prediction'] = best_lof_predictions
    
    return df

def evaluate_competition(df):
    
    print("\nFinal Competition Results")
    actual_anomalies = len(df[df['is_true_anomaly'] == 1])
    print(f"Fake Anomalies Injected: {actual_anomalies}\n")

    # Evaluate Model 1
    iso_tp = len(df[(df['is_true_anomaly'] == 1) & (df['iso_prediction'] == 1)])
    iso_recall = iso_tp / actual_anomalies if actual_anomalies > 0 else 0
    print(f"Isolation Forest Found: {iso_tp} (Detection Rate: {iso_recall * 100:.2f}%)")

    # Evaluate Model 2
    lof_tp = len(df[(df['is_true_anomaly'] == 1) & (df['lof_prediction'] == 1)])
    lof_recall = lof_tp / actual_anomalies if actual_anomalies > 0 else 0
    print(f"Local Outlier Factor Found: {lof_tp} (Detection Rate: {lof_recall * 100:.2f}%)")
    
    # Set the final official prediction column
    df['is_predicted_anomaly'] = df['iso_prediction']
    
    # Find real anomalies using the winning model
    real_anomalies_caught = df[(df['is_true_anomaly'] == 0) & (df['is_predicted_anomaly'] == 1)]
    print(f"\nThe Isolation Forest also found {len(real_anomalies_caught)} REAL anomalies in the actual data.")
    
    return df

if __name__ == "__main__":
    print("Loading clean dataset...")
    master_df = pd.read_pickle("ashrae_cleaned_p2.pkl")
    
    # For now, only Building 0's Electricity will be used to see if it works
    print("Filtering data for Building 0 (Electricity)...")
    b0_df = master_df[(master_df['building_id'] == 0) & (master_df['meter'] == 0)].copy()
    b0_df = b0_df.reset_index(drop=True)
    
    # Run the Machine Learning Pipeline
    test_df = inject_synthetic_anomalies(b0_df, contamination_rate=0.01)
    results_df = train_rival_models(test_df)
    final_results = evaluate_competition(results_df)
    
    # Final results for anomalies graph
    final_results.to_csv("building_0_anomaly_results.csv", index=False)
    print("Saved results to 'building_0_anomaly_results.csv'.")