import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 1. UI Configuration & Titles

st.set_page_config(page_title="Energy Anomaly Detector", layout="wide")
st.title("⚡ Building Energy Anomaly Detection")
st.markdown("Interactive diagnostic dashboard for ASHRAE Building 0.")

# 2. Data Loading & Caching (Makes the UI fast)

@st.cache_data
def load_and_inject_data():
    master_df = pd.read_pickle("ashrae_cleaned_p2.pkl")
    b0_df = master_df[(master_df['building_id'] == 0) & (master_df['meter'] == 0)].copy()
    b0_df = b0_df.reset_index(drop=True)
    
    # Inject synthetic anomalies for the live demo
    n_anomalies = int(len(b0_df) * 0.01)
    anomaly_indices = np.random.choice(b0_df.index, n_anomalies, replace=False)
    spike_factors = np.random.uniform(3.0, 5.0, size=n_anomalies)
    
    new_spiked_values = b0_df.loc[anomaly_indices, 'meter_reading'] * spike_factors
    b0_df.loc[anomaly_indices, 'meter_reading'] = new_spiked_values.astype(b0_df['meter_reading'].dtype)
    
    b0_df['is_true_anomaly'] = 0
    b0_df.loc[anomaly_indices, 'is_true_anomaly'] = 1
    return b0_df

df = load_and_inject_data()

# 3. Sidebar Controls (The Interactive Part)

st.sidebar.header("Model Hyperparameters")

# User selects which model to run
model_choice = st.sidebar.radio("Select Algorithm:", ("Isolation Forest", "Local Outlier Factor (LOF)"))

# User controls the sliders
contamination = st.sidebar.slider("Contamination Threshold (%)", min_value=1, max_value=10, value=6) / 100.0

if model_choice == "Local Outlier Factor (LOF)":
    n_neighbors = st.sidebar.slider("LOF Neighborhood Size", min_value=10, max_value=200, value=20, step=10)

# 4. Run Machine Learning Live

features = ['meter_reading', 'air_temperature', 'hour', 'day_of_week', 'heating_degree_hours', 'cooling_degree_hours']
X = df[features].fillna(0)

if model_choice == "Isolation Forest":
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    preds = model.fit_predict(X)
else:
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = model.fit_predict(X)

# Convert predictions
df['predicted_anomaly'] = np.where(preds == -1, 1, 0)

# Calculate Live Metrics
true_positives = len(df[(df['is_true_anomaly'] == 1) & (df['predicted_anomaly'] == 1)])
total_fake = len(df[df['is_true_anomaly'] == 1])
recall = (true_positives / total_fake) * 100

real_anomalies = len(df[(df['is_true_anomaly'] == 0) & (df['predicted_anomaly'] == 1)])

# 5. Render the Dashboard

# Display live scorecards
col1, col2, col3 = st.columns(3)
col1.metric("Detection Rate (Recall)", f"{recall:.2f}%")
col2.metric("Synthetic Faults Caught", f"{true_positives} / {total_fake}")
col3.metric("Real Anomalies Flagged", str(real_anomalies))

st.markdown("---")
st.subheader(f"Energy Usage vs. Detected Anomalies ({model_choice})")

# Create the visualization
fig, ax = plt.subplots(figsize=(15, 5))
# Plot the normal data
ax.plot(df.index, df['meter_reading'], color='lightgrey', label='Normal Usage', linewidth=1)

# Highlight the anomalies in RED
anomalies = df[df['predicted_anomaly'] == 1]
ax.scatter(anomalies.index, anomalies['meter_reading'], color='red', label='Detected Anomaly', zorder=5, s=20)

ax.set_ylabel("Energy Usage (kWh)")
ax.set_xlabel("Hours (Chronological)")
ax.legend()

# Render plot in Streamlit
st.pyplot(fig)