Building Energy Anomaly Detection

A machine learning architecture designed to identify abnormal energy consumption patterns in commercial buildings using the ASHRAE Great Energy Predictor III dataset. 

Because industrial energy data often lacks ground-truth labels for "faults," this project introduces a custom **Synthetic Anomaly Injection Protocol** to mathematically validate unsupervised models using supervised metrics (Recall/Precision).

## Project Overview
Building operations account for a massive percentage of global energy consumption. Faulty HVAC systems, broken sensors, or inefficient scheduling often go unnoticed for months, resulting in massive financial and environmental waste. 

This project builds a modular, memory-optimized data pipeline that processes millions of telemetry rows, engineers thermodynamic features, and pits two anomaly detection algorithms against each other in a programmatic benchmark.

## System Architecture

### 1. Data Engineering & Memory Optimization
Processing the 20M+ row ASHRAE dataset required aggressive memory management and vectorized cleaning.
* **Dynamic Downcasting:** Algorithmically evaluated numeric columns to compress 64-bit floats/integers into 8-bit, 16-bit, or 32-bit types, reducing RAM overhead by >50%.
* **Relational Alignment:** Shifted UTC weather timestamps to match local site times and corrected hidden kBTU to kWh unit discrepancies.
* **Algorithmic Fault Removal:** Utilized vectorized operations (`shift()` and `cumsum()`) to identify and drop "dead sensors" (48+ hours of consecutive `0.0` readings).
* **Thermodynamic Features:** Engineered contextual indicators including Heating Degree Hours (HDH) and Cooling Degree Hours (CDH) based on an 18°C baseline.

### 2. Machine Learning & Synthetic Validation
To solve the lack of ground-truth labels, a programmatic test-bed was built.
* **Synthetic Injection:** A control subset of data was spiked by a factor of 3x-5x, mathematically injecting verifiable faults while safely enforcing data-type casting.
* **Hyperparameter Grid Search:** Automated tuning of the `contamination` threshold to navigate the Precision vs. Recall tradeoff (preventing the "Boy Who Cried Wolf" over-flagging issue).
* **Competitive Benchmarking:** The optimized pipeline ran a localized density algorithm (Local Outlier Factor) against a global decision-tree algorithm (Isolation Forest).

### 3. Interactive Diagnostic Dashboard
A front-end UI built with **Streamlit** allows facility managers to interactively tweak model hyperparameters and visualize the algorithmic detection of real-world anomalies.

---

## Key Findings & Results
The project's benchmark proved that global thermodynamic variance must be accounted for in energy analysis. 

When restricted to local neighborhood densities (n_neighbors = 10 to 50), the **Local Outlier Factor (LOF)** severely underperformed, achieving only a **~39% detection rate**. It struggled to differentiate synthetic spikes from normal, dense clusters of seasonal usage. 

In contrast, the **Isolation Forest** successfully mapped the global temporal variance of the building's yearly lifecycle, achieving an **87% detection rate** on synthetic anomalies while successfully isolating ~280 hours of genuine, undocumented faults in the historical data.

---

## How to Run This Project

### Prerequisites
* Python 3.8+
* pandas, numpy, scikit-learn, matplotlib, streamlit

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/hadaperdida/Building-Energy-Anomaly-Detection.git](https://github.com/hadaperdida/Building-Energy-Anomaly-Detection.git)
   cd Building-Energy-Anomaly-Detection
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib streamlit

### Execution
To run the interactive Streamlit dashboard:
  ```bash
  streamlit run app.py
  ```

Note: The raw ASHRAE dataset is excluded from this repository due to size constraints. The repository utilizes the serialized, optimized data structures (.pkl files) generated from the Phase 1 and Phase 2 pipelines.
