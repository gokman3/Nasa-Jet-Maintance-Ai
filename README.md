# ✈️ Jet Engine Predictive Maintenance & Fleet Management AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nasa-jet-maintance-ai.streamlit.app/)

🌍 **Live Demo:** [Click here to launch the AI Dashboard](https://nasa-jet-maintance-ai.streamlit.app/)

## 📖 About The Project
This project is an interactive, web-based AI dashboard designed to predict the **Remaining Useful Life (RUL)** of turbofan jet engines. Built using the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset, the application leverages advanced Machine Learning models to provide early warnings for engine degradation, thereby optimizing maintenance schedules and preventing critical failures in fleet management.

## 🚀 Key Features
* **Two Operational Regimes:** * Standard (FD001): For engines operating under a single condition.
  * Complex (FD004): For engines operating across multiple flight regimes (varying altitudes, Mach numbers, and throttle angles).
* **Dynamic Feature Engineering:** Automatically extracts real-time features from 21 sensor readings, including 10-cycle rolling averages, standard deviations, lag variables, trend slopes, and global Z-Scores.
* **Flight Regime Clustering:** Utilizes KMeans clustering to identify and normalize sensor data based on different flight conditions in complex environments.
* **Fleet Health Overview:** Upload raw sensor data (TXT/CSV) to instantly analyze an entire fleet, identifying engines requiring emergency maintenance (< 30 cycles).
* **Single Engine Deep-Dive:** Drill down into specific engines to view exact RUL predictions and visualize highly correlated sensor trends over time.

## 🛠️ Technology Stack
* Language: Python 3.x
* Frontend/UI: Streamlit (v1.53.1)
* Data Manipulation: Pandas (v2.3.3), NumPy (v2.3.5)
* Machine Learning: XGBoost (v3.1.2), Scikit-Learn (v1.7.2)
* Model Serialization: Joblib (v1.5.2)

## 📂 Project Structure

```text
Nasa-Jet-Maintance-Ai/
├── app.py                 # Main Streamlit application script
├── requirements.txt       # Exact version dependencies for deployment
├── README.md              # Project documentation
└── models/                # Pre-trained ML models and scalers
    ├── xgboost_model.pkl  # XGBoost model for FD001 dataset
    ├── minmax_scaler.pkl  # Scaler for FD001 dataset
    ├── xgboost_fd004.pkl  # XGBoost model for FD004 dataset
    └── kmeans_fd004.pkl   # KMeans model for regime clustering (FD004)
```

## ⚙️ How to Run Locally

If you wish to run this project on your local machine, follow these steps:

1. Clone the repository:
git clone https://github.com/gokman3/Nasa-Jet-Maintance-Ai.git
cd Nasa-Jet-Maintance-Ai

2. Install the dependencies (It is recommended to use a virtual environment):
pip install -r requirements.txt

3. Run the Streamlit application:
streamlit run app.py

4. Usage:
* Open the local URL provided in your terminal (usually http://localhost:8501).
* Select the Model Type from the sidebar.
* Upload a NASA CMAPSS .txt dataset (e.g., test_FD001.txt).
* Choose between "Full Fleet Summary" or "Single Engine Detail" to view the AI predictions.
