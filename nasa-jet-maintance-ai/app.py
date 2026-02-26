import streamlit as st
import pandas as pd
import joblib
import numpy as np


# --- 1. HELPER FUNCTIONS ---
def calculate_slope(series):
    """Calculates the linear trend (slope) of sensor values over time."""
    y = series.values
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0] if len(y) > 1 else 0


def highlight_risks(val):
    """Applies color-coding based on the Remaining Useful Life (RUL)."""
    # Red for critical, Orange for warning, Green for healthy
    color = '#ff4b4b' if val < 30 else '#ffa500' if val < 75 else '#28a745'
    return f'background-color: {color}; color: white'


# --- 2. CONFIGURATION & ASSET LOADING ---
st.set_page_config(page_title="Jet Engine AI Dashboard", layout="wide")
st.title("✈️ Jet Engine Predictive Maintenance & Fleet Management")

# Sidebar: Model Selection
st.sidebar.header("🛠️ System Configuration")
dataset_mode = st.sidebar.selectbox(
    "Select Model Type",
    ["Standard (FD001)", "Complex/Multi-Regime (FD004)"]
)


@st.cache_resource
def load_assets(mode):
    """Loads models and scalers based on the selected engine environment."""
    if mode == "Standard (FD001)":
        return joblib.load('models/xgboost_model.pkl'), joblib.load('models/minmax_scaler.pkl'), None
    else:
        # Optimized models for multi-regime environments (High altitude, various speeds)
        return joblib.load('models/xgboost_fd004.pkl'), None, joblib.load('models/kmeans_fd004.pkl')


try:
    model, scaler, kmeans = load_assets(dataset_mode)
except Exception as e:
    st.error(f"Failed to load model assets! Ensure .pkl files are in the directory. Error: {e}")


# --- 3. SMART DATA PROCESSING ENGINE ---
def process_data(df, mode, model_asset, scaler_asset, kmeans_asset):
    """Handles feature engineering and normalization dynamically."""
    # NASA Standard Column Names
    col_names = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    df.columns = col_names

    # Process all 21 sensors to ensure the model has every possible feature it was trained on
    all_sensors = [f's_{i}' for i in range(1, 22)]

    # STEP 1: FD004 Regime-Based Normalization (Clustering)
    if mode == "Complex/Multi-Regime (FD004)":
        # Identify flight regime using KMeans (Settings 1-3 determine flight condition)
        df['regime'] = kmeans_asset.predict(df[['setting_1', 'setting_2', 'setting_3']])
        # Normalize each sensor within its own flight regime (Z-Score)
        for s in all_sensors:
            df[s] = df.groupby('regime')[s].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

    # STEP 2: Feature Engineering (Rolling Stats, Lag, Slope, Z-Score)
    # We calculate 'max_cycles' even if not used by the model for data integrity checks
    df['max_cycles'] = df.groupby('unit_nr')['time_cycles'].transform('max')

    for s in all_sensors:
        # 10-cycle rolling average and standard deviation
        df[f'{s}_av'] = df.groupby('unit_nr')[s].transform(lambda x: x.rolling(window=10).mean())
        df[f'{s}_sd'] = df.groupby('unit_nr')[s].transform(lambda x: x.rolling(window=10).std())
        # Previous cycle value (Lag)
        df[f'{s}_lag'] = df.groupby('unit_nr')[s].shift(1)
        # Rate of change (Slope)
        df[f'{s}_slope'] = df.groupby('unit_nr')[s].transform(lambda x: x.rolling(window=10).apply(calculate_slope))
        # Global Z-Score for anomaly detection
        df[f'{s}_zscore'] = df.groupby('unit_nr')[s].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

    # Fill initial NaNs created by rolling windows/lags with 0
    df = df.fillna(0)

    # STEP 3: DYNAMIC FEATURE ALIGNMENT
    # Extract exactly what the XGBoost model expects to avoid 'feature mismatch' errors
    model_expected_features = model_asset.feature_names_in_.tolist()

    final_df = df.copy()

    # STEP 4: Scaling for FD001 (If required by the model training pipeline)
    if mode == "Standard (FD001)" and scaler_asset is not None:
        final_df[model_expected_features] = scaler_asset.transform(df[model_expected_features])

    return final_df, model_expected_features


# --- 4. UI & ANALYSIS ---
uploaded_file = st.sidebar.file_uploader("Upload Sensor Data (TXT/CSV)", type=['txt', 'csv'])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file, sep='\s+', header=None)

    try:
        # Process the raw data and get the list of required features
        processed_df, input_features = process_data(raw_df, dataset_mode, model, scaler, kmeans)

        st.sidebar.divider()
        analysis_scope = st.sidebar.radio("Analysis Scope", ["Full Fleet Summary", "Single Engine Detail"])

        if analysis_scope == "Full Fleet Summary":
            st.header(f"📊 Fleet Health Overview ({dataset_mode})")

            if st.button("Run Fleet Analysis"):
                # Get the latest flight cycle for every engine in the file
                fleet_latest = processed_df.groupby('unit_nr').tail(1)
                # Predict RUL using only the features the model was trained on
                fleet_preds = model.predict(fleet_latest[input_features])

                results = pd.DataFrame({
                    'Engine ID': fleet_latest['unit_nr'].astype(int),
                    'Current Cycle': fleet_latest['time_cycles'].astype(int),
                    'Predicted RUL': np.maximum(0, fleet_preds.astype(int))
                }).reset_index(drop=True)

                # Overview Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Engines", len(results))
                c2.metric("Critical (<30 Cycles)", len(results[results['Predicted RUL'] < 30]), delta_color="inverse")
                c3.metric("Avg. Fleet Life", f"{int(results['Predicted RUL'].mean())} Cycles")

                st.divider()
                st.dataframe(results.style.applymap(highlight_risks, subset=['Predicted RUL']),
                             use_container_width=True)

        else:  # Single Engine Detail
            engine_id = st.sidebar.selectbox("Select Engine ID", processed_df['unit_nr'].unique())
            st.header(f"🔍 Technical Analysis: Engine {engine_id}")

            if st.button("Query Engine Status"):
                engine_data = processed_df[processed_df['unit_nr'] == engine_id].tail(1)
                prediction = model.predict(engine_data[input_features])
                res = max(0, int(prediction[0]))

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Remaining Useful Life", f"{res} Cycles")
                    if res < 30:
                        st.error("EMERGENCY MAINTENANCE REQUIRED!")
                    elif res < 75:
                        st.warning("PLAN MAINTENANCE SOON")
                    else:
                        st.success("ENGINE STATUS: HEALTHY")

                with col2:
                    # Visualization of Sensor 11 (High correlation with engine degradation)
                    history = raw_df[raw_df[0] == engine_id]
                    st.write("Sensor 11 (Temperature/Speed) Trend")
                    st.line_chart(history.iloc[:, 14])

    except Exception as e:
        st.error(f"An error occurred while processing data: {e}")
        st.info("Tip: Ensure the uploaded file matches the NASA CMAPSS format.")

else:
    st.info("👋 Select the dataset type and upload a file from the sidebar to begin.")