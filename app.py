import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os
from datetime import datetime
from catboost import CatBoostClassifier

# --- CONFIGURATION ---
st.set_page_config(page_title="Flight Delay AI", page_icon="✈️", layout="wide")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_artifacts():
    # 1. Load Feature Lookups (Risk scores, means, etc.)
    lookup_path = "/Users/nikitha/Documents/flight-delay-prediction-ice/notebooks/model_artifacts_final/feature_lookups.json"
    metadata_path = "/Users/nikitha/Documents/flight-delay-prediction-ice/notebooks/model_artifacts_final/model_metadata.json"
    
    if not os.path.exists(lookup_path):
        st.error(f"File not found: {lookup_path}. Please run the export script in your notebook first.")
        st.stop()
        
    with open(lookup_path, "r") as f:
        lookups = json.load(f)

    # 2. Load Model Metadata (CRITICAL for column ordering)
    if not os.path.exists(metadata_path):
        st.error(f"❌ File not found: {metadata_path}. Please run the pipeline script.")
        st.stop()

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        # Get the exact list of features the model expects
        feature_order = metadata["selected_features"]

    # 3. Load Model (Prioritize CatBoost)
    model = None
    cbm_path = "/Users/nikitha/Documents/flight-delay-prediction-ice/notebooks/model_artifacts_final/catboost_model.cbm"
    joblib_path = "model_artifacts_final/best_model.joblib"

    model_type = "unknown"
    if os.path.exists(cbm_path):
        try:
            model = CatBoostClassifier()
            model.load_model(cbm_path)
            model_type = "catboost"
        except Exception as e:
            st.error(f"Failed to load CatBoost model: {e}")
    
    elif os.path.exists(joblib_path):
        try:
            model = joblib.load(joblib_path)
            model_type = "sklearn"
        except Exception as e:
            st.error(f"Failed to load Joblib model: {e}")
    
    else:
        st.error("❌ No model file found! Checked for 'catboost_model.cbm' and 'best_model.joblib'.")
        st.stop()
        
    return model, lookups, feature_order, model_type

try:
    model, lookups, feature_order, model_type = load_artifacts()
except Exception as e:
    st.error(f"Critical Error loading artifacts: {e}")
    st.stop()

# --- INTELLIGENT PREPROCESSING ---
def preprocess_input(user_input, lookups, feature_order):
    """
    Converts raw user inputs into the exact feature set expected by the model.
    """
    data = {}
    
    # 1. Temporal Features
    dt = user_input['date']
    data['MONTH'] = int(dt.month)
    data['DAY_OF_WEEK'] = int(dt.isoweekday())
    data['DAY_OF_MONTH'] = int(dt.day)
    data['QUARTER'] = int((dt.month - 1) // 3 + 1)
    data['IS_WEEKEND'] = 1 if data['DAY_OF_WEEK'] >= 6 else 0
    
    # Seasonality
    if dt.month in [12, 1, 2]: season = "Winter"
    elif dt.month in [3, 4, 5]: season = "Spring"
    elif dt.month in [6, 7, 8]: season = "Summer"
    else: season = "Fall"
    
    data['SEASON'] = str(season) # Explicitly cast string
    data['IS_SUMMER'] = 1 if season == "Summer" else 0
    data['IS_WINTER'] = 1 if season == "Winter" else 0
    data['IS_HOLIDAY_SEASON'] = 1 if dt.month in [11, 12] else 0
    
    # Time Bins (Must be string for CatBoost)
    hour = user_input['time'].hour
    if hour <= 5: bin_ = "0-5"
    elif hour <= 8: bin_ = "6-8"
    elif hour <= 11: bin_ = "9-11"
    elif hour <= 14: bin_ = "12-14"
    elif hour <= 17: bin_ = "15-17"
    elif hour <= 20: bin_ = "18-20"
    elif hour <= 23: bin_ = "21-23"
    else: bin_ = "24+"
    data['DEP_HOUR_BIN'] = str(bin_)

    # 2. Key Identifiers (Strings)
    carrier = str(user_input['carrier'])
    origin = str(user_input['origin'])
    dest = str(user_input['dest'])
    route = f"{origin}-{dest}"
    carrier_route = f"{carrier}_{route}"
    
    data['UNIQUE_CARRIER'] = carrier
    data['ORIGIN'] = origin
    data['DEST'] = dest
    data['ROUTE'] = route
    data['CARRIER_ROUTE'] = carrier_route
    data['ORIGIN_STATE_ABR'] = "Unknown"  
    data['DEST_STATE_ABR'] = "Unknown"    

    # 3. Risk & Count Lookups
    mappings = lookups.get('mappings', {})
    defaults = lookups.get('defaults', {})
    
    # Risk Features
    features_to_map = [
        ('ORIGIN', 'ORIGIN_RISK'),
        ('DEST', 'DEST_RISK'),
        ('UNIQUE_CARRIER', 'UNIQUE_CARRIER_RISK'),
        ('ROUTE', 'ROUTE_RISK'),
        ('CARRIER_ROUTE', 'CARRIER_ROUTE_RISK'),
    ]
    
    for key_col, feature_col in features_to_map:
        val = str(data[key_col])
        data[feature_col] = float(mappings.get(feature_col, {}).get(val, defaults.get(feature_col, 0.15)))

    # Count Features
    features_to_count = [
        ('ORIGIN', 'TRAIN_ORIGIN_COUNT'),
        ('DEST', 'TRAIN_DEST_COUNT'),
        ('ROUTE', 'TRAIN_ROUTE_COUNT'),
        ('UNIQUE_CARRIER', 'TRAIN_UNIQUE_CARRIER_COUNT'),
        ('CARRIER_ROUTE', 'TRAIN_CARRIER_ROUTE_COUNT')
    ]
    
    for key_col, feature_col in features_to_count:
        val = str(data[key_col])
        data[feature_col] = int(mappings.get(feature_col, {}).get(val, 0))

    # 4. Distance Engineering
    dist = float(user_input['distance'])
    data['DISTANCE'] = dist
    
    if dist <= 750: dist_cat = "SHORT"
    elif dist <= 1500: dist_cat = "MEDIUM"
    else: dist_cat = "LONG"
    
    data['DISTANCE_CAT'] = str(dist_cat)
    data['IS_SHORT_HAUL'] = 1 if dist_cat == "SHORT" else 0
    data['IS_MEDIUM_HAUL'] = 1 if dist_cat == "MEDIUM" else 0
    data['IS_LONG_HAUL'] = 1 if dist_cat == "LONG" else 0
    
    # Z-Score normalization
    mean_dist = lookups['means'].get('DISTANCE', 800)
    std_dist = lookups['means'].get('DISTANCE_STD', 600)
    data['DISTANCE_NORMALIZED'] = (dist - mean_dist) / std_dist
    
    # 5. EDA Context Features 
    data['AIR_TIME'] = dist * 0.12
    data['DISTANCE_GROUP'] = int(dist // 250)
    
    # Hub/Traffic logic 
    hubs = ['ATL', 'ORD', 'LAX', 'DFW', 'DEN', 'JFK', 'SFO', 'CLT', 'LAS', 'PHX']
    data['IS_HUB_ORIGIN'] = 1 if origin in hubs else 0
    data['IS_HUB_DEST'] = 1 if dest in hubs else 0
    data['IS_HUB_TO_HUB'] = 1 if (data['IS_HUB_ORIGIN'] and data['IS_HUB_DEST']) else 0
    
    data['IS_BUSY_ORIGIN'] = 1 if data['TRAIN_ORIGIN_COUNT'] > 1000 else 0
    data['IS_BUSY_DEST'] = 1 if data['TRAIN_DEST_COUNT'] > 1000 else 0
    data['IS_POPULAR_ROUTE'] = 1 if data['TRAIN_ROUTE_COUNT'] > 500 else 0
    
    data['ROUTE_POPULARITY'] = data['TRAIN_ROUTE_COUNT']
    data['ORIGIN_TRAFFIC'] = data['TRAIN_ORIGIN_COUNT']
    data['DEST_TRAFFIC'] = data['TRAIN_DEST_COUNT']
    data['CARRIER_VOLUME'] = data['TRAIN_UNIQUE_CARRIER_COUNT']
    
    # Zero out interactions
    for interaction in ['WEEKEND_SUMMER', 'FRIDAY_SUMMER', 'MONDAY_WINTER', 
                       'HUB_TO_HUB_SUMMER', 'POPULAR_ROUTE_WEEKEND', 'LONG_HAUL_WINTER']:
        data[interaction] = 0

    # --- CRITICAL FIX: REORDER COLUMNS ---
    # Create DataFrame and enforce order using feature_order list from metadata
    df = pd.DataFrame([data])
    
    # Ensure all expected columns exist (fill 0 if missing)
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
            
    # Reindex to match training order exactly
    df = df[feature_order]
    
    return df

# --- UI LAYOUT ---
st.title("🤖 AeroGuard: Intelligent Flight Assistant")
st.markdown("Predict flight delays using advanced machine learning.")

if model_type == "catboost":
    st.caption(f"🚀 Powered by **CatBoost** (Found: `model_artifacts_final/catboost_model.cbm`)")
else:
    st.caption(f"🧠 Powered by **Scikit-Learn** (Found: `model_artifacts_final/best_model.joblib`)")

# Chat Interface Simulation
with st.chat_message("assistant"):
    st.write("Hello! Configure your flight details in the sidebar to get a risk prediction.")

# Form Input
with st.sidebar:
    st.header("Flight Configuration")
    
    # Define options
    carrier_opts = ["DL", "AA", "UA", "WN", "EV", "B6", "NK", "AS", "F9"]
    
    carrier = st.selectbox("Carrier Code", carrier_opts)
    origin = st.text_input("Origin Airport (IATA)", "ATL", help="e.g. JFK, SFO, LHR").upper()
    dest = st.text_input("Destination Airport (IATA)", "JFK", help="e.g. LAX, MIA, ORD").upper()
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        date_val = st.date_input("Date", datetime.today())
    with col_d2:
        time_val = st.time_input("Departure Time", datetime.now().time())
        
    distance = st.number_input("Est. Distance (miles)", value=760, step=50)
    
    st.markdown("---")
    btn_predict = st.button("Analyze Risk", type="primary", use_container_width=True)

if btn_predict:
    # 1. Prepare Data
    user_inputs = {
        "carrier": carrier, "origin": origin, "dest": dest,
        "date": date_val, "time": time_val, "distance": distance
    }
    
    # 2. Feature Engineering
    try:
        # Pass feature_order to the function
        features_df = preprocess_input(user_inputs, lookups, feature_order)
        
        # 3. Predict
        prediction_prob = model.predict_proba(features_df)[0][1]
        
        # 4. Display Results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Delay Probability", f"{prediction_prob:.1%}")
            
        with col2:
            if prediction_prob < 0.30:
                st.success("✅ **Low Risk**\n\nFlight is likely to be on time.")
            elif prediction_prob < 0.50:
                st.warning("⚠️ **Moderate Risk**\n\nMonitor flight status closely.")
            else:
                st.error("🚨 **High Risk**\n\nSignificant likelihood of >15min delay.")
        
        # 5. Contextual Insights
        st.markdown("### 📊 Factor Analysis")
        
        # Safely get risk scores for display
        origin_risk = lookups['mappings'].get('ORIGIN_RISK', {}).get(origin, "N/A")
        dest_risk = lookups['mappings'].get('DEST_RISK', {}).get(dest, "N/A")
        
        c1, c2 = st.columns(2)
        c1.metric(f"{origin} Risk Factor", f"{origin_risk:.3f}" if isinstance(origin_risk, float) else origin_risk)
        c2.metric(f"{dest} Risk Factor", f"{dest_risk:.3f}" if isinstance(dest_risk, float) else dest_risk)
        
        with st.expander("View Engineered Features (Debug)"):
            st.dataframe(features_df)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Tip: Ensure your feature lookup JSON matches the columns expected by the model.")