import streamlit as st
import pandas as pd
from feature_store import FeatureStore, build_feature_row
from model_service import ModelService
from utils import build_top_factors

ARTIFACTS_DIR = "/Users/nikitha/Documents/flight-delay-prediction-ice/notebooks/model_artifacts_final"
FEATURE_STORE_PATH = "/Users/nikitha/Documents/flight-delay-prediction-ice/notebooks/model_artifacts_final/feature_store.json"

st.set_page_config(page_title="Flight Delay Intelligent Assistant", layout="wide")

@st.cache_resource
def load_services():
    store = FeatureStore.load(FEATURE_STORE_PATH)
    model = ModelService(ARTIFACTS_DIR)
    return store, model

store = None
model = load_services()


st.title("Flight Delay Intelligent Assistant")
st.caption("Interactive demo: delay risk prediction + recommendation based on trained ML model.")

tab1, tab2, tab3 = st.tabs(["Predict Delay Risk", "Recommend Best Option", "Conversational Interface"])

# ---------------------------
# TAB 1: Single prediction
# ---------------------------
with tab1:
    st.subheader("Predict delay risk for a flight")

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.number_input("Year", min_value=2017, max_value=2018, value=2018)
        quarter = st.number_input("Quarter", min_value=1, max_value=4, value=2)
        month = st.number_input("Month", min_value=1, max_value=12, value=4)
        day_of_month = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
        day_of_week = st.number_input("Day of Week (1=Mon ... 7=Sun)", min_value=1, max_value=7, value=7)

    with col2:
        unique_carrier = st.text_input("Carrier Code (e.g., AA, DL, UA)", value="AA")
        origin = st.text_input("Origin Airport (e.g., JFK)", value="JFK")
        dest = st.text_input("Destination Airport (e.g., LAX)", value="LAX")
        origin_state = st.text_input("Origin State (optional)", value="NY")
        dest_state = st.text_input("Destination State (optional)", value="CA")

    with col3:
        dep_time = st.number_input("Departure Time HHMM (optional)", min_value=0, max_value=2359, value=1830)
        distance = st.number_input("Distance (miles, optional)", min_value=0.0, value=2475.0)
        air_time = st.number_input("Air Time (minutes, optional)", min_value=0.0, value=360.0)
        distance_group = st.number_input("Distance Group (optional)", min_value=0, value=10)

    if st.button("Predict"):
        row = build_feature_row(
            year=year, quarter=quarter, month=month, day_of_month=day_of_month, day_of_week=day_of_week,
            unique_carrier=unique_carrier.strip().upper(),
            origin=origin.strip().upper(),
            dest=dest.strip().upper(),
            origin_state_abr=(origin_state.strip().upper() if origin_state else None),
            dest_state_abr=(dest_state.strip().upper() if dest_state else None),
            dep_time=int(dep_time) if dep_time else None,
            distance=float(distance) if distance else None,
            air_time=float(air_time) if air_time else None,
            distance_group=int(distance_group) if distance_group else None,
            store=store
        )

        p = float(model.predict_proba([row])[0])
        band = model.risk_band(p)
        alert = int(p >= model.threshold)

        left, right = st.columns([1, 2])
        with left:
            st.metric("Delay Probability", f"{p:.3f}")
            st.metric("Risk Band", band)
            st.metric("Alert Flag", str(alert))

        with right:
            st.write("Top contributing factors:")
            for f in build_top_factors(row):
                st.write(f"- {f}")

            st.write("Features used (subset):")
            st.dataframe(pd.DataFrame([ {k: row.get(k) for k in model.selected_features} ]))

# ---------------------------
# TAB 2: Recommendation system
# ---------------------------
with tab2:
    st.subheader("Compare options and recommend the lowest-delay-risk flight")
    st.caption("Enter multiple flight options and the app will rank them by predicted delay risk.")

    n = st.number_input("Number of options", min_value=2, max_value=6, value=3)
    options = []

    for i in range(int(n)):
        st.markdown(f"### Option {i+1}")
        c1, c2, c3 = st.columns(3)
        with c1:
            o_carrier = st.text_input(f"Carrier (Option {i+1})", value="AA", key=f"c_{i}")
            o_origin = st.text_input(f"Origin (Option {i+1})", value="JFK", key=f"o_{i}")
            o_dest = st.text_input(f"Dest (Option {i+1})", value="LAX", key=f"d_{i}")
        with c2:
            o_deptime = st.number_input(f"Dep Time HHMM (Option {i+1})", min_value=0, max_value=2359, value=700 + i*200, key=f"t_{i}")
            o_price = st.number_input(f"Price (Option {i+1})", min_value=0.0, value=300.0 + i*20, key=f"p_{i}")
        with c3:
            o_duration = st.number_input(f"Duration minutes (Option {i+1})", min_value=0, value=380 + i*10, key=f"dur_{i}")
            o_stops = st.number_input(f"Stops (Option {i+1})", min_value=0, max_value=3, value=0, key=f"s_{i}")

        row = build_feature_row(
            year=2018, quarter=2, month=4, day_of_month=15, day_of_week=7,
            unique_carrier=o_carrier.strip().upper(),
            origin=o_origin.strip().upper(),
            dest=o_dest.strip().upper(),
            origin_state_abr=None,
            dest_state_abr=None,
            dep_time=int(o_deptime),
            distance=None,
            air_time=None,
            distance_group=None,
            store=store
        )

        options.append({
            "id": f"opt{i+1}",
            "price": float(o_price),
            "duration": int(o_duration),
            "stops": int(o_stops),
            "row": row
        })

    if st.button("Rank options"):
        probs = model.predict_proba([o["row"] for o in options])

        ranked = []
        for o, p in zip(options, probs):
            ranked.append({
                "option_id": o["id"],
                "delay_probability": float(p),
                "risk_band": model.risk_band(float(p)),
                "alert": int(float(p) >= model.threshold),
                "price": o["price"],
                "duration_minutes": o["duration"],
                "stops": o["stops"],
                "top_factors": "; ".join(build_top_factors(o["row"]))
            })

        ranked.sort(key=lambda x: (x["delay_probability"], x["stops"], x["duration_minutes"], x["price"]))
        st.dataframe(pd.DataFrame(ranked))

# ---------------------------
# TAB 3: Conversational interface
# ---------------------------
with tab3:
    st.subheader("Conversational interface")
    st.caption("This is a lightweight interview demo. In production, you would connect a full LLM tool-calling layer.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_msg = st.text_input("Ask a question", placeholder="e.g., Will my JFK to LAX flight at 6 PM be delayed?")

    if st.button("Send"):
        st.session_state.chat.append(("user", user_msg))

        assistant_reply = (
            "I can estimate delay risk if you provide: date, origin, destination, carrier, and departure time. "
            "You can also compare multiple options in the Recommend tab."
        )

        st.session_state.chat.append(("assistant", assistant_reply))

    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")
