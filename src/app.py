import streamlit as st
import pandas as pd
import openai
import json
from feature_store import FeatureStore, build_feature_row
from model_service import ModelService
from utils import build_top_factors

# --- CONFIGURATION ---
ARTIFACTS_DIR = "/Users/nikitha/Documents/flight-delay-prediction-ice/notebooks/model_artifacts_final"
FEATURE_STORE_PATH = f"{ARTIFACTS_DIR}/feature_store.json"

st.set_page_config(page_title="Flight Intelligence Hub", layout="wide")

# --- SERVICE LOADING ---
@st.cache_resource
def load_services():
    # Fix: Ensure variables are unpacked correctly
    store = FeatureStore.load(FEATURE_STORE_PATH)
    model = ModelService(ARTIFACTS_DIR)
    return store, model

store, model = load_services()

# --- HELPER: AI AGENT LOGIC ---
def run_ai_chat(user_input, api_key):
    client = openai.OpenAI(api_key=api_key)
    
    # Define the tool for the LLM to call your Python model logic
    tools = [{
        "type": "function",
        "function": {
            "name": "predict_delay",
            "description": "Calculate delay probability for a specific flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "carrier": {"type": "string"},
                    "origin": {"type": "string"},
                    "dest": {"type": "string"},
                    "dep_time": {"type": "integer", "description": "HHMM format"}
                },
                "required": ["carrier", "origin", "dest", "dep_time"]
            }
        }
    }]

    messages = [{"role": "system", "content": "You are a helpful flight assistant. Use the tool provided to check delay risks."},
                {"role": "user", "content": user_input}]

    response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            # Build feature row and predict
            row = build_feature_row(
                year=2026, quarter=1, month=1, day_of_month=20, day_of_week=1, # Default values
                unique_carrier=args['carrier'].upper(), origin=args['origin'].upper(), dest=args['dest'].upper(),
                origin_state_abr=None, dest_state_abr=None, dep_time=args['dep_time'],
                distance=None, air_time=None, distance_group=None, store=store
            )
            prob = float(model.predict_proba([row])[0])
            risk = model.risk_band(prob)
            
            # Send result back to LLM for natural language summary
            messages.append(msg)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps({"prob": prob, "risk": risk})})
            
            final_res = client.chat.completions.create(model="gpt-4o", messages=messages)
            return final_res.choices[0].message.content
    return msg.content

# --- UI LAYOUT ---
st.title("✈️ Flight Intelligence Hub")
tab1, tab2 = st.tabs(["💬 Conversational AI", "🏆 Smart Recommendations"])

# --- TAB 1: CONVERSATIONAL AI ---
with tab1:
    st.subheader("Flight Assistant Agent")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about a flight (e.g., 'Risk for AA123 JFK to LAX at 5pm?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar.")
        else:
            with st.chat_message("assistant"):
                res = run_ai_chat(prompt, api_key)
                st.markdown(res)
                st.session_state.messages.append({"role": "assistant", "content": res})

# --- TAB 2: RECOMMENDATION SYSTEM ---
with tab2:
    st.subheader("Intelligent Option Ranking")
    st.info("Rank flights based on a combination of delay risk, price, and duration.")

    # Input for multiple flights
    num_options = st.slider("Number of flight options to compare", 2, 5, 3)
    flight_data = []

    cols = st.columns(num_options)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Flight {i+1}**")
            c = st.text_input(f"Carrier", value="AA", key=f"car_{i}")
            o = st.text_input(f"Origin", value="JFK", key=f"ori_{i}")
            d = st.text_input(f"Dest", value="LAX", key=f"des_{i}")
            t = st.number_input(f"Dep Time", value=1200 + (i*200), key=f"tim_{i}")
            p = st.number_input(f"Price ($)", value=300.0 + (i*50), key=f"pri_{i}")
            
            row = build_feature_row(
                2026, 1, 1, 20, 1, c.upper(), o.upper(), d.upper(),
                None, None, int(t), None, None, None, store
            )
            flight_data.append({"id": i+1, "row": row, "price": p, "carrier": c})

    if st.button("Generate Recommendations"):
        results = []
        for f in flight_data:
            prob = float(model.predict_proba([f["row"]])[0])
            # Multi-objective score: lower is better (Risk + Price weight)
            score = (prob * 100) + (f["price"] / 10) 
            
            results.append({
                "Flight": f"Option {f['id']} ({f['carrier']})",
                "Delay Risk": f"{prob:.2%}",
                "Price": f"${f['price']}",
                "Recommendation Score": round(score, 2),
                "Factors": ", ".join(build_top_factors(f["row"]))
            })
        
        df = pd.DataFrame(results).sort_values("Recommendation Score")
        st.table(df)
        st.success(f"Top Recommendation: {df.iloc[0]['Flight']} (Best balance of price and reliability)")