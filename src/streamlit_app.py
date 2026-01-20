# src/streamlit_app.py
"""
Flight Delay AI Agent - Streamlit Frontend
=========================================

This Streamlit app provides:
1) A beautiful UI to predict flight delay risk using your FastAPI backend
2) A chat assistant powered by OpenAI tool-calling that can:
   - predict delay risk
   - recommend best times
   - compare airlines
"""


from __future__ import annotations

import os
import json
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional
from math import radians, sin, cos, asin, sqrt

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

def setup_logging() -> logging.Logger:
    """
    Configure a rotating file logger for Streamlit.
    Streamlit reruns the script often, so we must avoid adding duplicate handlers.
    """
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "streamlit_app.log")

    logger = logging.getLogger("streamlit_app")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on Streamlit reruns
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger initialized")
    logger.info(f"Logging to: {log_path}")
    return logger

logger = setup_logging()

# ----- Config -------
load_dotenv(override=True)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ------- UI Setup --------
st.set_page_config(
    page_title="Flight Delay AI Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu, footer, header, .stDeployButton {visibility: hidden; display: none;}
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    .hero-container {
        text-align: center;
        padding: 2rem;
        margin-bottom: 1rem;
    }
    
    .hero-icon {
        font-size: 4rem;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.7);
    }
    
    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .badge {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 0.4rem 0.8rem;
        border-radius: 2rem;
        color: white;
        font-size: 0.8rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1.5rem;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        color: white;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 0.75rem !important;
        color: white !important;
    }
    
    .stSelectbox label, .stNumberInput label {
        color: rgba(255,255,255,0.8) !important;
        font-weight: 500 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 0.75rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 2px solid rgba(16, 185, 129, 0.4);
        border-radius: 1.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem auto;
        max-width: 400px;
    }
    
    .result-card.moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%);
        border-color: rgba(245, 158, 11, 0.4);
    }
    
    .result-card.high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
        border-color: rgba(239, 68, 68, 0.4);
    }
    
    .result-probability {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
    }
    
    .result-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.25rem;
    }
    
    .result-risk {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 2rem;
        font-weight: 600;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    
    .risk-low { background: #10b981; color: white; }
    .risk-moderate { background: #f59e0b; color: white; }
    .risk-high { background: #ef4444; color: white; }
    
    .reasons-list {
        background: rgba(255,255,255,0.05);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .reason-item {
        color: rgba(255,255,255,0.9);
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        font-size: 0.9rem;
    }
    
    .reason-item:last-child { border-bottom: none; }
    
    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 1rem;
        padding: 1.25rem;
        text-align: center;
    }
    
    .stat-icon { font-size: 1.75rem; margin-bottom: 0.5rem; }
    .stat-value { color: #10b981; font-size: 1.3rem; font-weight: 700; }
    .stat-label { color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-top: 0.25rem; }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 1rem;
        padding: 0.5rem;
        gap: 0.5rem;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255,255,255,0.7);
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    .stChatMessage { background: rgba(255,255,255,0.05) !important; border-radius: 1rem !important; }
    
    p, span, label { color: rgba(255,255,255,0.9); }
    h1, h2, h3, h4 { color: white !important; }
    
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #10b981;
        font-size: 0.85rem;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .flight-route {
        text-align: center;
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        margin: 1rem 0;
        padding: 0.75rem;
        background: rgba(255,255,255,0.05);
        border-radius: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# -------- Backend Client ---------

class BackendClient:
    class BackendClient:
        """
        Thin HTTP client for FastAPI backend with logging + latency tracking.
        """

        def __init__(self, base_url: str):
            self.base_url = base_url.rstrip("/")
            logger.info(f"BackendClient initialized | base_url={self.base_url}")

        def _post(self, path: str, payload: dict) -> dict:
            request_id = str(uuid.uuid4())
            url = f"{self.base_url}{path}"
            start = time.time()

            logger.info(
                f"[{request_id}] POST {path} | payload={payload}"
            )

            try:
                resp = requests.post(url, json=payload, timeout=30)
                latency_ms = int((time.time() - start) * 1000)

                if resp.status_code >= 400:
                    logger.warning(
                        f"[{request_id}] POST {path} FAILED "
                        f"| status={resp.status_code} "
                        f"| latency_ms={latency_ms} "
                        f"| response={resp.text}"
                    )
                    return {"ok": False, "error": resp.text}

                data = resp.json()
                logger.info(
                    f"[{request_id}] POST {path} OK "
                    f"| latency_ms={latency_ms}"
                )

                return {"ok": True, "data": data}

            except Exception as e:
                latency_ms = int((time.time() - start) * 1000)
                logger.exception(
                    f"[{request_id}] POST {path} EXCEPTION "
                    f"| latency_ms={latency_ms} "
                    f"| error={str(e)}"
                )
                return {"ok": False, "error": str(e)}

        def health(self) -> dict:
            request_id = str(uuid.uuid4())
            url = f"{self.base_url}/health"

            try:
                resp = requests.get(url, timeout=5)
                logger.info(f"[{request_id}] GET /health OK")
                return {"ok": True, **resp.json()}
            except Exception as e:
                logger.warning(
                    f"[{request_id}] GET /health FAILED | error={str(e)}"
                )
                return {"ok": False, "status": "offline"}

        def predict(self, args: dict) -> dict:
            return self._post("/predict", args)

        def recommend_times(self, args: dict) -> dict:
            return self._post("/recommend/times", args)

        def recommend_airlines(self, args: dict) -> dict:
            return self._post("/recommend/airlines", args)

# ----- OpenAI Agent -----

class OpenAIAgent:
    SYSTEM_PROMPT = """You are a flight delay assistant for a travel app.

    Your job is to:
    1) Extract flight details from the user message.
    2) Call the right tool.
    3) Summarize the output in plain language.

    IMPORTANT INPUT RULES
    - Airports MUST be IATA airport codes like JFK, LAX, ORD (3 letters).
    - Carriers MUST be airline codes like AA, DL, UA, WN, B6, AS, NK, F9.
    - dep_hour MUST be an integer 0–23 in 24-hour format.
    - day_of_week MUST be 1–7 (1=Mon … 7=Sun).
    - month MUST be 1–12.

    If the user gives city names (e.g., “Chicago”, “New York”), ask ONE clarifying question OR make a clear assumption and state it.
    Examples:
    - Chicago defaults to ORD unless user says Midway (MDW).
    - New York defaults to JFK unless user says LaGuardia (LGA) or Newark (EWR).

    TIME PARSING
    - If the user says “morning”, use 8.
    - “afternoon” → 14
    - “evening” → 18
    - “late night” → 21
    - If the user gives “5 PM”, convert to 17.

    SUPPORTED CARRIER CODES
    AA=American, DL=Delta, UA=United, WN=Southwest, B6=JetBlue, AS=Alaska,
    NK=Spirit, F9=Frontier

    RISK LABELS (must match the app)
    - LOW: < 18%
    - MODERATE: 18%–30%
    - HIGH: ≥ 30%

    Only ask ONE clarifying question if required info is missing.
    """

    TOOLS = [
        {"type": "function", "function": {"name": "predict_flight", "description": "Predict delay",
            "parameters": {"type": "object", "properties": {
                "origin": {"type": "string"}, "destination": {"type": "string"},
                "carrier": {"type": "string"}, "dep_hour": {"type": "integer"},
                "day_of_week": {"type": "integer"}, "month": {"type": "integer"},
                "distance": {"type": "number", "description": "Optional route distance in miles"}},
                "required": ["origin", "destination", "carrier", "dep_hour", "day_of_week", "month"]}}},
        {"type": "function", "function": {"name": "recommend_best_times", "description": "Best times",
            "parameters": {"type": "object", "properties": {
                "origin": {"type": "string"}, "destination": {"type": "string"},
                "carrier": {"type": "string"}, "day_of_week": {"type": "integer"}, "month": {"type": "integer"}},
                "required": ["origin", "destination", "carrier", "day_of_week", "month"]}}},
        {"type": "function", "function": {"name": "recommend_best_airlines", "description": "Compare airlines",
            "parameters": {"type": "object", "properties": {
                "origin": {"type": "string"}, "destination": {"type": "string"},
                "dep_hour": {"type": "integer"}, "day_of_week": {"type": "integer"}, "month": {"type": "integer"}},
                "required": ["origin", "destination", "dep_hour", "day_of_week", "month"]}}}
    ]

    def __init__(self, api_key: str, backend: BackendClient):
        self.client = OpenAI(api_key=api_key)
        self.backend = backend

    # -----------------------------
    # Lightweight input normalizers
    # -----------------------------
    CITY_TO_DEFAULT_AIRPORT = {
        "CHICAGO": "ORD",
        "NEW YORK": "JFK",
        "NYC": "JFK",
        "LOS ANGELES": "LAX",
        "LA": "LAX",
        "SAN FRANCISCO": "SFO",
        "SF": "SFO",
        "MIAMI": "MIA",
        "BOSTON": "BOS",
        "SEATTLE": "SEA",
        "DENVER": "DEN",
        "DALLAS": "DFW",
        "HOUSTON": "IAH",
        "ATLANTA": "ATL",
        "LAS VEGAS": "LAS",
        "ORLANDO": "MCO",
        "PHOENIX": "PHX",
    }

    AIRLINE_NAME_TO_CODE = {
        "AMERICAN": "AA",
        "AMERICAN AIRLINES": "AA",
        "DELTA": "DL",
        "DELTA AIR LINES": "DL",
        "UNITED": "UA",
        "UNITED AIRLINES": "UA",
        "SOUTHWEST": "WN",
        "SOUTHWEST AIRLINES": "WN",
        "JETBLUE": "B6",
        "JETBLUE AIRWAYS": "B6",
        "ALASKA": "AS",
        "ALASKA AIRLINES": "AS",
        "SPIRIT": "NK",
        "SPIRIT AIRLINES": "NK",
        "FRONTIER": "F9",
        "FRONTIER AIRLINES": "F9",
    }

    @staticmethod
    def _to_iata(value: Any) -> str:
        """Normalize an origin/destination value into a likely IATA code."""
        if value is None:
            return ""
        s = str(value).strip().upper()
        # Already an airport code
        if len(s) == 3 and s.isalpha():
            return s
        return OpenAIAgent.CITY_TO_DEFAULT_AIRPORT.get(s, s)

    @staticmethod
    def _to_carrier_code(value: Any) -> str:
        if value is None:
            return ""
        s = str(value).strip().upper()
        # Already a carrier code
        if len(s) in (2, 3) and s.replace(" ", "").isalnum():
            # E.g. AA, DL, UA, WN, B6
            return s
        return OpenAIAgent.AIRLINE_NAME_TO_CODE.get(s, s)

    @staticmethod
    def _clamp_hour(h: Any) -> int:
        """Ensure dep_hour is an int in [0, 23]."""
        try:
            h_int = int(h)
        except Exception:
            return 12
        return max(0, min(23, h_int))

    @staticmethod
    def _validate_required(args: dict, required: List[str]) -> List[str]:
        missing = []
        for k in required:
            if k not in args or args[k] in (None, ""):
                missing.append(k)
        return missing

    def _normalize_tool_args(self, name: str, args: dict) -> dict:
        """Sanitize tool arguments before calling backend.

        This is defensive programming to reduce bad predictions caused by
        ambiguous user text or imperfect tool-call arguments.
        """
        clean = dict(args or {})

        if "origin" in clean:
            clean["origin"] = self._to_iata(clean["origin"])
        if "destination" in clean:
            clean["destination"] = self._to_iata(clean["destination"])
        if "carrier" in clean:
            clean["carrier"] = self._to_carrier_code(clean["carrier"])  # AA/DL/UA...
        if "dep_hour" in clean:
            clean["dep_hour"] = self._clamp_hour(clean["dep_hour"])  # 0–23

        # The backend can accept an optional `distance` field.
        if name == "predict_flight" and "distance" not in clean:
            try:
                est = haversine_miles(clean.get("origin", ""), clean.get("destination", ""))
                if est is not None:
                    clean["distance"] = float(round(est, 1))
            except Exception:
                pass

        # Coerce ints
        for k in ("day_of_week", "month"):
            if k in clean and clean[k] not in (None, ""):
                try:
                    clean[k] = int(clean[k])
                except Exception:
                    pass

        # Basic bounds
        if "day_of_week" in clean:
            clean["day_of_week"] = max(1, min(7, int(clean.get("day_of_week", 1))))
        if "month" in clean:
            clean["month"] = max(1, min(12, int(clean.get("month", 1))))

        return clean

    def _dispatch(self, name: str, args: dict) -> dict:
        args = self._normalize_tool_args(name, args)

        # Debug hooks for UI
        st.session_state.debug_last_call = {"tool": name, "args": args}

        # Validate required fields to prevent silent bad calls.
        req_map = {
            "predict_flight": ["origin", "destination", "carrier", "dep_hour", "day_of_week", "month"],
            "recommend_best_times": ["origin", "destination", "carrier", "day_of_week", "month"],
            "recommend_best_airlines": ["origin", "destination", "dep_hour", "day_of_week", "month"],
        }
        missing = self._validate_required(args, req_map.get(name, []))
        if missing:
            return {
                "ok": False,
                "error": f"Missing required fields for {name}: {', '.join(missing)}. "
                         f"Please specify airports (IATA codes), airline, day, month, and time.",
            }

        if name == "predict_flight":
            resp = self.backend.predict(args)
            st.session_state.debug_last_resp = resp
            return resp
        elif name == "recommend_best_times":
            resp = self.backend.recommend_times(args)
            st.session_state.debug_last_resp = resp
            return resp
        elif name == "recommend_best_airlines":
            resp = self.backend.recommend_airlines(args)
            st.session_state.debug_last_resp = resp
            return resp
        return {"ok": False}

    def chat(self, user_text: str, history: List[Dict]) -> tuple:
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend([{"role": h["role"], "content": h["content"]} for h in history])
        messages.append({"role": "user", "content": user_text})

        resp = self.client.chat.completions.create(model=OPENAI_MODEL, messages=messages,
                                                    tools=self.TOOLS, tool_choice="auto", temperature=0.3)
        msg = resp.choices[0].message
        tool_results = []

        if msg.tool_calls:
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                result = self._dispatch(call.function.name, args)
                tool_results.append({"tool": call.function.name, "result": result})
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
            resp2 = self.client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.3)
            return resp2.choices[0].message.content, tool_results
        return msg.content, tool_results


# =============================
# Initialize
# =============================
backend = BackendClient(BACKEND_URL)
if "messages" not in st.session_state: st.session_state.messages = []
if "prediction_result" not in st.session_state: st.session_state.prediction_result = None
if "debug_last_call" not in st.session_state: st.session_state.debug_last_call = None
if "debug_last_resp" not in st.session_state: st.session_state.debug_last_resp = None

# Data
AIRLINES = {"DL": "Delta Air Lines", "AA": "American Airlines", "UA": "United Airlines",
            "WN": "Southwest Airlines", "B6": "JetBlue Airways", "AS": "Alaska Airlines",
            "NK": "Spirit Airlines", "F9": "Frontier Airlines"}
AIRPORTS = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO", "LAS", "SEA", "MCO",
            "EWR", "BOS", "MSP", "DTW", "PHL", "LGA", "FLL", "BWI", "DCA", "SLC",
            "MIA", "TPA", "PHX", "SAN", "IAH", "CLT", "AUS", "BNA", "PDX", "STL"]
DAYS = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
MONTHS = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
          7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}

AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277),
    "ORD": (41.9742, -87.9073),
    "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737),
    "LAX": (33.9416, -118.4085),
    "JFK": (40.6413, -73.7781),
    "SFO": (37.6213, -122.3790),
    "LAS": (36.0840, -115.1537),
    "SEA": (47.4502, -122.3088),
    "MCO": (28.4312, -81.3081),
    "EWR": (40.6895, -74.1745),
    "BOS": (42.3656, -71.0096),
    "MSP": (44.8848, -93.2223),
    "DTW": (42.2162, -83.3554),
    "PHL": (39.8729, -75.2437),
    "LGA": (40.7769, -73.8740),
    "FLL": (26.0726, -80.1527),
    "BWI": (39.1754, -76.6684),
    "DCA": (38.8512, -77.0402),
    "SLC": (40.7899, -111.9791),
    "MIA": (25.7959, -80.2870),
    "TPA": (27.9755, -82.5332),
    "PHX": (33.4342, -112.0116),
    "SAN": (32.7338, -117.1933),
    "IAH": (29.9902, -95.3368),
    "CLT": (35.2144, -80.9473),
    "AUS": (30.1975, -97.6664),
    "BNA": (36.1263, -86.6774),
    "PDX": (45.5898, -122.5951),
    "STL": (38.7487, -90.3700),
}

def haversine_miles(a: str, b: str) -> Optional[float]:
    """Great-circle distance between two IATA airports in miles."""
    a = (a or "").upper().strip()
    b = (b or "").upper().strip()
    if a not in AIRPORT_COORDS or b not in AIRPORT_COORDS:
        return None
    lat1, lon1 = AIRPORT_COORDS[a]
    lat2, lon2 = AIRPORT_COORDS[b]
    # haversine
    r = 3958.7613  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(h))

# =============================
# Hero Section
# =============================

# Sidebar controls
st.sidebar.title("⚙️ Settings")
st.sidebar.caption("Use these to debug predictions and reduce ambiguous inputs.")
debug_mode = st.sidebar.checkbox("Show debug details", value=False)
st.sidebar.markdown("---")

st.markdown("""
<div class="hero-container">
    <div class="hero-icon">✈️</div>
    <h1 class="hero-title">Flight Delay Predictor</h1>
    <p class="hero-subtitle">AI-Powered predictions using Machine Learning</p>
    <div class="hero-badges">
        <span class="badge">🤖 CatBoost ML</span>
        <span class="badge">💬 OpenAI Chat</span>
        <span class="badge">⚡ Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)

health = backend.health()
if health.get("ok"):
    st.markdown(f'<div style="text-align:center;margin-bottom:1rem;"><span class="status-online"><span class="status-dot"></span> Model Online</span></div>', unsafe_allow_html=True)
else:
    st.warning("Backend is offline. Start your FastAPI backend before using the app.")

# =============================
# Tabs
# =============================
tab1, tab2 = st.tabs(["🎯 Predict Delay", "💬 AI Chat"])

# =============================
# Tab 1: Quick Predict
# =============================
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">🔮 Enter Flight Details</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        origin = st.selectbox("🛫 From", AIRPORTS, index=AIRPORTS.index("JFK"))
        carrier_code = st.selectbox("✈️ Airline", list(AIRLINES.keys()), format_func=lambda x: f"{x} - {AIRLINES[x]}")
    with col2:
        destination = st.selectbox("🛬 To", AIRPORTS, index=AIRPORTS.index("LAX"))
        dep_hour = st.selectbox("🕐 Time", list(range(5, 24)), index=9, format_func=lambda x: f"{x:02d}:00")
    with col3:
        day_of_week = st.selectbox("📅 Day", list(DAYS.keys()), index=4, format_func=lambda x: DAYS[x])
        month = st.selectbox("📆 Month", list(MONTHS.keys()), index=6, format_func=lambda x: MONTHS[x])

    # Distance (helps model realism)
    est_distance = haversine_miles(origin, destination)
    distance_to_send = None
    with st.expander("Advanced options (improves accuracy)", expanded=False):
        st.caption("The backend model performs better when route distance is provided. "
                   "We can estimate distance from airport coordinates.")
        use_est = st.checkbox("Use estimated route distance", value=True)
        if est_distance is not None:
            st.write(f"Estimated distance: **{est_distance:.0f} miles**")
        else:
            st.warning("Distance estimate unavailable for this airport pair.")
        manual_distance = st.number_input("Override distance (miles) (optional)", min_value=0.0,
                                          value=float(round(est_distance, 1)) if est_distance else 0.0,
                                          step=10.0)

        if manual_distance and manual_distance > 0:
            distance_to_send = float(round(manual_distance, 1))
        elif use_est and est_distance is not None:
            distance_to_send = float(round(est_distance, 1))

    st.markdown(
        "<small style='color:rgba(255,255,255,0.65)'>"
        "Note: This app predicts delay risk from historical patterns (carrier/route/time). "
        "It does not use live weather or ATC conditions." 
        "</small>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button("🔮 Predict Delay Risk", use_container_width=True)

    if predict_clicked:
        with st.spinner("🔄 Analyzing..."):
            payload = {"origin": origin, "destination": destination, "carrier": carrier_code,
                       "dep_hour": dep_hour, "day_of_week": day_of_week, "month": month}
            if distance_to_send is not None:
                payload["distance"] = distance_to_send
                
            st.session_state.last_payload = payload

            # Debug hooks
            st.session_state.debug_last_call = {"tool": "ui_predict", "args": payload}

            result = backend.predict(payload)
            st.session_state.debug_last_resp = result
            st.session_state.prediction_result = result

    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        if result.get("ok"):
            data = result["data"]
            prob, risk = data["delay_probability"], data["risk_level"]
            card_class = "result-card" + (" moderate" if risk == "MODERATE" else " high" if risk == "HIGH" else "")
            risk_class = {"LOW": "risk-low", "MODERATE": "risk-moderate", "HIGH": "risk-high"}.get(risk, "risk-low")
            
            lp = st.session_state.get("last_payload") or {}
            o = lp.get("origin", origin)
            d = lp.get("destination", destination)
            c = lp.get("carrier", carrier_code)
            h = lp.get("dep_hour", dep_hour)
            dow = lp.get("day_of_week", day_of_week)
            m = lp.get("month", month)

            st.markdown(f"""
            <div class="{card_class}">
                <div class="result-probability">{prob*100:.1f}%</div>
                <div class="result-label">Delay Probability</div>
                <span class="result-risk {risk_class}">{risk} RISK</span>
            </div>
            <div class="flight-route">
                <strong>{o}</strong> ✈️ <strong>{d}</strong> • 
                {AIRLINES.get(c, c)} • {DAYS[dow]} {h:02d}:00 • {MONTHS[m]}
            </div>
            """, unsafe_allow_html=True)
            
            reasons = data.get("top_reasons", [])
            # UI-side clean-up to avoid confusing contradictions from heuristic explanations.
            if reasons:
                cleaned = []
                for r in reasons:
                    if risk in ("MODERATE", "HIGH") and "morning flights typically have the lowest delay risk" in r.lower():
                        # Avoid "morning is low-risk" statement when risk isn't low
                        continue
                    if risk in ("MODERATE", "HIGH") and "no major risk factors" in r.lower():
                        continue
                    cleaned.append(r)
                if risk in ("MODERATE", "HIGH") and not cleaned:
                    cleaned = ["Multiple smaller factors contribute to the overall delay risk."]
                reasons = cleaned
            if reasons:
                st.markdown('<div class="reasons-list"><strong style="color:white;">📊 Contributing Factors:</strong>', unsafe_allow_html=True)
                for r in reasons:
                    st.markdown(f'<div class="reason-item">• {r}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Debug panel
            if debug_mode:
                with st.expander("🔎 Debug details", expanded=False):
                    st.write("**Request payload**")
                    st.json(st.session_state.debug_last_call)
                    st.write("**Backend response**")
                    st.json(st.session_state.debug_last_resp)
        else:
            st.error(f"❌ {result.get('error', 'Prediction failed')}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Stats
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    stats = [("🌅", "6-9 AM", "Best Time"), ("📅", "Saturday", "Best Day"), 
             ("🍂", "November", "Best Month"), ("✈️", "Alaska/Delta", "Best Airlines")]
    for col, (icon, value, label) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-icon">{icon}</div><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

# =============================
# Tab 2: AI Chat
# =============================
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="card-title">💬 Ask AI Assistant</h3>', unsafe_allow_html=True)
    
    if not OPENAI_API_KEY:
        with st.expander("🔑 Set OpenAI API Key"):
            key = st.text_input("API Key", type="password")
            if key: OPENAI_API_KEY = key; st.success("✅ Set!"); st.rerun()
    
    col1, col2, col3 = st.columns(3)
    if col1.button("🕐 Best times?", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "Best times to fly JFK to LAX?"}); st.rerun()
    if col2.button("✈️ Compare airlines", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "Which airline has lowest delays ORD to MIA?"}); st.rerun()
    if col3.button("💡 Delay tips", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "Tips to avoid flight delays?"}); st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Short prompt tips to reduce ambiguous city parsing
    with st.expander("✍️ Prompt tips (for more accurate answers)", expanded=False):
        st.markdown(
            """
            - Use **airport codes** (e.g., `ORD → JFK`) instead of city names.
            - Include **day, month, and departure time**.
            - Example: *"Predict delay risk for AA ORD to JFK on Wednesday at 08:00 in June"*
            """
        )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about flights..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            if OPENAI_API_KEY:
                with st.spinner("🤔 Thinking..."):
                    try:
                        agent = OpenAIAgent(OPENAI_API_KEY, backend)
                        response, tools = agent.chat(prompt, st.session_state.messages[:-1])
                        for t in tools:
                            if t["result"].get("ok"):
                                d = t["result"]["data"]
                                if "delay_probability" in d: st.info(f"📊 {d['delay_probability']*100:.1f}% ({d['risk_level']})")
                                elif "best_hours" in d: st.info(f"🕐 Best: {', '.join([h['time'] for h in d['best_hours'][:3]])}")
                                elif "best_carriers" in d: st.info(f"✈️ Best: {', '.join([c['carrier'] for c in d['best_carriers'][:3]])}")
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                        if debug_mode:
                            with st.expander("🔎 Debug details", expanded=False):
                                st.write("**Last tool call**")
                                st.json(st.session_state.debug_last_call)
                                st.write("**Last backend response**")
                                st.json(st.session_state.debug_last_resp)
                    except Exception as e: st.error(str(e))
            else: st.warning("⚠️ Set OpenAI API key above")
    
    if st.session_state.messages and st.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;padding:2rem;color:rgba(255,255,255,0.4);font-size:0.8rem;">Built with Streamlit • CatBoost • OpenAI</div>', unsafe_allow_html=True)