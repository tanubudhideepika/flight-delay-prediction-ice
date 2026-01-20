# src/app.py 
"""
Flight Delay Predictor API (FastAPI)
===================================

This service exposes three main endpoints:
- POST /predict           : Predict delay probability + risk band for a flight
- POST /recommend/times   : Recommend best departure hours for a given route/carrier/date
- POST /recommend/airlines: Recommend best carriers for a given route/time/date

Key design goals
----------------
1) Deterministic, training-consistent feature building
2) Calibrated probabilities (Platt scaling) to avoid overconfident outputs
3) Clear separation of concerns:
   - Config
   - Artifact loading (model, metadata, lookups, calibrator)
   - Feature building
   - Prediction service (business logic)
   - API layer (FastAPI endpoints)
4) Production-friendly logging (rotating file + console)

Environment variables
---------------------
Required:
- META_PATH       : Path to model_metadata.json
- LOOKUP_PATH     : Path to lookup.json (risk maps, counts, distance stats)
- CB_PATH         : Path to CatBoost model (.cbm)
- ART_DIR         : Artifact directory (optional, used in notebooks)

Optional:
- CALIB_PATH      : Path to platt_calibrator.joblib (Platt scaling)
- LOG_DIR         : Directory for logs (default: "logs")
"""


from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---- LOGGING -----
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "flight_delay_app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("flight_delay_app")


# ----- CONFIG -----
@dataclass(frozen=True)
class AppConfig:
    """Configuration loaded from environment variables."""
    meta_path: str
    lookup_path: str
    cb_path: Optional[str]
    joblib_path: Optional[str]
    calib_path: Optional[str]

    @staticmethod
    def from_env() -> "AppConfig":
        load_dotenv(override=True)
        return AppConfig(
            meta_path=os.getenv("META_PATH", ""),
            lookup_path=os.getenv("LOOKUP_PATH", ""),
            cb_path=os.getenv("CB_PATH"),
            joblib_path=os.getenv("JOBLIB_PATH"),
            calib_path=os.getenv("CALIB_PATH")
        )

# ------- API Schemas ----------
class PredictionRequest(BaseModel):
    origin: str
    destination: str
    carrier: str
    dep_hour: int
    day_of_week: int
    month: int
    distance: Optional[float] = None
class PredictionResponse(BaseModel):
    delay_probability: float
    risk_level: str
    threshold: float
    top_reasons: List[str]
    input_features_used: Dict[str, Any]
class RecommendTimesRequest(BaseModel):
    origin: str
    destination: str
    carrier: str
    day_of_week: int
    month: int
    hours: List[int] = Field(default_factory=lambda: list(range(5, 24)))
class RecommendAirlinesRequest(BaseModel):
    origin: str
    destination: str
    dep_hour: int
    day_of_week: int
    month: int
    carriers: List[str] = Field(default_factory=lambda: ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"])

# ------- ARTIFACTS STORE ---------
class ArtifactStore:
    """
    Loads all artifacts needed by the API:
    - metadata (selected features, operating threshold, etc.)
    - lookups (risk maps, counts, distance mean/std)
    - CatBoost model
    - optional Platt calibrator (sklearn LogisticRegression on log-odds)
    """
    
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.metadata: Dict[str, Any] = {}
        self.lookups: Dict[str, Any] = {}

        self.model: Optional[CatBoostClassifier] = None
        self.calibrator = None  # sklearn LogisticRegression

        self.model_type: str = "unknown"
        self.cat_feature_indices: List[int] = []
        self.feature_names: List[str] = []

    def load(self) -> None:
        logger.info("Loading artifacts....")
        
        # --- metadata ---
        if self.cfg.meta_path and os.path.exists(self.cfg.meta_path):
            with open(self.cfg.meta_path, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata: {self.cfg.meta_path}")
        else:
            logger.warning(f"Metadata not found: {self.cfg.meta_path}")

        # --- lookups ---
        if self.cfg.lookup_path and os.path.exists(self.cfg.lookup_path):
            with open(self.cfg.lookup_path, "r") as f:
                self.lookups = json.load(f)
            logger.info(f"Loaded lookups: {self.cfg.lookup_path}")
        else:
            logger.warning(f"Lookups not found: {self.cfg.lookup_path}")

        # --- calibrator (optional) ---
        if self.cfg.calib_path and os.path.exists(self.cfg.calib_path):
            try:
                self.calibrator = joblib.load(self.cfg.calib_path)
                logger.info(f"Loaded Platt calibrator: {self.cfg.calib_path}")
            except Exception as e:
                logger.warning(f"Could not load calibrator ({self.cfg.calib_path}): {e}")
                self.calibrator = None
        else:
            logger.info("No calibrator configured or file missing (CALIB_PATH).")

        # --- model (CatBoost) ---
        if self.cfg.cb_path and os.path.exists(self.cfg.cb_path):
            self.model = CatBoostClassifier()
            self.model.load_model(self.cfg.cb_path)
            self.model_type = "catboost"
            logger.info(f"Loaded CatBoost model: {self.cfg.cb_path}")

            # Attempt to fetch feature metadata from model
            try:
                self.cat_feature_indices = list(self.model.get_cat_feature_indices())
                self.feature_names = list(self.model.feature_names_)
                logger.info(f"Model features: {len(self.feature_names)} total")
                logger.info(f"Cat feature indices: {self.cat_feature_indices}")
            except Exception as e:
                logger.warning(f"Could not get model feature metadata: {e}")
        else:
            logger.error(f"CatBoost model not found: {self.cfg.cb_path}")

# -------- Feature Builder ----------- 
class FeatureBuilder:
    """
    Builds a feature row (DataFrame with 1 row) that matches training conventions.

    Key conventions (as used in your training/FE pipeline):
    - ROUTE          : "ORIGIN-DEST" (hyphen) e.g., "MCO-PHL"
    - CARRIER_ROUTE  : "CARRIER_ROUTE" with underscore e.g., "WN_MCO-PHL"
    - DEP_HOUR_BIN   : "0-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23"
    - Risk/count lookups keyed by ORIGIN/DEST/ROUTE/CARRIER_ROUTE/UNIQUE_CARRIER
    """
    
    # Known categorical columns from the training code
    CATEGORICAL_COLS = {
        'SEASON', 'DEP_HOUR_BIN', 'DISTANCE_CAT', 
        'UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'ROUTE', 'CARRIER_ROUTE',
        'ORIGIN_STATE_ABR', 'DEST_STATE_ABR'
    }
    
    HUB_AIRPORTS = {'ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'JFK', 'SFO', 'CLT', 'IAH', 'PHX'}
    BUSY_AIRPORTS = {'ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'JFK', 'LGA', 'EWR', 'SFO', 'BOS'}
    
    # State mappings for common airports
    AIRPORT_STATE = {
        'ATL': 'GA', 'ORD': 'IL', 'DFW': 'TX', 'DEN': 'CO', 'LAX': 'CA',
        'JFK': 'NY', 'SFO': 'CA', 'LAS': 'NV', 'SEA': 'WA', 'MCO': 'FL',
        'EWR': 'NJ', 'BOS': 'MA', 'MSP': 'MN', 'DTW': 'MI', 'PHL': 'PA',
        'LGA': 'NY', 'FLL': 'FL', 'BWI': 'MD', 'DCA': 'VA', 'SLC': 'UT',
        'MIA': 'FL', 'TPA': 'FL', 'PHX': 'AZ', 'SAN': 'CA', 'IAH': 'TX',
        'CLT': 'NC', 'AUS': 'TX', 'BNA': 'TN', 'PDX': 'OR', 'STL': 'MO',
        'HOU': 'TX', 'DAL': 'TX', 'MDW': 'IL', 'RDU': 'NC', 'SNA': 'CA',
        'SJC': 'CA', 'OAK': 'CA', 'SMF': 'CA', 'SAT': 'TX', 'MCI': 'MO',
        'IND': 'IN', 'CLE': 'OH', 'PIT': 'PA', 'CMH': 'OH', 'CVG': 'KY',
        'JAX': 'FL', 'RSW': 'FL', 'PBI': 'FL', 'MEM': 'TN', 'OKC': 'OK',
        'MSY': 'LA', 'ABQ': 'NM', 'TUS': 'AZ', 'ELP': 'TX', 'ONT': 'CA',
        'BUR': 'CA', 'HNL': 'HI', 'OGG': 'HI', 'ANC': 'AK',
    }

    def __init__(self, store: ArtifactStore):
        self.store = store
        self.metadata = store.metadata
        self.lookups = store.lookups
        self.selected = self.metadata.get("selected_features", [])
        
        self.mappings = self.lookups.get("mappings", {})
        self.defaults = self.lookups.get("defaults", {})
        self.means = self.lookups.get("means", {})
        
        # Build set of categorical column names from model indices
        self.cat_cols_from_model: set[str] = set()
        if store.feature_names and store.cat_feature_indices:
            for idx in store.cat_feature_indices:
                if 0 <= idx < len(store.feature_names):
                    self.cat_cols_from_model.add(store.feature_names[idx])

        logger.info(f"Categorical columns (from model): {sorted(self.cat_cols_from_model)}")

    def _map_val(self, feature_name: str, key: str):
        """Lookup risk/count values; use defaults for unknown keys."""
        mp = self.mappings.get(feature_name, {})
        
        # Counts default to 0; risks default to a base-ish rate
        if feature_name.endswith("_COUNT") or feature_name.startswith("TRAIN_"):
            default = self.defaults.get(feature_name, 0)  # counts default to 0
        else:
            default = self.defaults.get(feature_name, 0.15)  # risks default to base rate

        return mp.get(key, default)
    
    @staticmethod
    def _get_season(self, month: int) -> str:
        """Get season name matching training data format."""
        if month in (12, 1, 2):
            return "Winter"
        if month in (3, 4, 5):
            return "Spring"
        if month in (6, 7, 8):
            return "Summer"
        return "Fall"
    
    @staticmethod
    def _get_dep_hour_bin(self, hour: int) -> str:
        """
        FIXED: Return DEP_HOUR_BIN matching training data format.
        Training used pd.cut with bins like "0-5", "6-8", etc.
        """
        if hour <= 5: return "0-5"
        elif hour <= 8: return "6-8"
        elif hour <= 11: return "9-11"
        elif hour <= 14: return "12-14"
        elif hour <= 17: return "15-17"
        elif hour <= 20: return "18-20"
        return "21-23"
    
    @staticmethod
    def _get_distance_cat(self, distance: float) -> str:
        """Get distance category matching training data format."""
        if distance < 500: return "Short"
        elif distance < 1500: return "Medium"
        return "Long"

    def build(self, req: PredictionRequest) -> pd.DataFrame:
        """Build a single-row DataFrame of model features."""
        origin = req.origin.upper().strip()
        dest = req.destination.upper().strip()
        carrier = req.carrier.upper().strip()

        month = int(req.month)
        day_of_week = int(req.day_of_week)
        dep_hour = int(req.dep_hour)

        route = f"{origin}-{dest}"
        carrier_route = f"{carrier}_{route}"

        # distance normalization
        dist_mean = float(self.means.get("DISTANCE", 662.97))
        dist_std = float(self.means.get("DISTANCE_STD", 472.46)) or 1.0
        distance = float(req.distance) if req.distance is not None else dist_mean
        distance_norm = (distance - dist_mean) / dist_std

        season = self._get_season(month)
        dep_hour_bin = self._get_dep_hour_bin(dep_hour)
        distance_cat = self._get_distance_cat(distance)
        quarter = (month - 1) // 3 + 1

        origin_state = self.AIRPORT_STATE.get(origin, "NA")
        dest_state = self.AIRPORT_STATE.get(dest, "NA")

        row = {
            # Temporal
            "MONTH": month,
            "DAY_OF_WEEK": day_of_week,
            "DAY_OF_MONTH": 15,
            "QUARTER": quarter,
            "IS_WEEKEND": 1 if day_of_week in (6, 7) else 0,
            "SEASON": season,
            "IS_SUMMER": 1 if month in (6, 7, 8) else 0,
            "IS_WINTER": 1 if month in (12, 1, 2) else 0,
            "IS_HOLIDAY_SEASON": 1 if month in (11, 12) else 0,
            "DEP_HOUR_BIN": dep_hour_bin,

            # Distance
            "DISTANCE": distance,
            "AIR_TIME": distance / 8.0,  # simple proxy
            "DISTANCE_GROUP": min(int(distance / 250) + 1, 10),
            "DISTANCE_CAT": distance_cat,
            "DISTANCE_NORMALIZED": distance_norm,
            "IS_SHORT_HAUL": 1 if distance < 500 else 0,
            "IS_MEDIUM_HAUL": 1 if 500 <= distance < 1500 else 0,
            "IS_LONG_HAUL": 1 if distance >= 1500 else 0,

            # Hub/traffic proxies
            "IS_HUB_ORIGIN": 1 if origin in self.HUB_AIRPORTS else 0,
            "IS_HUB_DEST": 1 if dest in self.HUB_AIRPORTS else 0,
            "IS_HUB_TO_HUB": 1 if (origin in self.HUB_AIRPORTS and dest in self.HUB_AIRPORTS) else 0,
            "IS_BUSY_ORIGIN": 1 if origin in self.BUSY_AIRPORTS else 0,
            "IS_BUSY_DEST": 1 if dest in self.BUSY_AIRPORTS else 0,
            "IS_POPULAR_ROUTE": 0,
            "ROUTE_POPULARITY": 0.5,
            "ORIGIN_TRAFFIC": 0.5,
            "DEST_TRAFFIC": 0.5,
            "CARRIER_VOLUME": 0.5,

            # Categorical identifiers
            "UNIQUE_CARRIER": carrier,
            "ORIGIN": origin,
            "DEST": dest,
            "ROUTE": route,
            "CARRIER_ROUTE": carrier_route,
            "ORIGIN_STATE_ABR": origin_state,
            "DEST_STATE_ABR": dest_state,

            # Risk encodings
            "ORIGIN_RISK": self._map_val("ORIGIN_RISK", origin),
            "DEST_RISK": self._map_val("DEST_RISK", dest),
            "UNIQUE_CARRIER_RISK": self._map_val("UNIQUE_CARRIER_RISK", carrier),
            "ROUTE_RISK": self._map_val("ROUTE_RISK", route),
            "CARRIER_ROUTE_RISK": self._map_val("CARRIER_ROUTE_RISK", carrier_route),

            # Counts
            "TRAIN_ORIGIN_COUNT": self._map_val("TRAIN_ORIGIN_COUNT", origin),
            "TRAIN_DEST_COUNT": self._map_val("TRAIN_DEST_COUNT", dest),
            "TRAIN_ROUTE_COUNT": self._map_val("TRAIN_ROUTE_COUNT", route),
            "TRAIN_UNIQUE_CARRIER_COUNT": self._map_val("TRAIN_UNIQUE_CARRIER_COUNT", carrier),
            "TRAIN_CARRIER_ROUTE_COUNT": self._map_val("TRAIN_CARRIER_ROUTE_COUNT", carrier_route),
        }

        # Ensure correct column order / missing columns handling
        if self.selected:
            ordered: Dict[str, Any] = {}
            cat_cols = self.cat_cols_from_model or self.CATEGORICAL_COLS
            for col in self.selected:
                if col in row:
                    ordered[col] = row[col]
                else:
                    ordered[col] = "__MISSING__" if col in cat_cols else 0.0
            row = ordered

        df = pd.DataFrame([row])

        # Enforce dtype consistency (categorical as str, numeric as float)
        cat_cols = self.cat_cols_from_model or self.CATEGORICAL_COLS
        for col in df.columns:
            if col in cat_cols:
                df[col] = df[col].astype(str)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

        return df

# ==========================
# Explainer
# ==========================
class Explainer:
    @staticmethod
    def explain(row_dict: Dict[str, Any], prob: float) -> List[str]:
        """Generate explanations based on feature values and prediction."""
        reasons = []
        
        # Time-based factors
        dep_hour_bin = str(row_dict.get("DEP_HOUR_BIN", ""))
        if dep_hour_bin in ["18-20", "21-23"]:
            reasons.append("Evening/night departures have higher delays due to cascading effects.")
        elif dep_hour_bin in ["6-8", "9-11"]:
            reasons.append("Morning flights typically have the lowest delay risk.")

        # Origin risk
        origin_risk = float(row_dict.get("ORIGIN_RISK", 0))
        origin = row_dict.get("ORIGIN", "")
        if origin_risk > 0.18:
            reasons.append(f"Origin {origin} has elevated delay risk ({origin_risk:.1%}).")

        # Destination risk
        dest_risk = float(row_dict.get("DEST_RISK", 0))
        dest = row_dict.get("DEST", "")
        if dest_risk > 0.18:
            reasons.append(f"Destination {dest} has elevated delay risk ({dest_risk:.1%}).")

        # Carrier risk
        carrier_risk = float(row_dict.get("UNIQUE_CARRIER_RISK", 0))
        carrier = row_dict.get("UNIQUE_CARRIER", "")
        if carrier_risk > 0.17:
            reasons.append(f"Carrier {carrier} has above-average delays ({carrier_risk:.1%}).")
        elif carrier_risk < 0.14:
            reasons.append(f"Carrier {carrier} has excellent on-time performance ({carrier_risk:.1%}).")

        # Seasonal factors
        is_summer = row_dict.get("IS_SUMMER", 0)
        if is_summer == 1 or str(is_summer) == "1":
            reasons.append("Summer season typically has higher delays due to weather and travel volume.")
        
        is_holiday = row_dict.get("IS_HOLIDAY_SEASON", 0)
        if is_holiday == 1 or str(is_holiday) == "1":
            reasons.append("Holiday season (Nov-Dec) can have increased delays.")

        # Weekend
        is_weekend = row_dict.get("IS_WEEKEND", 0)
        if is_weekend == 1 or str(is_weekend) == "1":
            reasons.append("Weekend flights may have different delay patterns.")

        if not reasons:
            if prob < 0.18:
                reasons.append("No major risk factors identified - this looks like a good flight choice!")
            else:
                reasons.append("Multiple smaller factors contribute to the overall delay risk.")

        return reasons[:4]

# -------- EXPLAINER ------
class Explainer:
    """
    Lightweight heuristic explanation generator.
    """
    @staticmethod
    def explain(row_dict: Dict[str, Any], prob: float) -> List[str]:
        reasons: List[str] = []

        dep_hour_bin = str(row_dict.get("DEP_HOUR_BIN", ""))
        if dep_hour_bin in ("18-20", "21-23"):
            reasons.append("Evening/night departures have higher delays due to cascading effects.")
        elif dep_hour_bin in ("6-8", "9-11") and prob < 0.50:
            # Avoid contradiction when overall risk is high
            reasons.append("Morning flights typically have the lowest delay risk.")

        origin_risk = float(row_dict.get("ORIGIN_RISK", 0.0))
        origin = str(row_dict.get("ORIGIN", ""))
        if origin_risk > 0.18:
            reasons.append(f"Origin {origin} has elevated delay risk ({origin_risk:.1%}).")

        dest_risk = float(row_dict.get("DEST_RISK", 0.0))
        dest = str(row_dict.get("DEST", ""))
        if dest_risk > 0.18:
            reasons.append(f"Destination {dest} has elevated delay risk ({dest_risk:.1%}).")

        carrier_risk = float(row_dict.get("UNIQUE_CARRIER_RISK", 0.0))
        carrier = str(row_dict.get("UNIQUE_CARRIER", ""))
        if carrier_risk > 0.17:
            reasons.append(f"Carrier {carrier} has above-average delays ({carrier_risk:.1%}).")
        elif carrier_risk < 0.14 and prob < 0.50:
            reasons.append(f"Carrier {carrier} has excellent on-time performance ({carrier_risk:.1%}).")

        if str(row_dict.get("IS_SUMMER", "0")) == "1":
            reasons.append("Summer season typically has higher delays due to weather and travel volume.")
        if str(row_dict.get("IS_HOLIDAY_SEASON", "0")) == "1":
            reasons.append("Holiday season (Nov-Dec) can have increased delays.")

        if not reasons:
            reasons.append("Multiple smaller factors contribute to the overall delay risk." if prob >= 0.18
                           else "No major risk factors identified - this looks like a good flight choice!")

        return reasons[:4]

# -------- PREDICTION SERVICE ------------
class PredictionService:
    """
    Encapsulates model inference + calibration + UI-facing risk bands.
    """
    def __init__(self, store: ArtifactStore):
        self.store = store
        self.feature_builder = FeatureBuilder(store)
        self.threshold = float(store.metadata.get("operating_threshold", 0.30))

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_type": self.store.model_type,
            "threshold": self.threshold,
            "features": len(self.store.metadata.get("selected_features", [])),
            "cat_features": len(self.store.cat_feature_indices),
            "calibration_loaded": self.store.calibrator is not None,
        }

    def predict(self, req: PredictionRequest) -> PredictionResponse:
        request_id = str(uuid.uuid4())
        t0 = time.time()

        if self.store.model is None:
            logger.error(f"[{request_id}] Model not loaded")
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Build features
        X = self.feature_builder.build(req)
        if self.store.feature_names:
            X = X.reindex(columns=self.store.feature_names)

        try:
            # Raw model probability
            if self.store.cat_feature_indices:
                pool = Pool(X, cat_features=self.store.cat_feature_indices)
                proba = self.store.model.predict_proba(pool)
            else:
                proba = self.store.model.predict_proba(X)

            raw_p = float(proba[0, 1])

            # Calibrated probability (Platt scaling) if available
            p = raw_p
            if self.store.calibrator is not None:
                eps = 1e-6
                log_odds = np.log((raw_p + eps) / (1 - raw_p + eps))
                p = float(self.store.calibrator.predict_proba([[log_odds]])[0, 1])

        except Exception as e:
            logger.exception(f"[{request_id}] Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        # UI risk bands
        if p >= 0.50:
            risk = "HIGH"
        elif p >= 0.25:
            risk = "MODERATE"
        else:
            risk = "LOW"

        # Explanations
        row_dict = X.iloc[0].to_dict()
        reasons = Explainer.explain(row_dict, p)

        latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            f"[{request_id}] /predict ok | raw_p={raw_p:.4f} cal_p={p:.4f} risk={risk} latency_ms={latency_ms} "
            f"| {req.origin}->{req.destination} {req.carrier} dow={req.day_of_week} month={req.month} dep={req.dep_hour}"
        )

        return PredictionResponse(
            delay_probability=round(p, 4),
            risk_level=risk,
            threshold=self.threshold,
            top_reasons=reasons,
            input_features_used={k: str(v) for k, v in row_dict.items()},
        )

    def recommend_times(self, req: RecommendTimesRequest) -> Dict[str, Any]:
        results = []
        for h in req.hours:
            try:
                pred = self.predict(PredictionRequest(
                    origin=req.origin,
                    destination=req.destination,
                    carrier=req.carrier,
                    dep_hour=h,
                    day_of_week=req.day_of_week,
                    month=req.month,
                ))
                results.append({
                    "hour": h,
                    "time": f"{h:02d}:00",
                    "delay_probability": pred.delay_probability,
                    "risk_level": pred.risk_level,
                })
            except Exception:
                continue

        best = sorted(results, key=lambda x: x["delay_probability"])[:5]
        return {"best_hours": best}

    def recommend_airlines(self, req: RecommendAirlinesRequest) -> Dict[str, Any]:
        results = []
        for c in req.carriers:
            try:
                pred = self.predict(PredictionRequest(
                    origin=req.origin,
                    destination=req.destination,
                    carrier=c,
                    dep_hour=req.dep_hour,
                    day_of_week=req.day_of_week,
                    month=req.month,
                ))
                results.append({
                    "carrier": c,
                    "delay_probability": pred.delay_probability,
                    "risk_level": pred.risk_level,
                })
            except Exception:
                continue

        best = sorted(results, key=lambda x: x["delay_probability"])[:5]
        return {"best_carriers": best}


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="Flight Delay Predictor API", version="3.2-structured")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    """
    Load artifacts and initialize PredictionService once at startup.
    """
    cfg = AppConfig.from_env()
    store = ArtifactStore(cfg)
    store.load()
    app.state.service = PredictionService(store)
    logger.info("Service initialized and ready.")


def get_service() -> PredictionService:
    svc = getattr(app.state, "service", None)
    if svc is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return svc


@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "Flight Delay Predictor", "version": "3.2-structured"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return get_service().health()


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    return get_service().predict(req)


@app.post("/recommend/times")
def recommend_times(req: RecommendTimesRequest) -> Dict[str, Any]:
    return get_service().recommend_times(req)


@app.post("/recommend/airlines")
def recommend_airlines(req: RecommendAirlinesRequest) -> Dict[str, Any]:
    return get_service().recommend_airlines(req)
