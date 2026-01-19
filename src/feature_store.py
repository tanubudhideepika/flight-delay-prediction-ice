import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional

def hhmm_to_minutes(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    try:
        x = int(x)
        h = x // 100
        m = x % 100
        if h < 0 or h > 23 or m < 0 or m > 59:
            return np.nan
        return h * 60 + m
    except:
        return np.nan

def season_from_month(m: int) -> str:
    if m in [12, 1, 2]:
        return "Winter"
    if m in [3, 4, 5]:
        return "Spring"
    if m in [6, 7, 8]:
        return "Summer"
    return "Fall"

def distance_cat(distance: Optional[float]) -> Optional[str]:
    if distance is None or (isinstance(distance, float) and np.isnan(distance)):
        return None
    if distance < 500:
        return "Short"
    if distance < 1500:
        return "Medium"
    return "Long"

def dep_hour_bin(dep_time: Optional[int]) -> Optional[str]:
    mins = hhmm_to_minutes(dep_time)
    if np.isnan(mins):
        return None
    hour = int(mins // 60)
    if hour <= 5:
        return "0-5"
    if hour <= 8:
        return "6-8"
    if hour <= 11:
        return "9-11"
    if hour <= 14:
        return "12-14"
    if hour <= 17:
        return "15-17"
    if hour <= 20:
        return "18-20"
    return "21-23"

@dataclass
class FeatureStore:
    global_delay_rate: float
    risk_maps: Dict[str, Dict[str, float]]
    count_maps: Dict[str, Dict[str, int]]

    @staticmethod
    def load(path: str) -> "FeatureStore":
        with open(path, "r") as f:
            d = json.load(f)
        return FeatureStore(
            global_delay_rate=d["global_delay_rate"],
            risk_maps=d["risk_maps"],
            count_maps=d["count_maps"],
        )

    def lookup(self, map_name: str, key: str):
        m = self.risk_maps.get(map_name) or self.count_maps.get(map_name)
        if m is None:
            return None
        return m.get(key)

def build_feature_row(
    year: int, quarter: int, month: int, day_of_month: int, day_of_week: int,
    unique_carrier: str, origin: str, dest: str,
    origin_state_abr: Optional[str],
    dest_state_abr: Optional[str],
    dep_time: Optional[int],
    distance: Optional[float],
    air_time: Optional[float],
    distance_group: Optional[int],
    store: FeatureStore
) -> Dict[str, Any]:

    route = f"{origin}-{dest}"
    carrier_route = f"{unique_carrier}:{route}"

    season = season_from_month(month)
    dep_bin = dep_hour_bin(dep_time)
    dist_cat = distance_cat(distance)

    is_weekend = 1 if day_of_week in [6, 7] else 0
    is_business_day = 1 if day_of_week in [1,2,3,4,5] else 0

    def risk(map_name, key):
        v = store.lookup(map_name, key)
        return float(v) if v is not None else float(store.global_delay_rate)

    def cnt(map_name, key):
        v = store.lookup(map_name, key)
        return int(v) if v is not None else 0

    return {
        "YEAR": year,
        "QUARTER": quarter,
        "MONTH": month,
        "DAY_OF_MONTH": day_of_month,
        "DAY_OF_WEEK": day_of_week,

        "SEASON": season,
        "IS_WEEKEND": is_weekend,
        "IS_BUSINESS_DAY": is_business_day,
        "DEP_HOUR_BIN": dep_bin,

        "DISTANCE": distance,
        "AIR_TIME": air_time,
        "DISTANCE_GROUP": distance_group,
        "DISTANCE_CAT": dist_cat,

        "UNIQUE_CARRIER": unique_carrier,
        "ORIGIN": origin,
        "DEST": dest,
        "ROUTE": route,
        "CARRIER_ROUTE": carrier_route,

        "ORIGIN_STATE_ABR": origin_state_abr,
        "DEST_STATE_ABR": dest_state_abr,

        "ORIGIN_RISK": risk("ORIGIN_RISK", origin),
        "DEST_RISK": risk("DEST_RISK", dest),
        "UNIQUE_CARRIER_RISK": risk("UNIQUE_CARRIER_RISK", unique_carrier),
        "ROUTE_RISK": risk("ROUTE_RISK", route),
        "CARRIER_ROUTE_RISK": risk("CARRIER_ROUTE_RISK", carrier_route),

        "TRAIN_ORIGIN_COUNT": cnt("TRAIN_ORIGIN_COUNT", origin),
        "TRAIN_DEST_COUNT": cnt("TRAIN_DEST_COUNT", dest),
        "TRAIN_UNIQUE_CARRIER_COUNT": cnt("TRAIN_UNIQUE_CARRIER_COUNT", unique_carrier),
        "TRAIN_ROUTE_COUNT": cnt("TRAIN_ROUTE_COUNT", route),
        "TRAIN_CARRIER_ROUTE_COUNT": cnt("TRAIN_CARRIER_ROUTE_COUNT", carrier_route),
    }
