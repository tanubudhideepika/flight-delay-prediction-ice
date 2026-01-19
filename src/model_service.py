import os
import json
import numpy as np
import pandas as pd
import joblib

class ModelService:
    def __init__(self, artifacts_dir: str):
        meta_path = os.path.join(artifacts_dir, "model_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing: {meta_path}")

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.selected_features = self.meta["selected_features"]
        self.threshold = float(self.meta.get("operating_threshold", 0.30))
        self.best_model = self.meta["best_model"]

        cbm_path = os.path.join(artifacts_dir, "catboost_model.cbm")
        joblib_path = os.path.join(artifacts_dir, "best_model.joblib")

        self.model = None
        self.model_type = None

        if os.path.exists(cbm_path):
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(cbm_path)
            self.model = m
            self.model_type = "catboost"
        elif os.path.exists(joblib_path):
            self.model = joblib.load(joblib_path)
            self.model_type = "sklearn"
        else:
            raise FileNotFoundError("No model file found (catboost_model.cbm or best_model.joblib)")

    def predict_proba(self, feature_rows):
        X = pd.DataFrame(feature_rows)

        for c in self.selected_features:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.selected_features].copy()

        if self.model_type == "catboost":
            X_cb = X.copy()
            for c in X_cb.columns:
                if X_cb[c].dtype == "object":
                    X_cb[c] = X_cb[c].astype(str)
                    X_cb.loc[X_cb[c].isin(["nan", "None", "<NA>"]), c] = "__MISSING__"
                else:
                    X_cb[c] = pd.to_numeric(X_cb[c], errors="coerce")
            return self.model.predict_proba(X_cb)[:, 1]

        return self.model.predict_proba(X)[:, 1]

    def risk_band(self, p: float) -> str:
        if p < 0.20:
            return "Low"
        if p < 0.40:
            return "Medium"
        return "High"
