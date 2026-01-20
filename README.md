# Flight Delay Prediction – Intelligent AI System

An end-to-end **machine learning–powered flight delay prediction system** designed for travel applications.
The project predicts the probability of flight delays, provides actionable recommendations, and enables
natural-language interaction through an AI assistant.

## Key Features
- Calibrated CatBoost model for realistic delay probabilities
- FastAPI backend with structured logging
- Streamlit frontend with visual risk indicators
- OpenAI-powered conversational assistant
- Recommendations for best times and airlines

## High-Level Architecture
User
├─ Streamlit Frontend (UI + AI Chat)
│  ├─ Prediction form
│  ├─ Risk indicators (LOW / MODERATE / HIGH)
│  ├─ Recommendations
│  └─ Conversational AI (OpenAI)
│
└─ FastAPI Backend
   ├─ Feature Builder
   ├─ CatBoost Model
   ├─ Probability Calibration (Platt Scaling)
   ├─ Explanation Engine
   └─ Model Artifacts

## Machine Learning
- Dataset: US flights (2017–2018)
- Target: Arrival delay ≥ 15 minutes
- Model: CatBoostClassifier
- Calibration: Platt Scaling (logistic regression on log-odds)

## How to Run
```bash
pip install -r requirements.txt
python3 run src/run_notebooks.py
uvicorn src.app:app --port 8001
streamlit run src/streamlit_app.py
```

## Limitations
- No real-time weather or ATC data
- Historical patterns only
- Static model (no online learning)

## Future Improvements
- Live weather integration
- Automated retraining
- SHAP explanations
- User personalization

Built with **CatBoost, FastAPI, Streamlit, OpenAI**.
