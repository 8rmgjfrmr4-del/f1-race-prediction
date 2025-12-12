## Streamlit App (Race-Weekend Mode)

This project includes an interactive Streamlit app that predicts the probability
of an F1 podium finish for a given race weekend.

Features:
- Multi-season (2021–2023) training
- Pre-race feature engineering (no leakage)
- Random Forest classifier
- SHAP-based local explanations for each prediction

To run locally:
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py