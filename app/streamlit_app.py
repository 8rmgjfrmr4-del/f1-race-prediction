import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="F1 Podium Predictor", layout="centered")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "f1_multi_season_engineered.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "f1_podium_predictor_multi_season.pkl"

if not DATA_PATH.exists():
    st.error(f"Missing data file: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Missing model file: {MODEL_PATH}")
    st.stop()
    
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Ensure expected dtypes
    for col in ["season", "round", "grid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()

def get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features):
    """
    Returns feature names after ColumnTransformer:
      - numeric passthrough names
      - one-hot encoded categorical names
    """
    feature_names = []

    # ColumnTransformer stores transformers in order: ("num", ...), ("cat", ...)
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            # numeric_transformer -> SimpleImputer -> passthrough
            feature_names.extend(cols)

        elif name == "cat":
            # categorical_transformer -> SimpleImputer -> OneHotEncoder
            ohe = transformer.named_steps["onehot"]
            ohe_feature_names = ohe.get_feature_names_out(cols)
            feature_names.extend(ohe_feature_names.tolist())

    return feature_names


def compute_shap_for_single_row(pipeline_model, X_one_row, numeric_features, categorical_features):
    """
    Computes SHAP values for the positive class (podium=1) for a single-row input.
    Assumes the pipeline_model has steps: preprocess -> model (RandomForestClassifier).
    """
    preprocessor = pipeline_model.named_steps["preprocess"]
    estimator = pipeline_model.named_steps["model"]

    # Transform the single-row input the same way training did
    X_trans = preprocessor.transform(X_one_row)

    # Build feature names for the transformed matrix
    feature_names = get_feature_names_from_preprocessor(
        preprocessor, numeric_features, categorical_features
    )

    # SHAP for tree-based models
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_trans)

    
    # Robust extraction for binary classification:
    # Possible shapes:
    # - list of 2 arrays: [ (n_samples, n_features), (n_samples, n_features) ]
    # - array: (n_samples, n_features)
    # - array: (n_samples, n_features, n_classes)  (less common)
    if isinstance(shap_values, list):
        # class 1 array -> shape (n_samples, n_features)
        sv = shap_values[1]
    else:
        sv = shap_values

    sv = np.asarray(sv)

    # If sv includes the sample dimension, take the first row
    if sv.ndim == 2:
        shap_class1 = sv[0]           # (n_features,)
    elif sv.ndim == 3:
        # If shape is (n_samples, n_features, n_classes) take class 1
        shap_class1 = sv[0, :, 1]
    elif sv.ndim == 1:
        shap_class1 = sv
    else:
        raise ValueError(f"Unexpected SHAP values shape: {sv.shape}")

    shap_class1 = np.ravel(shap_class1)  # ensure 1-D
    if len(feature_names) != len(shap_class1):
        raise ValueError(f"Feature name count {len(feature_names)} != SHAP count {len(shap_class1)}")
    
    # Return paired feature contributions
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_class1
    })

    # Sort by absolute impact
    shap_df["abs_impact"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_impact", ascending=False).drop(columns=["abs_impact"])

    return shap_df

st.title("F1 Podium Predictor (Race-Weekend Mode)")

st.write("Select a race weekend and a driver. The app uses only pre-race engineered features.")

# --- Sidebar selections ---
seasons = sorted(df["season"].dropna().unique().astype(int).tolist())
season = st.sidebar.selectbox("Season", seasons, index=len(seasons)-1)

rounds = sorted(df.loc[df["season"] == season, "round"].dropna().unique().astype(int).tolist())
round_num = st.sidebar.selectbox("Round", rounds)

race_slice = df[(df["season"] == season) & (df["round"] == round_num)].copy()

# Driver selection: use driver code (3-letter)
drivers = sorted(race_slice["Driver.code"].dropna().unique().tolist())
driver_code = st.sidebar.selectbox("Driver", drivers)

row = race_slice[race_slice["Driver.code"] == driver_code]

if row.empty:
    st.error("No data found for that driver in this race. Try another selection.")
    st.stop()

# There should be exactly one row per driver per race
row = row.iloc[0]

st.subheader(f"{season} Round {round_num}: {row.get('raceName', 'Race')}")

# --- Inputs / features ---
# Allow user to use actual grid from dataset or override
use_actual_grid = st.checkbox("Use actual grid from dataset", value=True)

if use_actual_grid:
    grid = float(row["grid"]) if pd.notna(row["grid"]) else 20.0
    st.write(f"Grid position (from data): **{int(grid)}**")
else:
    grid = st.slider("Grid position (what-if)", min_value=1, max_value=20, value=int(row["grid"]) if pd.notna(row["grid"]) else 20)

# Build feature vector using SAFE pre-race features
numeric_features = [
    "grid",
    "constructor_cum_points",
    "driver_avg_last5",
    "driver_podiums_last5",
    "driver_best_last5",
    "driver_worst_last5",
    "grid_delta_vs_teammate",
]

categorical_features = [
    "Driver.code",
    "Constructor.name",
]

# Create a single-row DataFrame for prediction
x = {
    "grid": grid,
    "constructor_cum_points": row.get("constructor_cum_points", np.nan),
    "driver_avg_last5": row.get("driver_avg_last5", np.nan),
    "driver_podiums_last5": row.get("driver_podiums_last5", np.nan),
    "driver_best_last5": row.get("driver_best_last5", np.nan),
    "driver_worst_last5": row.get("driver_worst_last5", np.nan),
    "grid_delta_vs_teammate": row.get("grid_delta_vs_teammate", np.nan),
    "Driver.code": row.get("Driver.code", None),
    "Constructor.name": row.get("Constructor.name", None),
}

X_infer = pd.DataFrame([x])

st.markdown("### Features used for prediction")
st.dataframe(X_infer)

# --- Predict ---
st.markdown("### Prediction")
show_shap = st.checkbox("Show SHAP explanation", value=False)

if st.button("Predict podium probability"):
    # Pipeline model supports predict_proba if classifier
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_infer)[0][1]
        st.metric("Podium probability", f"{proba:.2%}")
    else:
        pred = model.predict(X_infer)[0]
        st.write(f"Predicted podium (0/1): {pred}")

    # Simple interpretation
    if hasattr(model, "predict_proba"):
        if proba >= 0.70:
            st.success("High podium likelihood (based on historical patterns).")
        elif proba >= 0.40:
            st.info("Moderate podium likelihood.")
        else:
            st.warning("Low podium likelihood.")
    
    if show_shap:
        # --- SHAP explanation (top drivers of the prediction) ---
        st.markdown("### Why the model predicted this (SHAP)")

        try:
            shap_df = compute_shap_for_single_row(
            model, X_infer, numeric_features, categorical_features
            )

            top_k = st.slider("How many factors to show?", 5, 25, 10)
            shap_top = shap_df.head(top_k)

            st.write("Positive SHAP pushes podium probability up; negative SHAP pushes it down.")
            st.dataframe(shap_top)

            # Simple bar plot of SHAP contributions
            fig, ax = plt.subplots()
            ax.barh(shap_top["feature"][::-1], shap_top["shap_value"][::-1])
            ax.set_xlabel("SHAP value (impact on podium prediction)")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

        except Exception as e:
            st.warning("SHAP explanation failed for this selection.")
            st.exception(e)

st.caption("Note: This is a learning project. Predictions reflect historical data patterns, not real-time factors (weather, penalties, DNFs, upgrades).")