import os
import streamlit as st
import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo

st.set_page_config(page_title="Mushroom Classifier", layout="centered")
st.title("Mushroom Classifier")
st.write("Classify mushrooms as edible or poisonous using a Random Forest model")

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "mushroom_model.pkl")

# Load model
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please train the model with 'python src/train_mushroom_model.py")
    st.stop()
model = joblib.load(MODEL_PATH)

# Load dataset
@st.cache_data(show_spinner=False)
def load_ucirepo():
    ds = fetch_ucirepo(id=73)
    X = ds.data.features.copy()
    y = ds.data.targets.squeeze().copy()
    # Handle missing
    if 'stalk-root' in X.columns:
        X['stalk-root'] = X['stalk-root'].replace('?', 'unknown')
    return X, y

X_full, y_full = load_ucirepo()

label_map = {"e": "Edible", "p": "Poisonous"}

st.divider()
# Input selector
mode = st.radio("Choose input mode", ["Pick an existing sample", "Enter manually"], horizontal=True)

# Prediction helper
def predict_and_show(x_row: pd.Series):
    X_df = pd.DataFrame([x_row.to_dict()])
    pred = model.predict(X_df)[0]

    proba = model.predict_proba(X_df)[0]
    classes = model.classes_
    p_poison = proba[list(classes).index("p")]
    st.markdown(f"**{label_map.get(pred, pred)}** (P(poisonous) ≃ {p_poison:.2%})")

# Mode 1: pick an existing sample from the dataset
if mode == "Pick an existing sample":
    st.write("Select an existing sample to compare the true label vs predicted label")
    idx = st.number_input("Row index", min_value=0, max_value=len(X_full)-1, value=0, step=1)
    x_row = X_full.iloc[int(idx)]
    true_label = y_full.iloc[int(idx)]

    cols = st.columns(2)
    with cols[0]:
        st.caption("True label")
        st.markdown(f"**{label_map.get(true_label, true_label)}**")
    with cols[1]:
        st.caption("Predicted label")
        predict_and_show(x_row)

    with st.expander("Show features of this row"):
        st.dataframe(x_row.to_frame(name="value"))

# Mode 2: enter features manually with widgets
else:
    st.write("Set the categorical features below.")
    inputs = {}
    for col in X_full.columns:
        choices = sorted(X_full[col].astype(str).unique().tolist())
        # Set a reasonable default (first choice)
        val = st.selectbox(col, choices, index=0)
        inputs[col] = val

    if st.button("Predict"):
        x_row = pd.Series(inputs)
        predict_and_show(x_row)

st.caption("Data Source: UCI ML Repository (dataset ID 73) via 'ucimlrepo'.")