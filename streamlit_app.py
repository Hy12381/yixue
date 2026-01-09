import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from delirium_web.features import WEB_BINARY_FEATURES, WEB_FEATURE_LABELS_EN


@st.cache_resource
def load_model():
    bundle = joblib.load("delirium_web_model.joblib")
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    labels = bundle.get("labels") or {k: WEB_FEATURE_LABELS_EN.get(k, k) for k in feature_names}
    return model, feature_names, labels


model, feature_names, labels = load_model()

st.set_page_config(page_title="ICU Delirium Risk Predictor", layout="wide")

st.title("ICU Delirium Risk Predictor")

st.write(
    "Binary variables: 0 = No, 1 = Yes. The model uses seven variables "
    "to estimate the probability of delirium during ICU stay."
)

inputs = {}
cols = st.columns(2)

for idx, name in enumerate(feature_names):
    label = labels.get(name, name)
    col = cols[idx % 2]
    if name in WEB_BINARY_FEATURES:
        inputs[name] = col.selectbox(label, options=[0, 1], index=0)
    else:
        inputs[name] = col.number_input(label, value=0.0, format="%.2f")

if st.button("Predict"):
    X = pd.DataFrame([[inputs[n] for n in feature_names]], columns=feature_names)
    proba = float(model.predict_proba(X)[0, 1])

    left, right = st.columns([1, 1])
    with left:
        st.metric("Predicted delirium probability", f"{proba * 100:.1f}%")

    with right:
        coefs = model.coef_[0]
        x_vec = np.array([float(inputs[n]) for n in feature_names])
        contrib = coefs * x_vec

        order = np.argsort(np.abs(contrib))[::-1]
        contrib = contrib[order]
        names = [labels.get(feature_names[i], feature_names[i]) for i in order]

        fig, ax = plt.subplots(figsize=(6, 3))
        y = np.arange(len(names))
        colors = ["#111111" if v >= 0 else "#9a9a9a" for v in contrib]
        ax.barh(y, contrib, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Contribution to log-odds")
        ax.set_title("Feature contributions")
        st.pyplot(fig)

