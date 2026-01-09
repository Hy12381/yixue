import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from delirium_web.features import WEB_BINARY_FEATURES, WEB_FEATURE_LABELS_EN


def _default_model_path():
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "delirium_web_model.joblib"),
        os.path.join(here, "新建文件夹", "delirium_web_model.joblib"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


@st.cache_resource
def load_model():
    model_path = os.environ.get("MODEL_PATH", _default_model_path())
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    labels = bundle.get("labels") or {k: WEB_FEATURE_LABELS_EN.get(k, k) for k in feature_names}
    return model, feature_names, labels


def make_contribution_figure(*, model, feature_names, labels, x_input, proba):
    coefs = model.coef_[0]
    x_vec = np.array(x_input, dtype=float)
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
    ax.set_title(f"Feature contributions (predicted risk = {proba*100:.1f}%)")
    fig.tight_layout()
    return fig


def make_force_figure(*, model, feature_names, labels, x_input, proba):
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])
    x = np.array(x_input, dtype=float)

    base_log_odds = intercept
    shap_vals = coefs * x
    final_log_odds = base_log_odds + float(np.sum(shap_vals))

    order = np.argsort(np.abs(shap_vals))[::-1]
    shap_vals = shap_vals[order]
    x = x[order]
    names = [labels.get(feature_names[i], feature_names[i]) for i in order]

    fig, ax = plt.subplots(figsize=(6, 1.8))
    ax.set_yticks([])

    seq = [base_log_odds]
    cur = base_log_odds
    segments = []
    for name, v, xi in zip(names, shap_vals, x):
        start = cur
        end = cur + float(v)
        seq.append(end)
        segments.append((name, float(v), float(xi), start, end))
        cur = end

    min_x = min(seq)
    max_x = max(seq)
    span = max(1e-6, max_x - min_x)
    pad = 0.08 * span
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(0, 1)

    y0 = 0.50
    h = 0.26
    head = 0.04 * span

    def arrow(x0, x1, color):
        dx = x1 - x0
        if abs(dx) < 1e-12:
            return
        direction = 1 if dx > 0 else -1
        body = abs(dx) - head
        if body <= 0:
            body = abs(dx)
            head_local = 0.0
        else:
            head_local = head
        x_body_end = x0 + direction * body

        y_top = y0 + h / 2
        y_bot = y0 - h / 2
        y_mid = y0

        if direction > 0:
            pts_x = [x0, x_body_end, x_body_end + head_local, x_body_end, x0]
        else:
            pts_x = [x0, x_body_end, x_body_end - head_local, x_body_end, x0]
        pts_y = [y_bot, y_bot, y_mid, y_top, y_top]
        ax.fill(pts_x, pts_y, color=color, edgecolor="#111111", linewidth=0.8)

    for name, v, xi, start, end in segments:
        color = "#f2c200" if v >= 0 else "#7a2f70"
        arrow(start, end, color)

    top_k = 4
    segments_sorted = sorted(segments, key=lambda s: abs(s[1]), reverse=True)[:top_k]
    label_levels = np.linspace(y0 + h * 0.7, 0.95, num=len(segments_sorted))
    for (name, v, xi, start, end), ly in zip(segments_sorted, label_levels):
        mid_x = 0.5 * (start + end)
        ax.vlines(mid_x, y0 + h / 2, ly, colors="#555555", linestyles="dashed", linewidth=0.7)
        ax.text(
            mid_x,
            ly,
            f"{name}={xi:g}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.axvline(base_log_odds, color="#333333", linewidth=1.0, alpha=0.7)
    ax.axvline(final_log_odds, color="#111111", linewidth=1.2)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    base_p = float(sigmoid(base_log_odds))
    final_p = float(sigmoid(final_log_odds))
    ax.set_xlabel("Prediction", fontsize=8)
    ax.text(
        base_log_odds,
        0.10,
        f"E[f(x)]={base_p:.3f}",
        ha="center",
        va="center",
        fontsize=8,
        color="#333333",
    )
    ax.text(
        final_log_odds,
        0.10,
        f"f(x)={final_p:.3f}",
        ha="center",
        va="center",
        fontsize=8,
        color="#111111",
    )
    ax.set_title(f"Force plot (predicted risk = {proba*100:.1f}%)", fontsize=9, pad=4)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="x", labelsize=7, colors="#333333")
    fig.tight_layout()
    return fig


model, feature_names, labels = load_model()

st.set_page_config(page_title="ICU Delirium Risk Predictor", layout="wide")

st.title("ICU Delirium Risk Predictor")
st.write("Enter inputs and click Predict. Binary variables: 0 = No, 1 = Yes.")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("New Observation")
    st.caption("7-variable logistic regression")

    inputs = {}
    for name in feature_names:
        label = labels.get(name, name)
        if name in WEB_BINARY_FEATURES:
            inputs[name] = st.selectbox(label, options=[0, 1], index=0, key=name)
        else:
            inputs[name] = st.number_input(label, value=0.0, format="%.2f", key=name)

    predict_clicked = st.button("Predict")

with right_col:
    st.subheader("About the model")
    st.markdown(
        "The model outputs a delirium risk probability using a logistic regression formula:"
    )
    st.code("p = sigmoid(b0 + Σ bi·xi)", language="text")

    st.markdown("---")
    st.subheader("Prediction")

    if "last_proba" not in st.session_state:
        st.session_state["last_proba"] = None

    if predict_clicked:
        X = pd.DataFrame([[inputs[n] for n in feature_names]], columns=feature_names)
        proba = float(model.predict_proba(X)[0, 1])
        st.session_state["last_proba"] = (proba, inputs)

    if st.session_state["last_proba"] is not None:
        proba, stored_inputs = st.session_state["last_proba"]
        st.write("Predicted probability")
        st.metric("Delirium during ICU stay", f"{proba * 100:.1f}%")
        st.progress(min(max(proba, 0.0), 1.0))

        st.markdown("### Explanation")
        st.caption("Contributions are shown on the log-odds scale.")

        x_input = [float(stored_inputs[n]) for n in feature_names]
        fig_bar = make_contribution_figure(
            model=model,
            feature_names=feature_names,
            labels=labels,
            x_input=x_input,
            proba=proba,
        )
        st.pyplot(fig_bar, use_container_width=True)

        fig_force = make_force_figure(
            model=model,
            feature_names=feature_names,
            labels=labels,
            x_input=x_input,
            proba=proba,
        )
        st.pyplot(fig_force, use_container_width=True)

st.markdown(
    "<div style='text-align:center;color:#777;font-size:12px;margin-top:24px;'>"
    "For research and education only. Not for clinical decision-making."
    "</div>",
    unsafe_allow_html=True,
)
