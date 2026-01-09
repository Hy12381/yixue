import os
import base64
import io

import joblib
import pandas as pd
from flask import Flask, render_template, request

from delirium_web.features import WEB_BINARY_FEATURES


def _make_contribution_plot_png(*, model, feature_names, labels, x_input, proba):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])
    x = np.array(x_input, dtype=float)

    contrib = coefs * x
    names = [labels.get(n, n) for n in feature_names]

    order = np.argsort(np.abs(contrib))[::-1]
    contrib = contrib[order]
    names = [names[i] for i in order]

    fig = plt.figure(figsize=(8.5, 3.8), dpi=140)
    ax = fig.add_subplot(111)

    y = np.arange(len(names))
    colors = ["#111111" if v >= 0 else "#9a9a9a" for v in contrib]
    ax.barh(y, contrib, color=colors, edgecolor="#111111", linewidth=0.6)
    ax.axvline(0, color="#111111", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to log-odds", fontsize=9)
    ax.set_title(f"Feature contributions (predicted risk = {proba*100:.1f}%)", fontsize=10, pad=10)

    lo = intercept + float(np.sum(contrib))
    ax.text(
        0.99,
        0.02,
        f"Intercept: {intercept:.3f}   Total log-odds: {lo:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _sigmoid(z):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-z))


def _make_force_plot_png(*, model, feature_names, labels, x_input, feature_means, proba):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])
    x = np.array(x_input, dtype=float)
    means = np.array([float(feature_means.get(n, 0.0)) for n in feature_names], dtype=float)

    base_log_odds = intercept + float(np.dot(coefs, means))
    shap_vals = coefs * (x - means)
    final_log_odds = base_log_odds + float(np.sum(shap_vals))

    order = np.argsort(np.abs(shap_vals))[::-1]
    shap_vals = shap_vals[order]
    x = x[order]
    means = means[order]
    names = [labels.get(feature_names[i], feature_names[i]) for i in order]

    fig = plt.figure(figsize=(8.5, 2.0), dpi=140)
    ax = fig.add_subplot(111)
    ax.set_yticks([])

    seq = [base_log_odds]
    cur = base_log_odds
    for v in shap_vals:
        cur = cur + float(v)
        seq.append(cur)

    min_x = min(seq)
    max_x = max(seq)
    span = max(1e-6, max_x - min_x)
    pad = 0.08 * span
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(0, 1)

    y0 = 0.50
    h = 0.26
    head = 0.04 * span

    def _arrow_patch(x0, x1, color, edge):
        dx = x1 - x0
        if abs(dx) < 1e-12:
            return None
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
            pts = [
                (x0, y_bot),
                (x_body_end, y_bot),
                (x_body_end, y_bot),
                (x_body_end, y_bot),
                (x_body_end + head_local, y_mid),
                (x_body_end, y_top),
                (x0, y_top),
            ]
        else:
            pts = [
                (x0, y_bot),
                (x_body_end, y_bot),
                (x_body_end, y_bot),
                (x_body_end, y_bot),
                (x_body_end - head_local, y_mid),
                (x_body_end, y_top),
                (x0, y_top),
            ]
        return patches.Polygon(pts, closed=True, facecolor=color, edgecolor=edge, linewidth=0.8)

    cur = base_log_odds
    for name, v, xi in zip(names, shap_vals, x):
        nxt = cur + float(v)
        color = "#f2c200" if v >= 0 else "#7a2f70"
        edge = "#111111"
        patch = _arrow_patch(cur, nxt, color, edge)
        if patch is not None:
            ax.add_patch(patch)

        label_text = f"{name}={xi:g}"
        ax.text(
            (cur + nxt) / 2,
            y0 + (h / 2) + 0.06,
            label_text,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#111111",
            rotation=0,
        )
        cur = nxt

    ax.axvline(base_log_odds, color="#333333", linewidth=1.0, alpha=0.7)
    ax.axvline(final_log_odds, color="#111111", linewidth=1.2)

    base_p = float(_sigmoid(base_log_odds))
    final_p = float(_sigmoid(final_log_odds))
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
    ax.set_title(f"Force plot (predicted risk = {proba*100:.1f}%)", fontsize=9, pad=6)

    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(axis="x", labelsize=7, colors="#333333")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def create_app(*, model_path):
    model_path = os.path.abspath(model_path)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    labels = bundle.get("labels") or bundle.get("labels_zh") or {}
    feature_means = bundle.get("feature_means") or {}

    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    @app.get("/")
    def index_get():
        fields = [
            {
                "name": n,
                "label": labels.get(n, n),
                "is_binary": n in WEB_BINARY_FEATURES,
                "value": "",
            }
            for n in feature_names
        ]
        return render_template("index.html", fields=fields, result=None, error=None, plot_png=None, force_png=None)

    @app.post("/")
    def index_post():
        values = {}
        error = None
        result = None
        plot_png = None
        force_png = None
        try:
            x = []
            for name in feature_names:
                raw = (request.form.get(name) or "").strip()
                if raw == "":
                    raise ValueError(f"Please fill in: {labels.get(name, name)}")
                val = float(raw)
                if name in WEB_BINARY_FEATURES and val not in (0.0, 1.0):
                    raise ValueError(f"{labels.get(name, name)} must be 0 or 1")
                values[name] = raw
                x.append(val)

            X_input = pd.DataFrame([x], columns=feature_names)
            result = float(model.predict_proba(X_input)[0, 1])
            plot_png = _make_contribution_plot_png(
                model=model,
                feature_names=feature_names,
                labels=labels,
                x_input=x,
                proba=result,
            )
            force_png = _make_force_plot_png(
                model=model,
                feature_names=feature_names,
                labels=labels,
                x_input=x,
                feature_means=feature_means,
                proba=result,
            )
        except Exception as e:
            error = str(e)

        fields = [
            {
                "name": n,
                "label": labels.get(n, n),
                "is_binary": n in WEB_BINARY_FEATURES,
                "value": values.get(n, ""),
            }
            for n in feature_names
        ]
        return render_template(
            "index.html",
            fields=fields,
            result=result,
            error=error,
            plot_png=plot_png,
            force_png=force_png,
        )

    return app
