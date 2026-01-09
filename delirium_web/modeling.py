import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from delirium_web.features import WEB_FEATURES, WEB_FEATURE_LABELS_EN
from delirium_web.io_utils import safe_read_csv, coerce_numeric_series

RANDOM_STATE = 52


def train_delirium_model(*, data_path, output_dir, external_validation_path=None, feature_names=None):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if not output_dir.endswith(os.sep):
        output_dir += os.sep

    feature_names = feature_names or WEB_FEATURES
    data = safe_read_csv(data_path)
    if "Result" not in data.columns:
        raise ValueError("CSV 缺少 Result 列")

    missing = [c for c in feature_names if c not in data.columns]
    if missing:
        raise ValueError(f"CSV 缺少特征列: {missing}")

    X = data[feature_names].copy()
    for c in feature_names:
        X[c] = coerce_numeric_series(X[c])

    y = coerce_numeric_series(data["Result"]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    metrics = [
        {"Dataset": "Train", "AUC": float(roc_auc_score(y_train, train_proba))},
        {"Dataset": "Test", "AUC": float(roc_auc_score(y_test, test_proba))},
    ]

    if external_validation_path and os.path.exists(external_validation_path):
        external_data = safe_read_csv(external_validation_path)
        if "Result" in external_data.columns:
            X_ext = external_data[feature_names].copy()
            for c in feature_names:
                X_ext[c] = coerce_numeric_series(X_ext[c])
            y_ext = coerce_numeric_series(external_data["Result"]).astype(int)
            ext_proba = model.predict_proba(X_ext)[:, 1]
            metrics.append({"Dataset": "External", "AUC": float(roc_auc_score(y_ext, ext_proba))})

    bundle = {
        "model_type": "logistic_regression",
        "model": model,
        "feature_names": feature_names,
        "labels": {k: WEB_FEATURE_LABELS_EN.get(k, k) for k in feature_names},
        "feature_means": X_train.mean(numeric_only=True).to_dict(),
        "trained_at": pd.Timestamp.now().isoformat(),
    }

    model_path = f"{output_dir}delirium_web_model.joblib"
    joblib.dump(bundle, model_path)
    pd.DataFrame(metrics).to_csv(f"{output_dir}delirium_web_model_auc.csv", index=False)

    return model_path
