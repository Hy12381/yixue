import pandas as pd
import numpy as np


def safe_read_csv(path):
    for encoding in ("gbk", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="gbk")


def coerce_numeric_series(s):
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all():
        return numeric
    codes = s.astype("category").cat.codes.replace(-1, np.nan)
    if codes.isna().any():
        mode = codes.mode(dropna=True)
        fill_value = int(mode.iloc[0]) if len(mode) > 0 else 0
        codes = codes.fillna(fill_value)
    return codes.astype(int)

