import joblib, pandas as pd, numpy as np
from pathlib import Path

ROOT = Path("/Users/ananyakapoor/Desktop/modeling")
MODEL_PATH = ROOT/"analysis/pls/svm/loocv/final_model.joblib"
X_PATH     = ROOT/"analysis/pls/pls_space.csv"
Y_PATH     = ROOT/"data/qualification_target.csv"
OUT_PATH   = ROOT/"analysis/pls/svm/final_predictions.csv"

# ----------------------------
# load model + data
# ----------------------------
model = joblib.load(MODEL_PATH)
X = pd.read_csv(X_PATH)

y_raw = pd.read_csv(Y_PATH)["Qualified Municipality"]

# coerce target to 0/1 robustly
if y_raw.dtype == object:
    s = y_raw.astype(str).str.strip().str.upper()
    mapping = {"YES": 1, "NO": 0, "TRUE": 1, "FALSE": 0, "1": 1, "0": 0}
    y = s.map(mapping)
else:
    y = pd.to_numeric(y_raw, errors="coerce")

if y.isna().any():
    bad = y_raw[y.isna()].head(10).tolist()
    raise ValueError(f"Target has values I can't coerce to 0/1. Examples: {bad}")

y = y.astype(int)

# ----------------------------
# predict
# ----------------------------
scores = model.decision_function(X)
scores = np.asarray(scores).ravel()

# for SVM decision_function, 0.0 is the natural threshold
y_pred = (scores >= 0.0).astype(int)

# ----------------------------
# save in the format plot_prediction_map expects
# ----------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "row_index": np.arange(len(X), dtype=int),
    "y_true": y.values.astype(int),
    "score": scores.astype(float),
    "y_pred": y_pred.astype(int),
}).to_csv(OUT_PATH, index=False)

print("[OK] wrote", OUT_PATH)
print("[OK] y distribution:", dict(pd.Series(y).value_counts().sort_index()))