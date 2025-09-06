import re
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "/content/drive/MyDrive/xgboost_model.joblib"  

N_KEYPOINTS = 21
FEATURE_NAMES = [f"{ax}{i}_rel" for i in range(N_KEYPOINTS) for ax in ("x","y","z")]

KP_MAP = {
    0: "Wrist",
    1: "Thumb1", 2: "Thumb2", 3: "Thumb3", 4: "Thumb4",
    5: "Index1", 6: "Index2", 7: "Index3", 8: "Index4",
    9: "Middle1", 10: "Middle2", 11: "Middle3", 12: "Middle4",
    13: "Ring1", 14: "Ring2", 15: "Ring3", 16: "Ring4",
    17: "Pinky1", 18: "Pinky2", 19: "Pinky3", 20: "Pinky4"
}

def unwrap_estimator(obj):
    """Vrati stvarni estimator iz GridSearchCV/Pipeline/itd."""
    if hasattr(obj, "best_estimator_"):
        return unwrap_estimator(obj.best_estimator_)
    if hasattr(obj, "steps"):
        return unwrap_estimator(obj.steps[-1][1])
    if hasattr(obj, "estimator"):
        return unwrap_estimator(obj.estimator)
    if hasattr(obj, "base_estimator"):
        return unwrap_estimator(obj.base_estimator)
    return obj

def get_feature_importances(est):
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        names = FEATURE_NAMES
        if imp.size != len(names):
            names = [f"f{i}" for i in range(imp.size)]
        return imp, names
    
    if hasattr(est, "get_booster"):
        booster = est.get_booster()
        score = booster.get_score(importance_type="gain")
        imp = np.zeros(len(FEATURE_NAMES), dtype=float)
        pattern = re.compile(r"^f(\d+)$")
        for k, v in score.items():
            m = pattern.match(k)
            if m:
                idx = int(m.group(1))
                if idx < len(imp):
                    imp[idx] = float(v)
        return imp, FEATURE_NAMES
    
    raise RuntimeError("Error - features")

model = joblib.load(MODEL_PATH)
est = unwrap_estimator(model)
importances, names = get_feature_importances(est)

# Normalize
if importances.sum() > 0:
    importances = importances / importances.sum()

# DataFrame 
df = pd.DataFrame({"feature": names, "importance": importances})
df = df.sort_values("importance", ascending=False).reset_index(drop=True)

print("Top 10 features:")
print(df.head(10).to_string(index=False))

# Agg
def kp_idx(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None

df["kp_idx"] = df["feature"].apply(kp_idx)
agg = df.dropna(subset=["kp_idx"]).groupby("kp_idx", as_index=False)["importance"].sum()
agg = agg.sort_values("importance", ascending=False)
agg["keypoint_name"] = agg["kp_idx"].map(KP_MAP)

print("\nTop 10 features (sum x/y/z):")
print(agg.head(10).to_string(index=False))
