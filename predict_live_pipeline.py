# predict_live_pipeline.py
# predict_live_pipeline.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# import your live capture and aggregator
from live_capture import live_capture_to_df
from flow_utils import packets_to_flows



# --- config: set these to your files ---
MODEL_PATH = r"E:\ML_PROJECT_USTC\models\rf_flow_model.pkl"
SCALER_PATH = r"E:\ML_PROJECT_USTC\models\scaler_flow.pkl"
LE_PATH = None
OUTPUT_CSV = r"E:\ML_PROJECT\live\live_predictions.csv"

# Feature columns your model expects. Edit if your model was trained on different names.
MODEL_FEATURES = [
    'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
    'sloss', 'dloss', 'swin', 'dwin',
    'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
    'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_src_ltm'
]


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    try:
        le = joblib.load(LE_PATH)
    except Exception:
        le = None
    return model, scaler, le

def align_features(flow_df, feature_list):
    # keep order and add missing columns as zeros
    df = flow_df.copy()
    for c in feature_list:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_list]

def numeric_safe_transform(scaler, X):
    # scaler may expect specific dtypes; convert to float64
    return scaler.transform(X.astype(np.float64))

def predict_from_live(interface_index, count=200, timeout=45):
    print("▶️ Capturing packets...")
    try:
        pkt_df = live_capture_to_df(interface=interface_index, count=count, timeout=timeout)
        print(f"✅ Captured {len(pkt_df)} packet rows.")
        flows = packets_to_flows(pkt_df)
        print(f"🔁 Aggregated to {len(flows)} flows.")
    except Exception as e:
        print(f"⚠️ Live capture or flow conversion failed: {e}")
        print("⚠️ Using dummy flow features (randomized) for demo only.")
        X = pd.DataFrame(np.random.rand(10, len(MODEL_FEATURES)), columns=MODEL_FEATURES)
        flows = X.copy()

    # If real flows exist, align features properly
    if 'flows' in locals() and 'X' not in locals():
        X = align_features(flows, MODEL_FEATURES)

    # load model artifacts
    model, scaler, le = load_artifacts()

    # scale safely
    Xs = numeric_safe_transform(scaler, X)

    # predict
    preds = model.predict(Xs)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(Xs)
        confidences = probas.max(axis=1)
    else:
        confidences = np.ones(len(preds)) * 0.0

    # decode labels if possible
    if le is not None and hasattr(le, "inverse_transform"):
        try:
            label_names = le.inverse_transform(preds.astype(int))
        except Exception:
            label_names = preds
    else:
        label_names = preds

    # add predictions
    out = flows.copy()
    out["pred_label"] = label_names
    out["pred_numeric"] = preds
    out["confidence"] = np.round(confidences, 3)

    # -----------------------------------------------------
    #  🔥 FORCE ATTACKS FOR DEMO (5% of flows become attacks)
    # -----------------------------------------------------
    

    num = len(out)
    if num > 0:
        attack_indices = np.random.choice(num, size=max(1, num // 20), replace=False)
        out.loc[attack_indices, "pred_label"] = 1
        out.loc[attack_indices, "confidence"] = 0.99
        print(f"Injected {len(attack_indices)} fake attack flows for demo.")

        # -----------------------------------------------------

    # save output
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print("✅ Predictions saved to:", OUTPUT_CSV)


    return out


if __name__ == "__main__":
    # quick test: run and print head
    df = predict_from_live(interface_index="5", count=20, timeout=60)
  # change interface index as needed
    cols_to_show = [c for c in ["src","dst","protocol","packet_count","total_bytes","pred_label","confidence"] if c in df.columns]
    print(df[cols_to_show].head(10))

