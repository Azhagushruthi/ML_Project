# train_flow_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from flow_utils import packets_to_flows  # You already have this

# ============================================================
# CONFIGURATION
# ============================================================
import os
import pandas as pd

# Use the UNSW dataset instead of csv_out
PCAP_CSV_DIR = r"E:\ML_PROJECT_USTC\merged"
DATA_FILE = os.path.join(PCAP_CSV_DIR, "unsw_combined_clean.csv")
   # Folder containing your converted .csv files
MODEL_PATH = r"E:\ML_PROJECT_USTC\models\rf_flow_model.pkl"
SCALER_PATH = r"E:\ML_PROJECT_USTC\models\scaler_flow.pkl"

# Adjust these depending on what CSVs you have
LABEL_MAP = {
    "Benign": 0,
    "Malware": 1
}

# ============================================================
# Step 1: Load and tag CSVs
# ============================================================
def load_and_label_data(csv_dir):
    data_frames = []
    for file in os.listdir(csv_dir):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(csv_dir, file)
        label = 0  # default benign
        for key, val in LABEL_MAP.items():
            if key.lower() in file.lower():
                label = val
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df["label"] = label
            data_frames.append(df)
            print(f"✅ Loaded {file} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Failed to load {file}: {e}")
    if not data_frames:
        raise ValueError("No valid CSV files found in csv_out directory.")
    return pd.concat(data_frames, ignore_index=True)

# ============================================================
# Step 2: Convert packets → flows
# ============================================================
def build_flow_dataset(packet_df):
    flows = packets_to_flows(packet_df)
    flows["label"] = packet_df["label"].mode()[0]  # rough label propagation
    print(f"🔁 Aggregated into {len(flows)} flows.")
    return flows

# ============================================================
# Step 3: Main
# ============================================================
if __name__ == "__main__":
    print("📥 Loading UNSW packet dataset...")
    pkt_df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"✅ Loaded UNSW dataset: {pkt_df.shape[0]} rows, {pkt_df.shape[1]} columns")

    # Make sure label column exists
    if "label" not in pkt_df.columns:
        label_col = "Label" if "Label" in pkt_df.columns else "attack_cat"
        pkt_df["label"] = pkt_df[label_col].apply(
            lambda x: 0 if str(x).lower() in ["normal", "benign"] else 1
        )

    # Choose only numeric columns (ML needs numbers)
    numeric_cols = pkt_df.select_dtypes(include=[np.number]).columns.tolist()

    MODEL_FEATURES = [
        'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
        'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
        'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
        'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_src_ltm'
    ]
    MODEL_FEATURES = [f for f in MODEL_FEATURES if f in pkt_df.columns]

    print(f"✅ Using {len(MODEL_FEATURES)} features: {MODEL_FEATURES}")

    X = pkt_df[MODEL_FEATURES].fillna(0.0)
    y = pkt_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🧮 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("🚀 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=120, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n🧠 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Scaler saved to: {SCALER_PATH}")
