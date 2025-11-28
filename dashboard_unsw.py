import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px

# === Paths ===
MODEL_PATH = r"E:\ML_PROJECT_USTC\models\unsw_rf_model.pkl"
SCALER_PATH = r"E:\ML_PROJECT_USTC\models\unsw_scaler.pkl"
DATA_PATH = r"E:\ML_PROJECT_USTC\merged\unsw_combined_clean.csv"

# === Load Model + Scaler ===
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    data = pd.read_csv(DATA_PATH, low_memory=False)
    return model, scaler, data

model, scaler, data = load_assets()

st.title("🧠 Real-Time Network Intrusion Detection Dashboard (UNSW-NB15)")
st.caption("Simulated live packet classification using your trained RandomForest model")

# === Encode Categorical Columns ===
cat_cols = ["proto", "service", "state"]
for col in cat_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    else:
        st.warning(f"⚠️ Column '{col}' missing — skipping encoding")

# === Prepare Data ===
if "label" not in data.columns:
    st.error("❌ 'label' column missing in dataset!")
    st.stop()

data = data.sample(frac=1, random_state=42).reset_index(drop=True)
X = data.drop(columns=["label"], errors="ignore")
y = data["label"]

# Ensure all numeric
X_scaled = scaler.transform(X.select_dtypes(include=np.number))

# === Sidebar Controls ===
st.sidebar.header("⚙️ Simulation Controls")
batch_size = st.sidebar.slider("Packets per update", 50, 500, 100, 50)
speed = st.sidebar.slider("Update speed (seconds)", 0.2, 3.0, 1.0, 0.1)
max_packets = st.sidebar.slider("Max packets to simulate", 1000, len(X), 5000, 500)

placeholder = st.empty()
attack_count, normal_count = 0, 0
history = []

if st.sidebar.button("🚀 Start Simulation"):
    st.sidebar.success("Simulation running...")

    for i in range(0, max_packets, batch_size):
        X_batch = X_scaled[i:i+batch_size]
        y_batch = y.iloc[i:i+batch_size]

        preds = model.predict(X_batch)

        # Count
        attack_count += np.sum(preds == 1)
        normal_count += np.sum(preds == 0)
        history.extend(preds)

        acc = accuracy_score(y_batch, preds)
        report = classification_report(y_batch, preds, output_dict=True)

        with placeholder.container():
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Accuracy (this batch)", f"{acc*100:.2f}%")
                st.metric("Normal Packets", normal_count)
                st.metric("Attack Packets", attack_count)

            with col2:
                fig_pie = px.pie(
                    names=["Normal", "Attack"],
                    values=[normal_count, attack_count],
                    color=["Normal", "Attack"],
                    color_discrete_map={"Normal": "green", "Attack": "red"},
                    title="Traffic Composition"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("### 📈 Recent Predictions (last 100)")
            recent = history[-100:]
            chart_data = pd.DataFrame({
                "Packet #": list(range(len(recent))),
                "Prediction": ["Attack" if x == 1 else "Normal" for x in recent]
            })
            fig_line = px.line(chart_data, x="Packet #", y="Prediction", color="Prediction", markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

            st.markdown(f"### 🧾 Batch {i//batch_size + 1} Summary")
            st.json(report)

        time.sleep(speed)

    st.success("✅ Simulation complete!")

else:
    st.info("Press '🚀 Start Simulation' in the sidebar to begin live monitoring.")
