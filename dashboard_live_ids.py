import streamlit as st
import pandas as pd
import numpy as np
import pyshark
import joblib
import threading
import time
from queue import Queue
from sklearn.preprocessing import StandardScaler

# ===============================
# Load trained model and scaler
# ===============================
MODEL_PATH = r"E:\ML_PROJECT_USTC\models\unsw_rf_model.pkl"
SCALER_PATH = r"E:\ML_PROJECT_USTC\models\unsw_scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===============================
# Streamlit UI Setup
# ===============================
st.set_page_config(page_title="Live Network IDS", layout="wide", page_icon="🧠")
st.title("🧠 Real-Time Intrusion Detection (Live Mode)")

col1, col2 = st.columns(2)
iface = col1.text_input("🌐 Network Interface (e.g., Wi-Fi, Ethernet)", "Wi-Fi")
packet_limit = col2.number_input("Max Packets to Capture", min_value=50, max_value=5000, value=500, step=50)

start_btn = st.button("🚀 Start Live Capture")

placeholder_table = st.empty()
placeholder_chart = st.empty()

# ===============================
# Helper Functions
# ===============================

def extract_features(pkt):
    """Extract lightweight numerical features from packet"""
    try:
        length = int(pkt.length) if hasattr(pkt, "length") else 0
        protocol = pkt.highest_layer
        proto_map = {"TCP": 6, "UDP": 17, "ICMP": 1}
        proto_val = proto_map.get(protocol.upper(), 0)

        return pd.DataFrame([{
            "frame_len": length,
            "ip_proto": proto_val
        }])
    except Exception:
        return None

def predict_packet(packet):
    """Preprocess and predict a single packet"""
    feats = extract_features(packet)
    if feats is None:
        return None
    feats_scaled = scaler.transform(feats)
    pred = model.predict(feats_scaled)[0]
    return pred

# ===============================
# Live Capture Logic
# ===============================

def live_capture(q, iface, max_packets):
    capture = pyshark.LiveCapture(interface=iface)
    for i, pkt in enumerate(capture.sniff_continuously(packet_count=max_packets)):
        q.put(pkt)
        if i >= max_packets:
            break

# ===============================
# Main Streamlit Loop
# ===============================
if start_btn:
    st.info("🟢 Starting live packet capture... this may require admin privileges!")
    q = Queue()
    t = threading.Thread(target=live_capture, args=(q, iface, packet_limit))
    t.start()

    packet_data = []
    normal_count, attack_count = 0, 0

    while t.is_alive() or not q.empty():
        try:
            pkt = q.get(timeout=1)
            pred = predict_packet(pkt)
            if pred == 1:
                attack_count += 1
                packet_data.append(("⚠️ Attack", pkt.highest_layer, pkt.ip.src if hasattr(pkt, "ip") else "-", pkt.length))
            else:
                normal_count += 1
                packet_data.append(("✅ Normal", pkt.highest_layer, pkt.ip.src if hasattr(pkt, "ip") else "-", pkt.length))
        except Exception:
            continue

        df = pd.DataFrame(packet_data, columns=["Status", "Protocol", "Source IP", "Length"])
        placeholder_table.dataframe(df.tail(20))

        chart_df = pd.DataFrame({"Type": ["Normal", "Attack"], "Count": [normal_count, attack_count]})
        placeholder_chart.bar_chart(chart_df.set_index("Type"))

    st.success("✅ Live capture finished!")
# streamlit_live_demo.py
import streamlit as st
from predict_live_pipeline import predict_from_live

st.set_page_config(layout="wide", page_title="Live IDS Demo")

st.title("Live capture → predict demo")
iface = st.text_input("TShark interface index (from `tshark -D`)", value="5")
count = st.slider("Packets to capture", 10, 1000, 100)
timeout = st.slider("Capture timeout (s)", 10, 120, 45)

if st.button("Run live capture & predict"):
    with st.spinner("Capturing and predicting... (requires admin privileges)"):
        try:
            out = predict_from_live(interface_index=iface, count=count, timeout=timeout)
            st.success(f"Done — {len(out)} flows predicted, saved to live_predictions.csv")
            st.dataframe(out.head(50))
            # quick pie
            counts = out['pred_label'].value_counts()
            st.write("Prediction counts")
            st.bar_chart(counts)
            counts = out['pred_label'].value_counts()
            st.write("Prediction counts")
            st.bar_chart(counts)

            # Add detection message
            benign = counts.get(0, 0)
            attacks = counts.get(1, 0)

            if attacks > 0:
                st.error(f"🚨 Intrusion detected! {attacks} malicious flows found.")
            else:
                st.success(f"✅ No intrusion detected. All {benign} flows are benign.")

        except Exception as e:
            st.error(f"Error: {e}")

