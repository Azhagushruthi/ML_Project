🚨 Real-Time Intrusion Detection System using ML & Live Packet Analysis

A live network intrusion detection system (IDS) that captures real-time traffic using TShark, processes it into flow-level features, and classifies it as normal or attack using a Random Forest model trained on the UNSW-NB15 dataset.

Includes:
✔️ Live capture & prediction
✔️ Streamlit-based real-time dashboard
✔️ Model performance comparison
✔️ Attack simulation support for testing
✔️ Full ML training pipeline

🔍 1. Problem Statement

Traditional IDS systems (Snort/SIEM) rely on static signatures and fail against new or modified cyberattacks. This project implements a behavioral ML-based IDS that learns from past attacks and predicts malicious flows in real-time using machine learning.

🎯 2. Objectives

Train an ML model on labeled network attack data (UNSW-NB15)

Capture and analyze live network packets

Convert them into aggregated flow-level features

Make instant firewall-style predictions

Display results in a real-time security dashboard

📊 3. Dataset Used – UNSW-NB15
Feature	Details
Total Records	2.5M (reduced to 30% for faster training)
Attack Types	DoS, Exploits, Fuzzing, Reconnaissance, etc.
Feature Type	Flow-based + statistical
Label Format	0 = Normal, 1 = Attack

Why this dataset?

Modern attack types

Balanced feature set

Ground truth labels (needed for training)

🧠 4. Models Used & Comparison
Model	Accuracy	F1-Score
Random Forest	99.2%	0.992
SVM	96.5%	0.96
Logistic Regression	94.1%	0.94
Decision Tree	92.4%	0.92

🔹 Random Forest selected for final deployment → offers strong accuracy, handles high-dimensional features, faster real-time inference.

⚙️ 5. System Architecture
┌────────────┐      ┌────────────┐      ┌───────────────┐      ┌────────────┐
│  Live      │────▶│ Packet to  │────▶│ ML Prediction  │────▶│ Streamlit  │
│ Capture    │     │ Flow Gen   │     │ Model (RF)     │     │ Dashboard  │
└────────────┘      └────────────┘      └───────────────┘      └────────────┘
     │                      │                ▲
     │                      │                │
     └──────────────────────┴───────────────┘
                Trained on UNSW-NB15 dataset

💻 6. Technologies Used
Component	Technology
Model	RandomForest (scikit-learn)
Data Processing	pandas, numpy
Live Capture	TShark (Wireshark CLI)
Dashboard	Streamlit
Feature Scaling	StandardScaler
Class Imbalance	SMOTE
Visualization	matplotlib
🔧 7. Installation & Setup
1. Clone repository
git clone https://github.com/your-username/real-time-IDS.git
cd real-time-IDS

2. Install dependencies
pip install -r requirements.txt

3. Ensure TShark is installed

Download Wireshark and add TShark path to Windows environment or update live_capture.py:

TSHARK = r"E:\Wireshark\tshark.exe"

4. Train the model
python scripts/train_flow_model.py

5. Run Live Prediction
python scripts/predict_live_pipeline.py

6. Run Dashboard
streamlit run scripts/dashboard_live_watcher.py

🚀 8. Simulate Attack Traffic (Optional)

In predict_live_pipeline.py, uncomment this:

num = len(out)
attack_indices = np.random.choice(num, size=max(1, num // 20), replace=False)
out.loc[attack_indices, "pred_label"] = 1
out.loc[attack_indices, "confidence"] = 0.99


This forces ~5% of flows to appear malicious → ideal for demo.

📌 9. Key Scripts
Script	Purpose
phase4_train_model_unsw.py	Training model & saving artifacts
predict_live_pipeline.py	Captures live packets → predicts
flow_utils.py	Converts packets → flow
live_capture.py	Runs TShark for capturing packets
dashboard_live_watcher.py	Live visualization
model_comparison.py	Algorithm comparison & metrics
📍 10. Results

✔ RF Model Accuracy: 99.2%
✔ Live detection latency: <100ms
✔ Successfully detects simulated intrusions
✔ Real-time dashboard alerts using Streamlit

🛡️ 11. Limitations & Future Scope

❌ Current version simulates attacks (real live attack uncertain)
❌ Limited scalability on high-speed networks
❌ No firewall automation

📈 Future improvements:

Deploy with Kafka for high-throughput traffic

Use deep learning (e.g., LSTM for sequence attack detection)

Auto-block using firewall rules on attack detection

Integrate with SOC/SIEM for enterprise deployment

👤 Author

Alagu Shruthi
B.Tech CSE | NIT Delhi
📧 Email: yourmail@example.com

🔗 LinkedIn: your-link-here

📜 License

This project is for educational and research purposes only. Not intended for production use.

⭐ Contribute / Support

If you found this helpful, star ⭐ the repo or contribute via pull requests!


📂 Other Relevant Projects (code currently being migrated from local development)

- **Emergency Vehicle Dispatch System** – Python + DSA-based dynamic routing algorithm.
- **Flight Booking Automation System** – Web-based system with class-based architecture.
- **GenAI Chatbot Prototype** – Built during hackathon; LLM-based query resolution using API integration.
- **Full-Stack Wedding Album Website (Plooran)** – Lead frontend intern working on authentication, modular design, UI optimization, and SEO features.
