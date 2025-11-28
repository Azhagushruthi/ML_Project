import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ------------------ CONFIG ------------------
DATA_PATH = r"E:\ML_PROJECT_USTC\merged\unsw_combined_clean.csv"
MODEL_PATH = r"E:\ML_PROJECT_USTC\models\unsw_rf_model.pkl"
SCALER_PATH = r"E:\ML_PROJECT_USTC\models\unsw_scaler.pkl"
LABEL_PATH = r"E:\ML_PROJECT_USTC\models\unsw_labelencoder.pkl"
# --------------------------------------------

print(f"📂 Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"✅ Loaded dataset: {df.shape}")

# Fill missing attack_cat if present
if "attack_cat" in df.columns:
    df["attack_cat"].fillna("normal", inplace=True)

# Label encode categorical features
cat_cols = ["proto", "state", "service"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype(str)
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])

# Replace inf with NaN and fill NaN with median
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Split features and target
X = df.drop(columns=["label"])
y = df["label"]

# Ensure numeric only
X = X.select_dtypes(include=[np.number])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🧮 Numeric features:", X_train.shape[1])
print("🧩 Balancing with SMOTE...")
sm = SMOTE(random_state=42, sampling_strategy="auto")
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print("✅ After SMOTE:", X_train_bal.shape)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Train model
print("🚀 Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_bal)

# Evaluate
y_pred = rf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {acc:.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧠 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save artifacts
joblib.dump(rf, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")
print(f"💾 Scaler saved to: {SCALER_PATH}")
