import pandas as pd
import os
import glob

CSV_FOLDER = r"E:\ML_PROJECT_USTC\csv_out"
OUTPUT_FILE = r"E:\ML_PROJECT_USTC\merged\all_flows.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def load_and_label(file_path):
    label = os.path.basename(file_path).replace(".csv", "")
    df = pd.read_csv(file_path, low_memory=False)
    df["label"] = label
    return df

def main():
    all_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    print(f"📦 Found {len(all_files)} CSVs")

    dfs = []
    for file in all_files:
        try:
            df = load_and_label(file)
            dfs.append(df)
            print(f"✅ Loaded: {os.path.basename(file)} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Combined dataset saved → {OUTPUT_FILE}")
        print(f"📊 Final shape: {combined.shape}")
    else:
        print("❌ No valid CSVs found")

if __name__ == "__main__":
    main()
