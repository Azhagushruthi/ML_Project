import pandas as pd
import os

base = r"E:\ML_PROJECT_USTC\merged"
files = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv"
]

cols = [
    "srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes",
    "sttl","dttl","sloss","dloss","service","Sload","Dload","Spkts","Dpkts",
    "swin","dwin","stcpb","dtcpb","smeansz","dmeansz","trans_depth",
    "res_bdy_len","Sjit","Djit","Stime","Ltime","Sintpkt","Dintpkt","tcprtt",
    "synack","ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd",
    "is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm",
    "ct_src_ ltm","ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm",
    "attack_cat","label"
]

dfs = []
for f in files:
    path = os.path.join(base, f)
    print(f"📂 Loading {path}")
    df = pd.read_csv(path, names=cols, skiprows=1, low_memory=False, on_bad_lines="skip")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
print("✅ Combined dataset shape:", merged.shape)

# Drop duplicates + NA
merged.drop_duplicates(inplace=True)
merged.dropna(subset=["label"], inplace=True)

print("🎯 Label distribution:\n", merged["label"].value_counts())

# Save cleaned dataset
out_path = os.path.join(base, "unsw_combined_clean.csv")
merged.to_csv(out_path, index=False)
print(f"💾 Clean dataset saved to: {out_path}")
