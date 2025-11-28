import os
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP
from tqdm import tqdm

# Input and output directories
INPUT_DIR = r"E:\ML_PROJECT_USTC\pcaps"
OUTPUT_DIR = r"E:\ML_PROJECT_USTC\csv_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features(pcap_path):
    """Extract basic features from a PCAP file using Scapy."""
    packets = rdpcap(pcap_path)
    data = []

    for pkt in packets:
        # Only process packets with IP layer
        if IP in pkt:
            src = pkt[IP].src
            dst = pkt[IP].dst
            length = len(pkt)
            proto = pkt[IP].proto  # numeric protocol ID

            # Initialize ports
            sport = dport = 0
            protocol_name = "OTHER"

            if TCP in pkt:
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
                protocol_name = "TCP"
            elif UDP in pkt:
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport
                protocol_name = "UDP"

            data.append([src, dst, length, proto, sport, dport, protocol_name])

    df = pd.DataFrame(data, columns=["ip.src", "ip.dst", "frame.len", "ip.proto", "srcport", "dstport", "protocol"])
    return df

def main():
    pcap_files = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.endswith(".pcap"):
                pcap_files.append(os.path.join(root, f))

    print(f"📦 Found {len(pcap_files)} pcap files")

    for pcap_path in tqdm(pcap_files, desc="Processing PCAPs"):
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(pcap_path).replace(".pcap", ".csv"))
        if os.path.exists(out_path):
            continue
        try:
            df = extract_features(pcap_path)
            df.to_csv(out_path, index=False)
        except Exception as e:
            print(f"⚠️ Error reading {pcap_path}: {e}")

    print("✅ Done converting all PCAPs!")

if __name__ == "__main__":
    main()
