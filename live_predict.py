import subprocess, shlex, tempfile, os, pandas as pd, time

# ✅ Path to your Wireshark tshark executable
TSHARK = r"E:\Wireshark\tshark.exe"   # Update this if Wireshark is installed elsewhere

# ✅ Choose the interface number from list_tshark_interfaces() output (usually Wi-Fi = 5)
INTERFACE = "5"

# ✅ Configuration
COUNT = 50       # Number of packets to capture
TIMEOUT = 30     # Max seconds to wait

# ✅ Temporary files
tmpdir = tempfile.mkdtemp(prefix="livecap_")
pcap_path = os.path.join(tmpdir, "capture.pcap")
csv_path = os.path.join(tmpdir, "capture.csv")

print(f"\n📂 Temporary folder: {tmpdir}")

# -------------------------------
# 1️⃣ CAPTURE LIVE TRAFFIC
# -------------------------------
cap_cmd = f'"{TSHARK}" -i {INTERFACE} -c {COUNT} -w "{pcap_path}"'
print(f"\n▶️ Running capture:\n{cap_cmd}")

try:
    result = subprocess.run(
        shlex.split(cap_cmd),
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=True
    )
    print("✅ Capture complete.")
except subprocess.TimeoutExpired:
    print("⚠️ Capture timed out. Partial file may exist.")
except FileNotFoundError:
    raise SystemExit("❌ Error: tshark.exe not found. Check your TSHARK path.")
except subprocess.CalledProcessError as e:
    raise SystemExit(f"❌ tshark capture failed:\n{e.stderr}")

print("📦 Saved pcap:", pcap_path)

# -------------------------------
# 2️⃣ CONVERT PCAP → CSV
# -------------------------------
fields = [
    "frame.number",
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "frame.len",
    "_ws.col.Protocol",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport"
]

field_args = " ".join([f"-e {f}" for f in fields])
csv_cmd = (
    f'"{TSHARK}" -r "{pcap_path}" -T fields '
    f'-E header=y -E separator=, -E quote=d -E occurrence=f {field_args}'
)

print("\n▶️ Converting to CSV...")

try:
    with open(csv_path, "w", encoding="utf-8") as f:
        subprocess.run(
            shlex.split(csv_cmd),
            stdout=f,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
    print("✅ CSV created:", csv_path)
except subprocess.CalledProcessError as e:
    raise SystemExit(f"❌ Conversion failed:\n{e.stderr}")
except FileNotFoundError:
    raise SystemExit("❌ Error: tshark.exe not found. Check your TSHARK path.")

# -------------------------------
# 3️⃣ LOAD INTO PANDAS
# -------------------------------
print("\n📊 Loading CSV into DataFrame...")

try:
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅ Loaded {len(df)} rows.")
    print(df.head(10))
except pd.errors.EmptyDataError:
    raise SystemExit("⚠️ No packets captured — CSV is empty.")
except Exception as e:
    raise SystemExit(f"❌ Error reading CSV:\n{e}")
