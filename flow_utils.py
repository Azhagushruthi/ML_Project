import pandas as pd
import numpy as np

def packets_to_flows(pkt_df, src_col="ip.src", dst_col="ip.dst", proto_col="_ws.col.Protocol",
                     time_col="frame.time_epoch", len_col="frame.len"):
    """
    Convert per-packet DataFrame into aggregated per-flow DataFrame.
    Returns flow-level rows with a broad set of features.
    """
    df = pkt_df.copy()

    # Ensure numeric types
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[len_col] = pd.to_numeric(df[len_col], errors="coerce")

    for c in ["tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[time_col, len_col], how="any")

    # Handle protocol column
    if proto_col in df.columns:
        df["protocol"] = df[proto_col].astype(str).str.lower().fillna("unk")
    else:
        df["protocol"] = "unk"

    # Build flow_id
    df["flow_id"] = df[src_col].astype(str) + "-" + df[dst_col].astype(str) + "-" + df["protocol"]

    # Sort by flow and time
    df = df.sort_values(["flow_id", time_col])

    # Inter-arrival time
    df["iat"] = df.groupby("flow_id")[time_col].diff().fillna(0.0)

    # Aggregate features per flow
    agg = df.groupby("flow_id").agg(
        src=(src_col, "first"),
        dst=(dst_col, "first"),
        protocol=("protocol", "first"),
        packet_count=(len_col, "count"),
        total_bytes=(len_col, "sum"),
        avg_packet_size=(len_col, "mean"),
        packet_size_var=(len_col, "var"),
        duration=(time_col, lambda x: x.max() - x.min()),
        iat_mean=("iat", "mean"),
        iat_std=("iat", "std"),
        iat_max=("iat", "max"),
        iat_min=("iat", "min"),
        avg_duration_per_packet=(time_col, lambda x: (x.max() - x.min()) / max(len(x), 1)),
    ).reset_index()

    # Handle NaNs and zeros
    agg["duration"] = agg["duration"].replace(0, 1e-6).fillna(1e-6)

    # Derived features
    agg["bytes_per_sec"] = agg["total_bytes"] / agg["duration"]
    agg["packets_per_sec"] = agg["packet_count"] / agg["duration"]
    agg["burstiness"] = agg["avg_packet_size"] / (agg["packet_count"] + 1e-6)

    # Private IP direction heuristic
    def is_private(ip):
        try:
            if ip.startswith("10.") or ip.startswith("192.168."):
                return True
            if ip.startswith("172."):
                second = int(ip.split(".")[1])
                return 16 <= second <= 31
            return False
        except Exception:
            return False

    agg["direction_from_private"] = agg.apply(
        lambda r: 1 if is_private(str(r["src"])) and not is_private(str(r["dst"])) else 0,
        axis=1
    )

    # Fill numeric NaNs
    num_cols = agg.select_dtypes(include=[np.number]).columns
    agg[num_cols] = agg[num_cols].fillna(0.0)

    return agg
