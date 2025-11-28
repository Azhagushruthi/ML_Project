# live_capture.py
import subprocess, shlex, tempfile, os, pandas as pd, shutil

TSHARK = r"E:\Wireshark\tshark.exe"  # update path or "tshark" if in PATH

def live_capture_to_df(interface, count=200, timeout=30, keep_tmp=False):
    """
    Capture 'count' packets on 'interface' (index or name) and return pandas DataFrame.
    Must be run from an Admin terminal if running on Windows and using interfaces.
    """
    tmpdir = tempfile.mkdtemp(prefix="livecap_")
    pcap_path = os.path.join(tmpdir, "capture.pcap")
    csv_path = os.path.join(tmpdir, "capture.csv")

    cap_cmd = f'"{TSHARK}" -i {interface} -c {count} -w "{pcap_path}"'
    try:
        subprocess.run(shlex.split(cap_cmd), capture_output=True, text=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Capture timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"tshark capture failed: {e.stderr.strip()}") from e

    fields = ["frame.number","frame.time_epoch","ip.src","ip.dst","frame.len","_ws.col.Protocol",
              "tcp.srcport","tcp.dstport","udp.srcport","udp.dstport"]
    field_args = " ".join([f"-e {f}" for f in fields])
    csv_cmd = f'"{TSHARK}" -r "{pcap_path}" -T fields -E header=y -E separator=, -E quote=d -E occurrence=f {field_args}'
    try:
        with open(csv_path, "w", encoding="utf-8") as f:
            subprocess.run(shlex.split(csv_cmd), stdout=f, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"tshark csv conversion failed: {e.stderr.strip()}") from e

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        # fallback if headers missing
        df = pd.read_csv(csv_path, names=fields, header=0, low_memory=False)

    if not keep_tmp:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    return df
