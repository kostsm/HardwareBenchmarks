import os
import json
import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt


# ----- Config -----
INPUT_FOLDER = "gpu_architecture_details"
OUTPUT_FILE = "ml_gpu_scores.xlsx"

WEIGHTS = {
    "Tensor Cores": 0.30,
    "FP16 TFLOPS": 0.25,
    "Memory Bandwidth (GB/s)": 0.25,
    "Memory Size (GB)": 0.10,
    "L2 Cache (MB)": 0.10,
}

# ----- Helpers -----
def parse_numeric(val):
    if not val or not isinstance(val, str):
        return 0.0
    match = re.search(r"[\d.]+", val.replace(",", ""))
    return float(match.group(0)) if match else 0.0

def convert_fp16(val):
    if not val or not isinstance(val, str):
        return 0.0
    val = val.upper()
    match = re.search(r"([\d.]+)\s*(T|G|M)FLOPS", val)
    if not match:
        return 0.0
    num = float(match.group(1))
    unit = match.group(2)
    if unit == "T":
        return num
    elif unit == "G":
        return num / 1000
    elif unit == "M":
        return num / 1_000_000
    return 0.0

def convert_bandwidth(val):
    if not val or not isinstance(val, str):
        return 0.0
    val = val.upper()
    match = re.search(r"([\d.]+)\s*(T|G|M)B/S", val)
    if not match:
        return 0.0
    num = float(match.group(1))
    unit = match.group(2)
    if unit == "T":
        return num * 1000
    elif unit == "G":
        return num
    elif unit == "M":
        return num / 1000
    return 0.0

def convert_memory_size(val):
    if not val or not isinstance(val, str):
        return 0.0
    val = val.upper()
    match = re.search(r"([\d.]+)\s*(T|G|M)B", val)
    if not match:
        return 0.0
    num = float(match.group(1))
    unit = match.group(2)
    if unit == "T":
        return num * 1000
    elif unit == "G":
        return num
    elif unit == "M":
        return num / 1000
    return 0.0

def convert_cache(val):
    if not val or not isinstance(val, str):
        return 0.0
    val = val.upper()
    if "MB" in val:
        return parse_numeric(val)
    elif "KB" in val:
        return parse_numeric(val) / 1000.0
    return 0.0

def parse_release_date(date_str):
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip().lower()
    if "never released" in date_str or "unknown" in date_str or date_str == "none":
        return None

    # Remove ordinal suffixes: 1st, 2nd, 3rd, 21st, etc.
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)

    # Try full formats first
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %Y", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


# ----- Scoring -----
def compute_score(row):
    score = 0.0
    for field, weight in WEIGHTS.items():
        score += weight * row.get(field, 0.0)
    return score

# ----- Core Processing -----
def score_gpus_from_json(json_data):
    gpu_rows = []

    for card_name, card_data in json_data.items():
        date_str = card_data.get("Graphics Card", {}).get("Release Date")
        release_date = parse_release_date(date_str)
        if not release_date or release_date.year < 2010:
            if release_date == None:
                print(card_name, release_date, date_str)
            continue

        tensor_cores_str = card_data.get("Render Config", {}).get("Tensor Cores", "")
        fp16_str = card_data.get("Theoretical Performance", {}).get("FP16 (half)", "")
        mem_bw_str = card_data.get("Memory", {}).get("Bandwidth", "")
        mem_size_str = card_data.get("Memory", {}).get("Memory Size", "")
        l2_cache_str = card_data.get("Render Config", {}).get("L2 Cache", "")

        tensor_cores = parse_numeric(tensor_cores_str)
        fp16 = convert_fp16(fp16_str)
        mem_bw = convert_bandwidth(mem_bw_str)
        mem_size = convert_memory_size(mem_size_str)
        l2_cache = convert_cache(l2_cache_str)

        row = {
            "Card Name": card_name,
            "Release Date": release_date,
            "Release Date Original": date_str,
            "Tensor Cores": tensor_cores,
            "Tensor Cores Original": tensor_cores_str,
            "FP16 TFLOPS": fp16,
            "FP16 TFLOPS Original": fp16_str,
            "Memory Bandwidth (GB/s)": mem_bw,
            "Memory Bandwidth Original": mem_bw_str,
            "Memory Size (GB)": mem_size,
            "Memory Size Original": mem_size_str,
            "L2 Cache (MB)": l2_cache,
            "L2 Cache Original": l2_cache_str,
        }

        row["ML Performance Score"] = compute_score(row)
        gpu_rows.append(row)

    return pd.DataFrame(gpu_rows)

# ----- Batch Load -----
def process_all_jsons(folder_path):
    all_gpu_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            path = os.path.join(folder_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    all_gpu_data.update(data)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse {filename}: {e}")

    df = score_gpus_from_json(all_gpu_data)
    df.sort_values("ML Performance Score", ascending=False, inplace=True)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Scoring complete. Output written to {OUTPUT_FILE}")

    # --- Plot Release Date vs ML Score ---
    df_plot = df.dropna(subset=["Release Date"])
    plt.figure(figsize=(12, 6))
    plt.scatter(df_plot["Release Date"], df_plot["ML Performance Score"], alpha=0.7, color='teal')
    plt.title("GPU ML Performance Score vs Release Date")
    plt.xlabel("Release Date")
    plt.ylabel("ML Performance Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ml_gpu_score_vs_release_date.png")
    print("Scatter plot saved as ml_gpu_score_vs_release_date.png")


# ----- Run -----
if __name__ == "__main__":
    process_all_jsons(INPUT_FOLDER)
