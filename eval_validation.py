"""
Validation Script for Condition-Based Image Editing

This script:
1. Loads the TA-provided validation dataset (unprocessed images)
2. Automatically constructs source-target pairs based on filename conditions
3. Runs inference using a specified model pipeline
4. Computes evaluation metrics:
   - LPIPS (perceptual similarity)
   - SSIM (structural similarity)
   - PSNR (pixel similarity)

"""

import os
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

# =========================
# CONFIG (EDIT THESE)
# =========================

ZIP_PATH = "/path/to/validation.zip"
EXTRACT_PATH = "./val_data"
OUTPUT_DIR = "./outputs_val"

SYSTEM_NAME = "sd_condition_lora"
LORA_PATH = "/path/to/your/lora/checkpoint"

IMAGE_SIZE = 512

# =========================
# EXTRACT DATA
# =========================

def extract_dataset():
    if not os.path.exists(EXTRACT_PATH):
        os.makedirs(EXTRACT_PATH, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
    print(f"Dataset extracted to {EXTRACT_PATH}")

# =========================
# BUILD PAIRS FROM FILENAMES
# =========================

def build_pairs():
    from collections import defaultdict
    
    pairs = defaultdict(dict)
    
    for fname in os.listdir(EXTRACT_PATH):
        if not fname.endswith(".jpg"):
            continue
        
        parts = fname.replace(".jpg", "").split("_")
        scene = parts[0]
        time = parts[1]
        weather = parts[2]
        
        pairs[scene][f"{time}_{weather}"] = fname

    pair_list = []

    for scene, imgs in pairs.items():
        keys = list(imgs.keys())
        for i in range(len(keys)):
            for j in range(len(keys)):
                if i == j:
                    continue

                src = imgs[keys[i]]
                tgt = imgs[keys[j]]

                tgt_time, tgt_weather = keys[j].split("_")

                pair_list.append({
                    "scene": scene,
                    "src": src,
                    "tgt": tgt,
                    "time": tgt_time,
                    "weather": tgt_weather
                })

    print(f"Total pairs constructed: {len(pair_list)}")
    return pair_list

# =========================
# LOAD MODEL PIPELINE
# =========================

def load_model():
    from src.models.inference import load_pipeline
    pipe = load_pipeline(SYSTEM_NAME, lora_path=LORA_PATH)
    return pipe

# =========================
# METRICS
# =========================

lpips_model = lpips.LPIPS(net='alex').cuda()

def compute_lpips(img1, img2):
    t1 = torch.tensor(img1).permute(2,0,1).unsqueeze(0).float() / 255.
    t2 = torch.tensor(img2).permute(2,0,1).unsqueeze(0).float() / 255.
    return lpips_model(t1.cuda(), t2.cuda()).item()

# =========================
# MAIN EVALUATION LOOP
# =========================

def run_evaluation(pipe, pair_list, limit=None):
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    for p in tqdm(pair_list[:limit] if limit else pair_list):

        src_path = os.path.join(EXTRACT_PATH, p["src"])
        tgt_path = os.path.join(EXTRACT_PATH, p["tgt"])

        src_img = Image.open(src_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        tgt_img = Image.open(tgt_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

        # ---- model inference ----
        out = pipe.generate(
            src_img,
            location="penn campus",
            time_of_day=p["time"],
            weather=p["weather"]
        )

        pred = out.image

        save_path = os.path.join(
            OUTPUT_DIR,
            f"{p['scene']}_{p['time']}_{p['weather']}.png"
        )
        pred.save(save_path)

        pred_np = np.array(pred)
        tgt_np = np.array(tgt_img)

        results.append({
            "lpips": compute_lpips(pred_np, tgt_np),
            "ssim": ssim(pred_np, tgt_np, channel_axis=2),
            "psnr": psnr(pred_np, tgt_np)
        })

    df = pd.DataFrame(results)

    print("\n===== FINAL RESULTS =====")
    print(df.mean())

    df.to_csv(os.path.join(OUTPUT_DIR, "val_results.csv"), index=False)

# =========================
# RUN
# =========================

if __name__ == "__main__":
    extract_dataset()
    pairs = build_pairs()
    pipe = load_model()
    run_evaluation(pipe, pairs, limit=20)
