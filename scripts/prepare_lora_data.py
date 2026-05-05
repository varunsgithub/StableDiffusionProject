import json
import shutil
from pathlib import Path

import pandas as pd

from src import config


def pick(row, keys, default=""):
    for key in keys:
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value:
                return value
    return default


def infer_time_weather(condition):
    condition = str(condition).lower()

    time_val = ""
    weather_val = ""

    for t in config.TIME_LABELS:
        if t in condition:
            time_val = t
            break

    for w in config.WEATHER_LABELS:
        if w in condition:
            weather_val = w
            break

    return time_val, weather_val


def main():
    df = pd.read_csv(config.CLEAN_METADATA_CSV)

    train_dir = config.PROCESSED_DIR / "condition_lora_dataset"
    train_dir.mkdir(parents=True, exist_ok=True)

    records = []
    seen = set()

    for _, row in df.iterrows():
        filename = pick(row, ["filename", "file_name", "image", "img"])
        if not filename or filename in seen:
            continue

        seen.add(filename)

        src_path = config.IMAGES_512_DIR / filename
        if not src_path.exists():
            continue

        location = pick(row, ["location", "scene_id", "site", "place"], "a Penn campus building")
        time_val = pick(row, ["time", "time_of_day", "tgt_time"], "")
        weather_val = pick(row, ["weather", "tgt_weather"], "")
        condition = pick(row, ["condition", "tgt_condition", "src_condition"], "")

        if not time_val or not weather_val:
            inferred_time, inferred_weather = infer_time_weather(condition)
            time_val = time_val or inferred_time
            weather_val = weather_val or inferred_weather

        time_val = time_val or "day"
        weather_val = weather_val or "cloudy"

        dst_path = train_dir / filename
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        caption = (
            f"a realistic photo of {location} on University of Pennsylvania campus, "
            f"at {time_val}, with {weather_val} weather, "
            f"campus architecture, photorealistic"
        )

        records.append({
            "file_name": filename,
            "text": caption,
        })

    metadata_path = train_dir / "metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Prepared LoRA dataset at: {train_dir}")
    print(f"Metadata file: {metadata_path}")
    print(f"Number of images: {len(records)}")


if __name__ == "__main__":
    main()
