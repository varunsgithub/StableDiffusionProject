import json
import shutil
from pathlib import Path
import pandas as pd
from src import config


def main():
    df = pd.read_csv(config.CLEAN_METADATA_CSV)

    train_dir = config.PROCESSED_DIR / "condition_lora_dataset"
    train_dir.mkdir(parents=True, exist_ok=True)

    def pick(row, keys, default=""):
        for k in keys:
            if k in row and pd.notna(row[k]):
                val = str(row[k]).strip()
                if val:
                    return val
        return default

    records = []

    for _, row in df.iterrows():
        filename = pick(row, ["filename"])
        if not filename:
            continue

        src = config.IMAGES_512_DIR / filename
        if not src.exists():
            continue

        dst = train_dir / filename
        if not dst.exists():
            shutil.copy2(src, dst)

        location = pick(row, ["location"], "a Penn campus building")
        time_val = pick(row, ["time_of_day"], "day")
        weather_val = pick(row, ["weather"], "cloudy")

        caption = f"a realistic photo of {location} on Penn campus, at {time_val}, with {weather_val} weather"

        records.append({
            "file_name": filename,
            "text": caption
        })

    meta_path = train_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("Dataset ready:", train_dir)


if __name__ == "__main__":
    main()
