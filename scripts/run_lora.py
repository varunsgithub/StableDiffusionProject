from pathlib import Path

from PIL import Image

from src import config
from src.models.inference import load_pipeline


def main():
    lora_path = config.CHECKPOINTS_DIR / "condition_lora"

    pipe = load_pipeline(
        "sd_condition_lora",
        lora_path=str(lora_path),
    )

    image_files = sorted(config.IMAGES_512_DIR.glob("*.jpg")) + sorted(config.IMAGES_512_DIR.glob("*.JPG"))

    if not image_files:
        raise RuntimeError(f"No images found in {config.IMAGES_512_DIR}")

    source = Image.open(image_files[0]).convert("RGB")

    result = pipe.generate(
        source=source,
        location="a Penn campus building",
        time_of_day="night",
        weather="clear",
        seed=42,
    )

    out_dir = config.OUTPUTS_DIR / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "lora_demo.png"
    result.image.save(output_path)

    print(f"Saved LoRA demo output to: {output_path}")


if __name__ == "__main__":
    main()
