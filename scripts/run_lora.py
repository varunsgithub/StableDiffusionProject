from PIL import Image
from src import config
from src.models.inference import load_pipeline


def main():
    lora_path = config.CHECKPOINTS_DIR / "condition_lora"

    pipe = load_pipeline("sd_condition_lora", lora_path=str(lora_path))

    img = Image.open(list(config.IMAGES_512_DIR.glob("*.jpg"))[0])

    result = pipe.generate(
        source=img,
        location="a Penn campus building",
        time_of_day="night",
        weather="clear"
    )

    result.image.save("outputs/lora_result.png")


if __name__ == "__main__":
    main()
