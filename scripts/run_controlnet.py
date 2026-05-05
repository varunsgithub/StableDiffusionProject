import os
from PIL import Image

from src import config
from src.models.inference import load_pipeline


def main():
    # load pipeline
    pipe = load_pipeline("sd_condition_controlnet")

    # pick one sample image
    image_files = list(config.IMAGES_512_DIR.glob("*.jpg"))
    if not image_files:
        raise RuntimeError("No images found in dataset")

    img = Image.open(image_files[0])

    # example condition
    location = "a Penn campus building"
    time_of_day = "night"
    weather = "clear"

    result = pipe.generate(
        source=img,
        location=location,
        time_of_day=time_of_day,
        weather=weather,
        seed=42,
    )

    os.makedirs("outputs", exist_ok=True)
    result.image.save("outputs/controlnet_result.png")

    print("Saved result to outputs/controlnet_result.png")


if __name__ == "__main__":
    main()
