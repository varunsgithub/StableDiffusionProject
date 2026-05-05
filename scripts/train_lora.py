import subprocess
from pathlib import Path

from src import config


def main():
    tools_dir = Path("tools")
    tools_dir.mkdir(exist_ok=True)

    train_script = tools_dir / "train_text_to_image_lora.py"

    if not train_script.exists():
        subprocess.run(
            [
                "wget",
                "-q",
                "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py",
                "-O",
                str(train_script),
            ],
            check=True,
        )

    train_dir = config.PROCESSED_DIR / "condition_lora_dataset"
    output_dir = config.CHECKPOINTS_DIR / "condition_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        str(train_script),
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        f"--train_data_dir={train_dir}",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-4",
        "--max_train_steps=1200",
        "--checkpointing_steps=300",
        "--mixed_precision=fp16",
        "--rank=16",
        "--validation_prompt=a realistic photo of collegehall on Penn campus, at night, with clear weather, campus architecture, photorealistic",
        "--validation_epochs=999999",
        f"--output_dir={output_dir}",
    ]

    subprocess.run(command, check=True)

    print(f"LoRA checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()
