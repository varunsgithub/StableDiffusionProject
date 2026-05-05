import os
from src import config


def main():
    train_dir = config.PROCESSED_DIR / "condition_lora_dataset"
    output_dir = config.CHECKPOINTS_DIR / "condition_lora"

    os.makedirs(output_dir, exist_ok=True)

    cmd = f"""
    python tools/train_text_to_image_lora.py \
      --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
      --train_data_dir={train_dir} \
      --resolution=512 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=4 \
      --learning_rate=1e-4 \
      --max_train_steps=1200 \
      --checkpointing_steps=300 \
      --mixed_precision=fp16 \
      --rank=16 \
      --output_dir={output_dir}
    """

    os.system(cmd)


if __name__ == "__main__":
    main()
