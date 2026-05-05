import torch
from PIL import Image

from src import config


class InferenceResult:
    def __init__(self, image, depth=None):
        self.image = image
        self.depth = depth


def build_prompt(location, time_of_day, weather):
    return f"A realistic photo of {location} on a university campus, at {time_of_day}, with {weather} weather, natural lighting"


def estimate_depth(image):
    from transformers import DPTImageProcessor, DPTForDepthEstimation

    processor = DPTImageProcessor.from_pretrained(config.DEPTH_ESTIMATOR_ID)
    model = DPTForDepthEstimation.from_pretrained(config.DEPTH_ESTIMATOR_ID).to("cuda")

    inputs = processor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype("uint8")

    return Image.fromarray(depth)


class SDConditionControlNet:
    name = "sd_condition_controlnet"

    def __init__(self, device=None):
        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # load controlnet
        controlnet = ControlNetModel.from_pretrained(
            config.CONTROLNET_DEPTH_ID,
            torch_dtype=dtype,
        )

        # load pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            config.SD_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(self.device)

        # use a faster scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

    def generate(self, source, location, time_of_day, weather, seed=42):
        # resize input
        src = source.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))

        prompt = build_prompt(location, time_of_day, weather)
        negative_prompt = "distorted, unrealistic, blurry"

        # compute depth map
        depth = estimate_depth(src)

        # generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # run diffusion
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
            generator=generator,
        ).images[0]

        return InferenceResult(image=output, depth=depth)


def load_pipeline(system):
    if system == "sd_condition_controlnet":
        return SDConditionControlNet()

    raise ValueError(f"Unknown system: {system}")
