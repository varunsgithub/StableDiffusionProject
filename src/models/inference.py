"""Inference pipelines for all four systems in the ablation.

Each returns a PIL.Image given (source_image, target_time, target_weather).
They share a single class hierarchy so the ablation runner can loop over them.

Systems:
  - IP2PBaseline           : InstructPix2Pix, off the shelf (what you had)
  - SDControlNetDepth      : SD 1.5 + ControlNet-depth, no LoRA
  - SDControlNetDepthLoRA  : SD 1.5 + ControlNet-depth + our campus LoRA  (the full system)
  - SDLoRAOnly             : SD 1.5 + LoRA only (img2img, no ControlNet) — isolates LoRA's contribution

Depth maps are generated on demand from the source image so the caller just
passes the source PIL image.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import config
from src.data.prompts import build_prompt, build_edit_instruction, NEGATIVE_PROMPT


# ---------- Depth estimator (module-level singleton) ----------
_depth_cache: dict = {}


def estimate_depth(img: Image.Image, device: str | None = None) -> Image.Image:
    """Return an 8-bit RGB depth map sized IMAGE_SIZE x IMAGE_SIZE."""
    if "model" not in _depth_cache:
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _depth_cache["processor"] = DPTImageProcessor.from_pretrained(config.DEPTH_ESTIMATOR_ID)
        _depth_cache["model"] = DPTForDepthEstimation.from_pretrained(config.DEPTH_ESTIMATOR_ID).to(dev).eval()
        _depth_cache["device"] = dev
    proc, model, dev = _depth_cache["processor"], _depth_cache["model"], _depth_cache["device"]
    with torch.no_grad():
        inputs = proc(images=img.convert("RGB"), return_tensors="pt").to(dev)
        d = model(**inputs).predicted_depth
        d = torch.nn.functional.interpolate(
            d.unsqueeze(1), size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            mode="bicubic", align_corners=False,
        ).squeeze().cpu().numpy()
    d = d - d.min()
    d = d / (d.max() + 1e-8)
    d8 = (d * 255.0).astype(np.uint8)
    return Image.fromarray(d8).convert("RGB")


# ---------- Base class ----------
@dataclass
class InferenceResult:
    image: Image.Image
    depth: Image.Image | None = None


class Pipeline:
    name: str = "base"

    def generate(self, source: Image.Image, location: str,
                 time_of_day: str, weather: str, seed: int = 42) -> InferenceResult:
        raise NotImplementedError


# ---------- 1. InstructPix2Pix baseline ----------
class IP2PBaseline(Pipeline):
    name = "ip2p"

    def __init__(self, device: str | None = None):
        from diffusers import StableDiffusionInstructPix2PixPipeline
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            config.IP2P_MODEL_ID, torch_dtype=dtype, safety_checker=None,
        ).to(self.device)

    def generate(self, source, location, time_of_day, weather, seed=42):
        src = source.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        instr = build_edit_instruction(time_of_day, weather)
        g = torch.Generator(self.device).manual_seed(seed)
        out = self.pipe(
            prompt=instr, image=src,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            image_guidance_scale=config.IMAGE_GUIDANCE_SCALE,
            generator=g,
        ).images[0]
        return InferenceResult(image=out)


# ---------- 2 & 3. SD + ControlNet(-depth) (+ optional LoRA) ----------
class SDControlNetDepth(Pipeline):
    name = "sd_controlnet"

    def __init__(self, lora_path: str | None = None, device: str | None = None):
        from diffusers import (
            StableDiffusionControlNetPipeline, ControlNetModel,
            UniPCMultistepScheduler,
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            config.CONTROLNET_DEPTH_ID, torch_dtype=dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            config.SD_MODEL_ID, controlnet=controlnet,
            torch_dtype=dtype, safety_checker=None,
        ).to(self.device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if lora_path:
            # Load our fine-tuned PEFT LoRA adapter into the UNet
            from peft import PeftModel
            self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_path)
            self.name = "sd_controlnet_lora"
            self.lora_path = lora_path
        else:
            self.lora_path = None

    def generate(self, source, location, time_of_day, weather, seed=42):
        src = source.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        depth = estimate_depth(src)
        prompt = build_prompt(location, time_of_day, weather)
        g = torch.Generator(self.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
            image=depth,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
            generator=g,
        ).images[0]
        return InferenceResult(image=out, depth=depth)



class SDConditionLoRAImg2Img(Pipeline):
    name = "sd_condition_lora_img2img"

    def __init__(self, lora_path: str, strength: float = 0.55, device: str | None = None):
        from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.SD_MODEL_ID, torch_dtype=dtype, safety_checker=None,
        ).to(self.device)

        self.pipe.load_lora_weights(lora_path)
        self.strength = strength

    def generate(self, source, location, time_of_day, weather, seed=42):
        src = source.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        prompt = build_prompt(location, time_of_day, weather)
        g = torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=src,
            strength=self.strength,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            generator=g,
        ).images[0]
        return InferenceResult(image=out)


# ---------- 4. LoRA-only (img2img, no ControlNet) — ablation to isolate LoRA ----------
class SDLoRAOnly(Pipeline):
    name = "sd_lora_img2img"

    def __init__(self, lora_path: str, strength: float = 0.6, device: str | None = None):
        from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            config.SD_MODEL_ID, torch_dtype=dtype, safety_checker=None,
        ).to(self.device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        from peft import PeftModel
        self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_path)
        self.strength = strength

    def generate(self, source, location, time_of_day, weather, seed=42):
        src = source.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        prompt = build_prompt(location, time_of_day, weather)
        g = torch.Generator(self.device).manual_seed(seed)
        out = self.pipe(
            prompt=prompt, negative_prompt=NEGATIVE_PROMPT, image=src,
            strength=self.strength,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            generator=g,
        ).images[0]
        return InferenceResult(image=out)


def load_pipeline(system: str, lora_path: str | None = None) -> Pipeline:
    """Factory. `system` ∈ {'ip2p', 'sd_cn', 'sd_cn_lora', 'sd_lora'}."""
    if system == "ip2p":
        return IP2PBaseline()
    if system == "sd_cn":
        return SDControlNetDepth(lora_path=None)
    if system == "sd_cn_lora":
        if not lora_path:
            raise ValueError("sd_cn_lora requires --lora-path")
        return SDControlNetDepth(lora_path=lora_path)
    if system == "sd_lora":
        if not lora_path:
            raise ValueError("sd_lora requires --lora-path")
        return SDLoRAOnly(lora_path=lora_path)
    if system == "sd_condition_controlnet":
        return SDConditionControlNet()

    if system == "sd_condition_lora":
        if not lora_path:
            raise ValueError("sd_condition_lora requires --lora-path")
        return SDConditionLoRAImg2Img(lora_path=lora_path)
    raise ValueError(f"unknown system: {system}")

class SDConditionControlNet(Pipeline):
    name = "sd_condition_controlnet"

    def __init__(self, device: str | None = None):
        from diffusers import (
            StableDiffusionControlNetPipeline, ControlNetModel,
            UniPCMultistepScheduler,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            config.CONTROLNET_DEPTH_ID, torch_dtype=dtype,
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            config.SD_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(self.device)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

    def generate(self, source, location, time_of_day, weather, seed=42):
        src = source.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))

        prompt = f"A realistic photo of {location} on a university campus, at {time_of_day}, with {weather} weather, natural lighting, preserve structure"

        negative_prompt = "distorted, unrealistic, blurry"

        depth = estimate_depth(src)

        g = torch.Generator(self.device).manual_seed(seed)

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            guidance_scale=config.GUIDANCE_SCALE,
            controlnet_conditioning_scale=config.CONTROLNET_CONDITIONING_SCALE,
            generator=g,
        ).images[0]

        return InferenceResult(image=out, depth=depth)
