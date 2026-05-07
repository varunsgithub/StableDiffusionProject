# StableDiffusionProject — Scene Lighting & Weather Transfer

CIS 5190 Final Project. Fine-tunes Stable Diffusion v1.5 with LoRA + dual ControlNet (Canny edges + semantic segmentation) to transfer **time-of-day** and **weather** conditions onto an input scene while preserving its structure.

Given a **source image** (e.g. a building at dawn) and a **prompt** (e.g. *"night-time, heavy rain, wet pavement"*), the model produces a generated image that keeps the geometry and layout of the source but matches the lighting and weather described in the prompt.

---

## Repository Structure

```
StableDiffusionProject/
├── inference_pipeline_revised.ipynb   ← run this for inference (Colab-ready)
├── Image pre processing/
│   ├── cannyProcess.ipynb             ← adaptive Canny edge generation
│   └── Depth_and_Seg.ipynb            ← OneFormer segmentation + MiDaS depth
├── Model Training Books/
│   ├── Training Books/
│   │   └── Model_Training_rev.ipynb   ← LoRA fine-tuning pipeline
│   └── Standalone train/
│       └── Seg_maps_+_Canny_ProjectC.ipynb
└── Model Evaluation Books/
    ├── eval_canny.ipynb               ← Canny-only baseline
    ├── eval_canny_seg.ipynb           ← Canny + Seg (final model)
    ├── eval_canny_depth.ipynb         ← Canny + Depth ablation
    └── eval_canny_depth_seg.ipynb     ← all three conditions
```

The **final reported model** uses Canny + Segmentation (`eval_canny_seg.ipynb`). The inference notebook reproduces this exact configuration.

---

## Quick Start (5 minutes)

### 1. Open the inference notebook in Google Colab

Upload `inference_pipeline_revised.ipynb` to Google Colab, or open it directly via `File → Upload notebook`.

### 2. Set the runtime to GPU

`Runtime → Change runtime type → Hardware accelerator: GPU`

Recommended: **T4** (free tier) or better. Tested on **L4** (~24 GB VRAM). Inference uses ~10 GB VRAM at peak.

### 3. Run the cells top-to-bottom

The notebook is structured in 8 numbered sections. Just click `Runtime → Run all`, then interact with the prompts as they appear.

| Section | What it does |
|---------|--------------|
| 1 | Installs all Python dependencies and verifies GPU |
| 2 | Sets inference hyperparameters (matches training eval) |
| 3 | Downloads LoRA weights from Google Drive via `gdown` |
| 4 | Loads SD v1.5 base + dual ControlNet + applies the LoRA adapter |
| 5 | **You upload source + target images and type a prompt** |
| 6 | Preprocesses images (resize, Canny edge, semantic segmentation) |
| 7 | Runs inference and displays a 5-panel result grid |
| 8 | Zips and downloads the 3 output files |

---

## Requirements

### Hardware

- **GPU with ≥ 12 GB VRAM** (T4, L4, A100, V100, RTX 3060+ all work)
- ~10 GB free disk for cached model weights
- CPU-only inference is *not* supported (would take hours per image)

### Software

All Python packages are installed automatically by Section 1 of the notebook:

| Package | Pinned version |
|---------|----------------|
| `diffusers` | 0.31.0 |
| `transformers` | 4.44.0 |
| `peft` | 0.12.0 |
| `accelerate` | 0.34.0 |
| `gdown` | latest |
| `opencv-python-headless` | latest |
| `Pillow`, `matplotlib` | latest |

`torch` comes pre-installed in Colab GPU runtimes.

### Hugging Face Token

The base model `runwayml/stable-diffusion-v1-5` and the two ControlNet checkpoints (`lllyasviel/sd-controlnet-canny`, `lllyasviel/sd-controlnet-seg`) are downloaded from Hugging Face Hub.

These repos are **publicly accessible** — no token is required for the default Colab download. If your environment hits a rate limit or auth error:

1. Create a free account at https://huggingface.co
2. Go to **Settings → Access Tokens → New token** (read-access is enough)
3. Add it to Colab as an environment variable, before Section 4:

   ```python
   import os
   os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   ```

   or use the Colab secrets sidebar (key icon) and add `HF_TOKEN`.

If `runwayml/stable-diffusion-v1-5` is unavailable in your region (Runway removed their HF org in 2024 — community mirrors still exist), substitute the equivalent mirror in Section 4:

```python
"benjamin-paine/stable-diffusion-v1-5"   # community mirror, identical weights
```

### LoRA Weights

The fine-tuned LoRA adapter (~3 MB) lives in this Google Drive folder:

> https://drive.google.com/drive/folders/1i3fQelMbLIr7q6HDM0JJUKU-Rkglp3KY

Section 3 of the notebook downloads it automatically using `gdown`. The folder must remain shared as **"Anyone with the link – Viewer"**. No Google login is needed inside Colab.

If the download fails, you can manually upload the contents of that folder to `/content/lora_weights/` in Colab — the notebook auto-detects `adapter_config.json`.

---

## Input / Output

### What you upload (Section 5)

1. **Source image** (JPG/PNG) — the scene to transform
2. **Target image** (JPG/PNG) — the ground-truth reference at the desired time/weather (used by the TA to compute LPIPS and condition accuracy externally)
3. **Prompt** (text) — describe the target lighting and weather

   Example prompts:
   - `night time, heavy rain, wet reflective pavement, dramatic lighting`
   - `golden hour sunset, warm orange sky, long shadows, clear weather`
   - `overcast daytime, soft diffuse light, cloudy grey sky`

### What you download (Section 8)

A zip file `inference_results.zip` containing **only 3 images**, all 512x512:

| File | Description |
|------|-------------|
| `<source_filename>-src.png` | Preprocessed source (centre-cropped to 512x512) |
| `<target_filename>-target.png` | Preprocessed target / ground truth (centre-cropped to 512x512) |
| `<source_filename>-generated.png` | Model output |

The original filenames (without extension) are preserved in the output names, so multiple runs don't overwrite each other.

---

## Inference Hyperparameters

Set in Section 2 — these match the values used in `eval_canny_seg.ipynb` for the reported numbers. Do **not** change unless ablating.

```python
IMG_SIZE       = (512, 512)   # must match training
STRENGTH       = 0.80         # img2img denoising strength
NUM_STEPS      = 30           # UniPC denoising steps
GUIDANCE_SCALE = 9.0          # classifier-free guidance
CANNY_SCALE    = 0.65         # Canny ControlNet conditioning weight
SEG_SCALE      = 0.70         # Segmentation ControlNet conditioning weight
```

---

## Reproducing Training (optional)

The training pipeline lives in `Model Training Books/Training Books/Model_Training_rev.ipynb`. It expects the dataset structure described in `BoundaryMetadata.xlsx` (not committed — provided separately to graders) and a Colab Pro / A100 runtime due to the long training time. The README focuses on inference; training reproduction is documented inside that notebook.

---

## Models Used

| Component | Hugging Face ID | Size |
|-----------|-----------------|------|
| Base diffusion | `runwayml/stable-diffusion-v1-5` | ~4 GB |
| ControlNet (Canny) | `lllyasviel/sd-controlnet-canny` | ~1.45 GB |
| ControlNet (Seg) | `lllyasviel/sd-controlnet-seg` | ~1.45 GB |
| Segmentation pre-processor | `shi-labs/oneformer_ade20k_swin_tiny` | ~200 MB |
| LoRA adapter (ours) | Google Drive (link above) | ~3 MB |

Total first-run download: **~7 GB** (cached afterwards in `~/.cache/huggingface/`).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `WARNING: No GPU detected` | `Runtime → Change runtime type → GPU` |
| OOM during pipeline load | Restart runtime, verify GPU has ≥ 12 GB VRAM |
| `gdown` download fails | Re-share the Drive folder as "Anyone with the link – Viewer", or upload the LoRA folder manually to `/content/lora_weights/` |
| HF 401 / rate-limit error | Add an `HF_TOKEN` env var (see above) |
| `runwayml/stable-diffusion-v1-5` 404 | Replace with the mirror `benjamin-paine/stable-diffusion-v1-5` in Section 4 |
| Prompt input is unresponsive | The `input()` text box appears at the bottom of the cell — scroll down and press Enter after typing |

---

## Citation

If you use this work, please cite:

> Singh V., et al. *LoRA + Dual ControlNet for Scene Lighting and Weather Transfer*. CIS 5190, University of Pennsylvania, 2026.
