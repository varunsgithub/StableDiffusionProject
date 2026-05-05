"""Run every system on the val pairs and report all metrics.

Output:
  outputs/<system>/<scene_id>__<src_cond>__to__<tgt_cond>.png
  outputs/<system>/per_sample.csv
  outputs/results_summary.csv

Run:
    python -m src.eval.ablate --systems ip2p sd_cn sd_cn_lora --lora-path checkpoints/lora_campus/final
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src import config
from src.models.inference import load_pipeline
from src.eval.metrics import lpips_score, ssim_score, psnr_score, aggregate
from src.eval.condition_clf import classify, condition_accuracy


def load_val_pairs() -> list[dict]:
    with open(config.PAIRS_CSV, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "val"]


def run_system(system: str, pairs: list[dict], lora_path: str | None,
               limit: int | None = None, seed: int = 42) -> Path:
    """Generate images for every pair and write a per-sample CSV of metrics."""
    pipe = load_pipeline(system, lora_path=lora_path)
    out_dir = config.OUTPUTS_DIR / pipe.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    clf_inputs: list[dict] = []

    if limit:
        pairs = pairs[:limit]

    for p in tqdm(pairs, desc=pipe.name):
        src_path = config.IMAGES_512_DIR / p["src_filename"]
        tgt_path = config.IMAGES_512_DIR / p["tgt_filename"]
        if not src_path.exists() or not tgt_path.exists():
            continue
        src_img = Image.open(src_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")

        res = pipe.generate(
            src_img, location=p["location"],
            time_of_day=p["tgt_time"], weather=p["tgt_weather"], seed=seed,
        )
        pred = res.image
        name = f"{p['scene_id']}__{p['src_condition']}__to__{p['tgt_condition']}.png"
        pred.save(out_dir / name)

        lp = lpips_score(pred, tgt_img)
        ss = ssim_score(pred, tgt_img)
        ps = psnr_score(pred, tgt_img)
        clf = classify(pred)

        rows_out.append({
            "system": pipe.name,
            "scene_id": p["scene_id"],
            "src_condition": p["src_condition"],
            "tgt_condition": p["tgt_condition"],
            "tgt_time": p["tgt_time"],
            "tgt_weather": p["tgt_weather"],
            "pred_time": clf["time_pred"],
            "pred_weather": clf["weather_pred"],
            "lpips": lp, "ssim": ss, "psnr": ps,
            "output": str((out_dir / name).relative_to(config.PROJECT_ROOT)),
        })
        clf_inputs.append({
            "tgt_time": p["tgt_time"], "tgt_weather": p["tgt_weather"],
            "pred": clf,
        })

    # Per-sample CSV
    per_sample = out_dir / "per_sample.csv"
    with open(per_sample, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    # Summary JSON
    lpips_stats = aggregate(r["lpips"] for r in rows_out)
    ssim_stats = aggregate(r["ssim"] for r in rows_out)
    psnr_stats = aggregate(r["psnr"] for r in rows_out)
    cond_acc = condition_accuracy(clf_inputs)
    summary = {
        "system": pipe.name,
        "n": len(rows_out),
        "lpips": lpips_stats,
        "ssim": ssim_stats,
        "psnr": psnr_stats,
        "condition_accuracy": cond_acc,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[ablate] {pipe.name}:")
    print(json.dumps(summary, indent=2))

    # Free GPU between systems
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_sample


def write_results_table(system_summaries: list[dict]) -> None:
    rows = []
    for s in system_summaries:
        rows.append({
            "system": s["system"],
            "n": s["n"],
            "lpips_mean": s["lpips"]["mean"],
            "ssim_mean": s["ssim"]["mean"],
            "psnr_mean": s["psnr"]["mean"],
            "time_acc": s["condition_accuracy"]["time_acc"],
            "weather_acc": s["condition_accuracy"]["weather_acc"],
            "joint_acc": s["condition_accuracy"]["joint_acc"],
        })
    out = config.OUTPUTS_DIR / "results_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ablate] Wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+",
                   default=["ip2p", "sd_cn", "sd_cn_lora"],
                   choices=["ip2p", "sd_cn", "sd_cn_lora", "sd_lora", "sd_condition_controlnet", "sd_condition_lora"])
    p.add_argument("--lora-path", default=str(config.CHECKPOINTS_DIR / "lora_campus" / "final"))
    p.add_argument("--limit", type=int, default=None,
                   help="Only run first N val pairs (for smoke testing)")
    args = p.parse_args()

    pairs = load_val_pairs()
    print(f"[ablate] {len(pairs)} val pairs, running systems: {args.systems}")

    summaries = []
    for sys_name in args.systems:
        lora = args.lora_path if "lora" in sys_name else None
        run_system(sys_name, pairs, lora_path=lora, limit=args.limit)
        summ = json.loads((config.OUTPUTS_DIR /
                          {"ip2p": "ip2p", "sd_cn": "sd_controlnet",
                           "sd_cn_lora": "sd_controlnet_lora",
                           "sd_lora": "sd_lora_img2img", "sd_condition_controlnet": "sd_condition_controlnet", "sd_condition_lora": "sd_condition_lora_img2img"}[sys_name] /
                          "summary.json").read_text())
        summaries.append(summ)

    write_results_table(summaries)


if __name__ == "__main__":
    main()
