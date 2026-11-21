"""Train and evaluate the Stable Diffusion 1.5 full-UNet baseline offline."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import sys
import tempfile
import time
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/sd15"
DATA = ROOT / "data/butterflies"
MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DATA_REVISION = "3cdedf844922ab40393d46d4c7f81c596e1c6d45"
UNET_WEIGHTS = Path("unet/diffusion_pytorch_model.safetensors")
PROMPT = "a photo of a winged insect"
SEED = 42
RESOLUTION = 512
LEARNING_RATE = 1e-5
WINDOW = 20
SEGMENT = 50
CSV_FIELDS = (
    "step", "attempt", "epoch", "image_id", "timestep", "train_loss",
    "learning_rate", "grad_norm_before_clip", "amp_scale",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def offline_cuda():
    for name, value in {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ.setdefault(name, value)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Check GPU visibility before running this command.")
    return torch


def read_training_rows(data):
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_indices = random.Random(SEED).sample(range(1000), 40)
    if (
        manifest["repo"] != "huggan/smithsonian_butterflies_subset"
        or manifest["revision"] != DATA_REVISION
        or [row["source_index"] for row in manifest["records"]] != expected_indices
        or [row["split"] for row in manifest["records"]] != ["train"] * 32 + ["val"] * 8
    ):
        raise ValueError("The data manifest does not match the specified dataset and split.")
    rows = manifest["records"][:32]
    with (data / "train/metadata.jsonl").open(encoding="utf-8") as stream:
        metadata = [json.loads(line) for line in stream if line.strip()]
    if metadata != [{"file_name": row["file_name"], "text": PROMPT} for row in rows]:
        raise ValueError("Training metadata does not match the manifest and fixed caption.")
    for row in rows:
        if row["file_name"] != f"{row['source_index']:04d}.png" or row["text"] != PROMPT:
            raise ValueError(f"Invalid training manifest row: {row}")
        if sha256(data / "train" / row["file_name"]) != row["sha256"]:
            raise ValueError(f"Training image checksum mismatch: {row['file_name']}")
    return rows


def train(args):
    torch = offline_cuda()
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor2_0
    from diffusers.optimization import get_scheduler
    from PIL import Image
    from torch.nn.functional import mse_loss
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from transformers import CLIPTextModel, CLIPTokenizer

    rows = read_training_rows(args.data)
    if not (args.model / UNET_WEIGHTS).is_file():
        raise FileNotFoundError(args.model / UNET_WEIGHTS)
    if args.run_dir.exists() or args.output_dir.exists():
        raise FileExistsError("Choose a fresh run directory and a fresh model output directory.")
    if args.run_dir == args.output_dir or args.run_dir.is_relative_to(args.output_dir):
        raise ValueError("Put the logs beside or outside the model output directory.")
    args.run_dir.mkdir(parents=True)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "environment-lock.txt"
    config = {
        "base_model": str(args.model), "model_revision": MODEL_REVISION,
        "base_unet_sha256": sha256(args.model / UNET_WEIGHTS),
        "data": str(args.data), "data_revision": DATA_REVISION,
        "data_manifest_sha256": sha256(args.data / "manifest.json"),
        "source_sha256": sha256(Path(__file__)),
        "environment_lock_sha256": sha256(lock),
        "seed": SEED, "resolution": RESOLUTION, "prompt": PROMPT, "steps": args.steps,
        "batch_size": 1, "gradient_accumulation_steps": 1, "num_workers": 0,
        "shuffle": True, "drop_last": False, "mixed_precision": "fp16",
        "unet_dtype": "float32", "frozen_dtype": "float16", "loss_dtype": "float32",
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "betas": [0.9, 0.999],
        "weight_decay": 0.01, "adam_eps": 1e-8, "max_grad_norm": 1.0,
        "lr_scheduler": "constant", "lr_warmup_steps": 0, "prediction_type": "epsilon",
        "gradient_checkpointing": True, "attention": "SDPA", "output_dir": str(args.output_dir),
    }
    write_json(args.run_dir / "run_config.json", config)
    (args.run_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )

    with (args.run_dir / "train_steps.csv").open("w", newline="", encoding="utf-8") as raw, (
        args.run_dir / "train_events.jsonl"
    ).open("w", encoding="utf-8") as events:
        writer = csv.DictWriter(raw, fieldnames=CSV_FIELDS)
        writer.writeheader()
        raw.flush()

        def event(name, **fields):
            events.write(json.dumps({"event": name, **fields}, allow_nan=False) + "\n")
            events.flush()

        step = 0
        attempt = 0
        try:
            accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=1)
            if accelerator.num_processes != 1 or accelerator.device.type != "cuda":
                raise RuntimeError("Run directly with Python on one CUDA device.")
            device = accelerator.device
            set_seed(SEED)
            tokenizer = CLIPTokenizer.from_pretrained(args.model / "tokenizer", local_files_only=True)
            input_ids = tokenizer(
                PROMPT, padding="max_length", truncation=True,
                max_length=tokenizer.model_max_length, return_tensors="pt",
            ).input_ids
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ])

            class TrainingImages(torch.utils.data.Dataset):
                def __len__(self):
                    return len(rows)

                def __getitem__(self, index):
                    row = rows[index]
                    with Image.open(args.data / "train" / row["file_name"]) as image:
                        if image.mode != "RGB" or image.size != (RESOLUTION, RESOLUTION):
                            raise ValueError(f"Invalid image mode or size: {row['file_name']}")
                        pixels = image_transform(image)
                    return pixels, row["source_index"]

            loader = DataLoader(TrainingImages(), batch_size=1, shuffle=True, num_workers=0)
            noise_scheduler = DDPMScheduler.from_pretrained(
                args.model / "scheduler", prediction_type="epsilon", local_files_only=True
            )
            vae = AutoencoderKL.from_pretrained(args.model / "vae", local_files_only=True)
            text_encoder = CLIPTextModel.from_pretrained(
                args.model / "text_encoder", local_files_only=True
            )
            unet = UNet2DConditionModel.from_pretrained(args.model / "unet", local_files_only=True)
            vae.requires_grad_(False).eval()
            text_encoder.requires_grad_(False).eval()
            unet.requires_grad_(True).train()
            unet.set_attn_processor(AttnProcessor2_0())
            unet.enable_gradient_checkpointing()
            sampled_weight = unet.conv_in.weight.detach().cpu().clone()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            vae.to(device, dtype=torch.float16)
            text_encoder.to(device, dtype=torch.float16)
            unet.to(device, dtype=torch.float32)
            parameters = list(unet.parameters())
            parameter_count = sum(parameter.numel() for parameter in parameters)
            optimizer = torch.optim.AdamW(
                parameters, lr=LEARNING_RATE, betas=(0.9, 0.999),
                weight_decay=0.01, eps=1e-8,
            )
            scheduler = get_scheduler(
                "constant", optimizer, num_warmup_steps=0, num_training_steps=args.steps
            )
            unet, optimizer, loader, scheduler = accelerator.prepare(unet, optimizer, loader, scheduler)
            if parameter_count != 859_520_964 or any(p.dtype != torch.float32 for p in unet.parameters()):
                raise RuntimeError("The full SD 1.5 UNet must be trainable in FP32.")
            input_ids = input_ids.to(device)
            optimizer.zero_grad()
            epoch = 0
            consecutive_skips = 0

            while step < args.steps:
                epoch += 1
                for pixels, image_id in loader:
                    attempt += 1
                    amp_scale = accelerator.scaler.get_scale()
                    lr = optimizer.param_groups[0]["lr"]
                    with torch.no_grad():
                        latents = vae.encode(pixels.to(torch.float16)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                        noise = torch.randn_like(latents)
                        timestep = torch.randint(
                            noise_scheduler.config.num_train_timesteps,
                            (latents.shape[0],), device=device,
                        )
                        noisy = noise_scheduler.add_noise(latents, noise, timestep)
                        context = text_encoder(input_ids, return_dict=False)[0]
                    prediction = unet(noisy, timestep, context, return_dict=False)[0]
                    loss = mse_loss(prediction.float(), noise.float(), reduction="mean")
                    value = loss.detach().item()
                    record = {
                        "attempt": attempt, "epoch": epoch, "image_id": image_id.item(),
                        "timestep": timestep.item(), "learning_rate": lr, "amp_scale": amp_scale,
                    }
                    if not math.isfinite(value):
                        event("nonfinite_loss", **record, train_loss=None)
                        raise FloatingPointError(f"Non-finite loss at attempt {attempt}")
                    accelerator.backward(loss)
                    norm = accelerator.clip_grad_norm_(parameters, 1.0).item()
                    optimizer.step()
                    skipped = accelerator.optimizer_step_was_skipped
                    optimizer.zero_grad()
                    if skipped:
                        consecutive_skips += 1
                        event("amp_skip", **record, train_loss=value,
                              grad_norm_before_clip=norm if math.isfinite(norm) else None)
                        if consecutive_skips >= 32:
                            raise FloatingPointError("AMP skipped 32 consecutive optimizer attempts.")
                        continue
                    scheduler.step()
                    consecutive_skips = 0
                    step += 1
                    writer.writerow(dict(record, step=step, train_loss=value,
                                         grad_norm_before_clip=norm))
                    raw.flush()
                    if step == 1 or step % 20 == 0 or step == args.steps:
                        print(f"step {step}/{args.steps}  loss={value:.6f}", flush=True)
                    if step == args.steps:
                        break

            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device)
            trained_unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=False).eval()
            if torch.equal(sampled_weight, trained_unet.conv_in.weight.detach().cpu()):
                raise RuntimeError("The UNet input convolution did not change.")
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.model, unet=trained_unet, vae=vae, text_encoder=text_encoder,
                tokenizer=tokenizer, local_files_only=True,
            )
            save_started = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix=".baseline-model-", dir=args.output_dir.parent) as tmp:
                staging = Path(tmp) / "pipeline"
                pipeline.save_pretrained(staging, safe_serialization=True)
                final_hash = sha256(staging / UNET_WEIGHTS)
                if args.output_dir.exists():
                    raise FileExistsError(args.output_dir)
                staging.rename(args.output_dir)
            write_json(args.run_dir / "train_summary.json", {
                "successful_updates": step, "attempts": attempt, "trainable_parameters": parameter_count,
                "training_seconds": seconds, "training_peak_allocated_bytes": peak,
                "save_and_hash_seconds": time.perf_counter() - save_started,
                "unet_sha256": final_hash, "output_dir": str(args.output_dir),
            })
            event("completed", step=step, attempt=attempt)
        except (Exception, KeyboardInterrupt) as exc:
            event("failed", step=step, attempt=attempt, error=str(exc))
            raise


def infer(args):
    torch = offline_cuda()
    import numpy as np
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import save

    metadata_path = args.output.with_suffix(".json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError("Choose a new image output name.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.latents.parent.mkdir(parents=True, exist_ok=True)
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True
    ).to("cuda")
    if pipeline.safety_checker is None:
        raise RuntimeError("The full pipeline must include its Safety Checker.")
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    shape = (1, pipeline.unet.config.in_channels, RESOLUTION // pipeline.vae_scale_factor,
             RESOLUTION // pipeline.vae_scale_factor)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    if not args.latents.exists():
        latent = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16).cpu()
        with args.latents.open("xb") as stream:
            stream.write(save({"initial_noise": latent}, metadata={"seed": str(SEED)}))
    with safe_open(args.latents, framework="pt", device="cpu") as archive:
        if (archive.metadata() or {}).get("seed") != str(SEED):
            raise ValueError("The initial noise file must be recorded with seed 42.")
        latent = archive.get_tensor("initial_noise")
    if tuple(latent.shape) != shape or latent.dtype != torch.float16 or not torch.isfinite(latent).all():
        raise ValueError("Invalid initial noise shape, dtype or values.")

    with torch.inference_mode():
        result = pipeline(
            PROMPT, negative_prompt="", height=RESOLUTION, width=RESOLUTION,
            num_inference_steps=20, guidance_scale=7.5, eta=0.0, latents=latent.to("cuda"),
            output_type="np",
        )
    pixels = result.images[0]
    finite = bool(np.isfinite(pixels).all())
    safe = result.nsfw_content_detected == [False]
    nonconstant = finite and bool(np.ptp(pixels) > 0)
    if finite:
        image = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image).save(args.output)
    write_json(metadata_path, {
        "model": str(args.model), "unet_sha256": sha256(args.model / UNET_WEIGHTS),
        "initial_noise": str(args.latents), "initial_noise_sha256": sha256(args.latents),
        "seed": SEED, "prompt": PROMPT, "negative_prompt": "", "scheduler": "DDIM",
        "timesteps": pipeline.scheduler.timesteps.tolist(), "num_inference_steps": 20,
        "width": RESOLUTION, "height": RESOLUTION, "guidance_scale": 7.5, "eta": 0.0,
        "finite": finite, "nonconstant": nonconstant, "safety_checker_passed": safe,
    })
    if not (safe and nonconstant):
        raise RuntimeError(f"Image did not pass the automatic checks. See {metadata_path}.")
    print(f"Image saved to {args.output}. Inspect the insect subject visually.")


def plot(args):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sd-lab-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.csv == (args.output_dir / "loss_curve.csv").resolve():
        raise ValueError("Use a separate path for the derived curve CSV.")
    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["train_loss"]) for row in rows]
    if not steps or steps != list(range(1, len(rows) + 1)) or not all(map(math.isfinite, losses)):
        raise ValueError("Expected consecutive successful steps starting at 1 with finite losses.")
    smoothed = [fmean(losses[end - WINDOW:end]) for end in range(WINDOW, len(losses) + 1)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "loss_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "train_loss", "moving_mean_20"))
        for step, loss in zip(steps, losses):
            writer.writerow((step, loss, smoothed[step - WINDOW] if step >= WINDOW else ""))
    fig, (curve, comparison) = plt.subplots(1, 2, figsize=(10.5, 4), width_ratios=(3, 1.15),
                                            layout="constrained")
    curve.plot(steps, losses, color="#7994b2", alpha=0.65, linewidth=0.8, label="Raw loss")
    if smoothed:
        curve.plot(steps[WINDOW - 1:], smoothed, color="#174a81", linewidth=2, label="20-step mean")
    curve.set(xlabel="Optimizer update", ylabel="Noise prediction MSE", title="Training loss")
    curve.legend(frameon=False)
    summary = {"updates": len(steps), "moving_window": WINDOW, "segment_size": SEGMENT,
               "first_50_mean": None, "last_50_mean": None, "last_minus_first": None}
    if len(steps) >= 2 * SEGMENT:
        early = fmean(losses[:SEGMENT])
        late = fmean(losses[-SEGMENT:])
        last_start = len(steps) - SEGMENT + 1
        summary.update(first_50_mean=early, last_50_mean=late, last_minus_first=late - early,
                       first_steps=[1, SEGMENT], last_steps=[last_start, len(steps)])
        curve.axvspan(1, SEGMENT, color="#4d79a4", alpha=0.12)
        curve.axvspan(last_start, len(steps), color="#dc8346", alpha=0.15)
        bars = comparison.bar(["First 50", "Last 50"], [early, late], color=["#4d79a4", "#dc8346"])
        comparison.bar_label(bars, fmt="%.4f", padding=4)
        comparison.margins(y=0.22)
        comparison.set(ylabel="Mean MSE", title=f"Last − first: {late - early:+.4f}")
    else:
        comparison.set_axis_off()
        comparison.text(0.5, 0.5, "Segment means require\n100 logged updates.",
                        va="center", ha="center", transform=comparison.transAxes)
    write_json(args.output_dir / "loss_summary.json", summary)
    fig.savefig(args.output_dir / "training_loss.png", dpi=300)
    fig.savefig(args.output_dir / "training_loss.pdf")
    plt.close(fig)


def positive_steps(value):
    steps = int(value)
    if steps < 1:
        raise argparse.ArgumentTypeError("steps must be positive")
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Fine-tune the entire UNet")
    training.add_argument("--model", type=Path, default=MODEL)
    training.add_argument("--data", type=Path, default=DATA)
    training.add_argument("--steps", type=positive_steps, default=300)
    training.add_argument("--run-dir", type=Path, required=True, help="Directory for logs and metadata")
    training.add_argument("--output-dir", type=Path, required=True, help="Fresh directory for the pipeline")
    training.set_defaults(handler=train)
    inference = commands.add_parser("infer", help="Generate the fixed DDIM comparison image")
    inference.add_argument("--model", type=Path, default=MODEL)
    inference.add_argument("--latents", type=Path, required=True, help="Shared initial noise file")
    inference.add_argument("--output", type=Path, required=True, help="New PNG path")
    inference.set_defaults(handler=infer)
    plotting = commands.add_parser("plot", help="Plot loss and calculate report statistics from a CSV")
    plotting.add_argument("--csv", type=Path, required=True)
    plotting.add_argument("--output-dir", type=Path, required=True)
    plotting.set_defaults(handler=plot)
    args = parser.parse_args()
    if args.command == "infer" and args.output.suffix.lower() != ".png":
        parser.error("--output must be a PNG path")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    args.handler(args)


if __name__ == "__main__":
    main()
