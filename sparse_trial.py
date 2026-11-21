"""Run the Appendix E sparse attention experiment as separate offline stages."""

import argparse
import csv
import json
import math
import random
import shlex
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from statistics import median

from baseline import PROMPT, ROOT, SEED, UNET_WEIGHTS, offline_cuda, sha256, write_json
from sparse_attention import (
    CHANNELS, ERROR_LIMIT, HEAD_DIM, HEADS, LATENT_SCALE, RESOLUTIONS, TARGET, WINDOW,
    WindowAttentionProcessor, check_window_layout, reference_attention, validate_layer,
)


BASELINE_RUN = ROOT / "runs/baseline-001"
DEFAULT_RUN = ROOT / "runs/sparse-001"
METHODS = ("full", "sparse")
WARMUP = 5
REPEATS = 30
ORDER_SEED = 42
FIRST_TIMESTEP = 951
TIMESTEPS = list(range(FIRST_TIMESTEP, 0, -50))
SAMPLING = {
    "prompt": PROMPT, "negative_prompt": "", "num_inference_steps": 20,
    "guidance_scale": 7.5, "eta": 0.0,
}
CAPTURE_SPEC = {
    "target": TARGET, "resolutions": list(RESOLUTIONS), "window": WINDOW,
    "heads": HEADS, "head_dim": HEAD_DIM, "dtype": "float16", "seed": SEED,
    "scheduler": "DDIM", "timestep": FIRST_TIMESTEP, "sampling": SAMPLING,
}
RAW_FIELDS = (
    "resolution", "tokens", "method", "pair_id", "run_order", "elapsed_ms",
    "entry_alloc_bytes", "peak_alloc_bytes", "extra_alloc_bytes", "input_sha256",
)
QUALIFY_FIELDS = (
    "resolution", "tokens", "shape_ok", "finite", "max_abs_error",
    "relative_rms_error", "passed", "input_sha256",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_hashes():
    return {
        filename: sha256(ROOT / filename)
        for filename in ("sparse_attention.py", "sparse_trial.py", "baseline.py", "environment-lock.txt")
    }


def check_sha(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


@contextmanager
def record_stage(directory, name):
    """Reserve a stage in a run and retain failures alongside completed artifacts."""
    path = directory / f"{name}_status.json"
    status = {
        "status": "running",
        "command": shlex.join([sys.executable, *sys.argv]),
        "source_sha256": code_hashes(),
    }
    if name == "plot":
        status["source_sha256"]["sparse_plots.py"] = sha256(ROOT / "sparse_plots.py")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2)
    try:
        yield
    except (Exception, KeyboardInterrupt) as error:
        write_json(path, dict(status, status="failed", error=str(error)))
        raise
    else:
        write_json(path, dict(status, status="completed"))


def single_cuda():
    torch = offline_cuda()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES.")
    return torch, torch.device("cuda:0")


def target_attention(unet):
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0

    target = unet.get_submodule(TARGET)
    if not isinstance(target, Attention) or type(target.get_processor()) is not AttnProcessor2_0:
        raise ValueError("Expected the original Diffusers Attention with AttnProcessor2_0.")
    validate_layer(target)
    return target


def pipeline_on_cuda(model, torch, device):
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.eval().requires_grad_(False)
    if pipe.vae_scale_factor != LATENT_SCALE or pipe.unet.config.in_channels != 4:
        raise ValueError("The model does not have the SD 1.5 latent geometry.")
    if pipe.safety_checker is None:
        raise ValueError("The full pipeline needs its Safety Checker.")
    return pipe


def noise_path(directory, size):
    return directory / "initial_latents" / f"{size}.safetensors"


def hidden_path(directory, size):
    return directory / "captures" / f"{size}.safetensors"


def load_noise(path, size, torch):
    from safetensors import safe_open

    side = size // LATENT_SCALE
    with safe_open(path, framework="pt", device="cpu") as archive:
        if (archive.metadata() or {}).get("seed") != str(SEED):
            raise ValueError(f"Unexpected initial noise seed: {path}")
        noise = archive.get_tensor("initial_noise")
    if noise.shape != (1, 4, side, side) or noise.dtype != torch.float16:
        raise ValueError(f"Unexpected initial noise shape or dtype: {path}")
    if not torch.isfinite(noise).all().item():
        raise ValueError(f"Non-finite initial noise: {path}")
    return noise


class CaptureComplete(Exception):
    """Stop the pipeline after capturing the first target input."""


def capture_input(pipe, attn, noise, size, torch, device):
    captured = None
    timestep = None
    original_stride = None

    def record_timestep(_module, args, kwargs):
        nonlocal timestep
        t = args[1] if len(args) > 1 else kwargs["timestep"]
        timestep = int(t.item()) if torch.is_tensor(t) else int(t)
        if timestep != FIRST_TIMESTEP:
            raise ValueError(f"Expected DDIM timestep {FIRST_TIMESTEP}, got {timestep}.")

    def record_hidden(_module, args, kwargs):
        nonlocal captured, original_stride
        if timestep != FIRST_TIMESTEP:
            raise RuntimeError("The target was reached without the expected UNet timestep.")
        hidden = args[0] if args else kwargs["hidden_states"]
        encoder = args[1] if len(args) > 1 else kwargs.get("encoder_hidden_states")
        mask = args[2] if len(args) > 2 else kwargs.get("attention_mask")
        if encoder is not None or mask is not None:
            raise ValueError("Expected unmasked self-attention at the capture hook.")
        expected = (2, (size // LATENT_SCALE)**2, CHANNELS)
        if hidden.shape != expected or hidden.dtype != torch.float16:
            raise ValueError(f"Expected {expected} in FP16, got {tuple(hidden.shape)}, {hidden.dtype}.")
        original_stride = list(hidden.stride())
        captured = hidden.detach().cpu().contiguous()
        if not torch.isfinite(captured).all().item():
            raise ValueError("The captured activations contain non-finite values.")
        raise CaptureComplete

    with (
        torch.no_grad(),
        pipe.unet.register_forward_pre_hook(record_timestep, with_kwargs=True),
        attn.register_forward_pre_hook(record_hidden, with_kwargs=True),
    ):
        try:
            pipe(
                **SAMPLING, height=size, width=size, latents=noise.to(device),
                output_type="latent",
            )
        except CaptureComplete:
            pass
        else:
            raise RuntimeError("The target attention layer was never captured.")
    if captured is None or pipe.scheduler.timesteps.tolist() != TIMESTEPS:
        raise RuntimeError("The capture did not use the expected DDIM timestep sequence.")
    return captured, original_stride


def capture(args):
    torch, device = single_cuda()
    from safetensors.torch import save_file

    base = args.baseline_run
    summary = read_json(base / "train/train_summary.json")
    comparison = read_json(base / "images/finetuned_full.json")
    model = args.model or Path(summary["output_dir"]).resolve()
    unet_sha = summary["unet_sha256"]
    if summary["successful_updates"] <= 0 or comparison["unet_sha256"] != unet_sha:
        raise ValueError("The baseline training and generation records disagree.")
    check_sha(model / UNET_WEIGHTS, unet_sha)
    starting_noise = base / "images/initial_latents.safetensors"
    check_sha(starting_noise, comparison["initial_noise_sha256"])
    run = args.run_dir
    (run / "initial_latents").mkdir()
    (run / "captures").mkdir()
    shutil.copyfile(starting_noise, noise_path(run, 512))
    config_path = run / "unet_config.json"
    shutil.copyfile(model / "unet/config.json", config_path)
    manifest = {
        "spec": CAPTURE_SPEC, "model": str(model), "unet_sha256": unet_sha,
        "unet_config_sha256": sha256(config_path), "baseline_run": str(base),
        "scheduler_config_sha256": sha256(model / "scheduler/scheduler_config.json"),
        "source_sha256": code_hashes(), "inputs": {},
    }
    write_json(run / "config.json", manifest)
    pipe = pipeline_on_cuda(model, torch, device)
    attn = target_attention(pipe.unet)
    weights = {name: value.detach().cpu().contiguous() for name, value in attn.state_dict().items()}
    if any(weight.dtype != torch.float16 for weight in weights.values()):
        raise ValueError("Expected the actual FP16 attention weights from the pipeline.")
    weights_path = run / "attention_weights.safetensors"
    save_file(weights, weights_path)
    manifest["weights_sha256"] = sha256(weights_path)

    for size in RESOLUTIONS:
        latents_path = noise_path(run, size)
        if size != 512:
            side = size // LATENT_SCALE
            generator = torch.Generator(device=device).manual_seed(SEED)
            noise = torch.randn(
                (1, 4, side, side), generator=generator, device=device, dtype=torch.float16,
            ).cpu()
            save_file({"initial_noise": noise}, latents_path, metadata={"seed": str(SEED)})
        noise = load_noise(latents_path, size, torch)
        hidden, stride = capture_input(pipe, attn, noise, size, torch, device)
        input_path = hidden_path(run, size)
        save_file({"hidden_states": hidden}, input_path)
        manifest["inputs"][str(size)] = {
            "shape": list(hidden.shape), "dtype": "float16", "timestep": FIRST_TIMESTEP,
            "original_stride": stride, "replay_stride": list(hidden.stride()),
            "hidden_sha256": sha256(input_path), "noise_sha256": sha256(latents_path),
        }
        print(f"Captured {size}×{size}: {tuple(hidden.shape)} at t={FIRST_TIMESTEP}", flush=True)
    write_json(run / "capture_manifest.json", manifest)


def checked_capture(directory):
    path = directory / "capture_manifest.json"
    manifest = read_json(path)
    if manifest["spec"] != CAPTURE_SPEC or set(manifest["inputs"]) != set(map(str, RESOLUTIONS)):
        raise ValueError("The capture does not match the Appendix E settings.")
    check_sha(directory / "attention_weights.safetensors", manifest["weights_sha256"])
    check_sha(directory / "unet_config.json", manifest["unet_config_sha256"])
    for size in RESOLUTIONS:
        entry = manifest["inputs"][str(size)]
        check_sha(hidden_path(directory, size), entry["hidden_sha256"])
        check_sha(noise_path(directory, size), entry["noise_sha256"])
    return manifest


def replay_layer(directory, torch, device):
    from accelerate import init_empty_weights
    from diffusers import UNet2DConditionModel
    from safetensors.torch import load_file

    with init_empty_weights():
        skeleton = UNet2DConditionModel.from_config(read_json(directory / "unet_config.json"))
    attn = target_attention(skeleton)
    del skeleton
    weights = load_file(directory / "attention_weights.safetensors")
    if any(weight.dtype != torch.float16 for weight in weights.values()):
        raise ValueError("Replay weights must be FP16.")
    attn.load_state_dict(weights, strict=True, assign=True)
    return attn.eval().requires_grad_(False).to(device)


def replay_input(directory, size, torch, device):
    from safetensors.torch import load_file

    hidden = load_file(hidden_path(directory, size))["hidden_states"]
    if hidden.shape != (2, (size // LATENT_SCALE)**2, CHANNELS) or hidden.dtype != torch.float16:
        raise ValueError(f"Invalid saved activation at {size}×{size}.")
    return hidden.to(device)


def qualify(args):
    import copy

    directory = args.capture_dir or args.run_dir
    manifest = checked_capture(directory)
    torch, device = single_cuda()
    check_window_layout()
    attn = replay_layer(directory, torch, device)
    reference = copy.deepcopy(attn).float()
    # The reference uses FP32 matmul without TF32 input rounding.
    torch.set_float32_matmul_precision("highest")
    run = args.run_dir
    verdicts = []
    csv_path = run / "qualify.csv"
    with torch.no_grad(), csv_path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=QUALIFY_FIELDS)
        writer.writeheader()
        for size in RESOLUTIONS:
            side = size // LATENT_SCALE
            hidden = replay_input(directory, size, torch, device)
            attn.set_processor(WindowAttentionProcessor(side, side))
            actual = attn(hidden)
            expected = reference_attention(reference, hidden.float(), side, side)
            shape_ok = actual.shape == expected.shape == hidden.shape
            finite = (torch.isfinite(actual).all() & torch.isfinite(expected).all()).item()
            absolute = relative = None
            if shape_ok and finite:
                difference = actual.float() - expected
                absolute = difference.abs().amax().item()
                denominator = expected.square().mean().sqrt().item()
                numerator = difference.square().mean().sqrt().item()
                relative = numerator / denominator if denominator else (0.0 if numerator == 0 else None)
            passed = absolute is not None and relative is not None and max(absolute, relative) <= ERROR_LIMIT
            writer.writerow({
                "resolution": size, "tokens": side**2, "shape_ok": shape_ok, "finite": finite,
                "max_abs_error": absolute, "relative_rms_error": relative, "passed": passed,
                "input_sha256": manifest["inputs"][str(size)]["hidden_sha256"],
            })
            file.flush()
            verdicts.append(passed)
            print(f"{size}×{size}: max_abs={absolute}, relative_rms={relative}, pass={passed}", flush=True)
            del hidden, actual, expected
    write_json(run / "qualification.json", {
        "passed": all(verdicts), "threshold": ERROR_LIMIT,
        "capture_dir": str(directory), "capture_manifest_sha256": sha256(directory / "capture_manifest.json"),
        "qualify_csv_sha256": sha256(csv_path), "source_sha256": code_hashes(),
    })
    if not all(verdicts):
        raise RuntimeError("Qualification failed. Inspect qualify.csv before benchmarking.")


def qualified_capture(run):
    result = read_json(run / "qualification.json")
    if not result["passed"] or result["threshold"] != ERROR_LIMIT:
        raise ValueError("All three input sizes must pass qualification.")
    if result["source_sha256"] != code_hashes():
        raise ValueError("Source or lock file changed. Qualify in a new run using --capture-dir.")
    directory = Path(result["capture_dir"])
    check_sha(run / "qualify.csv", result["qualify_csv_sha256"])
    check_sha(directory / "capture_manifest.json", result["capture_manifest_sha256"])
    return directory, checked_capture(directory)


def measurement_order():
    rng = random.Random(ORDER_SEED)
    return [rng.sample(METHODS, len(METHODS)) for _ in range(REPEATS)]


def summarize_benchmark(rows):
    import numpy as np

    results = []
    for size in RESOLUTIONS:
        tokens = (size // LATENT_SCALE)**2
        for method in METHODS:
            draws = [row for row in rows if row["resolution"] == size and row["method"] == method]
            if len(draws) != REPEATS:
                raise ValueError(f"Expected {REPEATS} readings for {size} / {method}.")
            times = [row["elapsed_ms"] for row in draws]
            memory = median(row["extra_alloc_bytes"] for row in draws)
            p25, p75 = np.percentile(times, [25, 75])
            results.append({
                "resolution": size, "tokens": tokens, "method": method, "readings": len(draws),
                "median_ms": median(times), "p25_ms": p25, "p75_ms": p75,
                "median_extra_alloc_bytes": memory, "median_extra_alloc_mib": memory / 2**20,
                "sparsity_pct": 100 * (1 - WINDOW**2 / tokens) if method == "sparse" else 0.0,
                "full_over_sparse": None,
            })
        full, sparse = results[-2:]
        sparse["full_over_sparse"] = full["median_ms"] / sparse["median_ms"]
    return results


def benchmark(args):
    directory, manifest = qualified_capture(args.run_dir)
    torch, device = single_cuda()
    attn = replay_layer(directory, torch, device)
    full = attn.get_processor()
    order = {size: measurement_order() for size in RESOLUTIONS}
    write_json(args.run_dir / "benchmark_schedule.json", {
        "seed": ORDER_SEED, "warmup_per_method": WARMUP, "pairs_per_size": REPEATS, "order": order,
    })
    stream = torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    end.record(stream)
    end.synchronize()

    rows = []
    with torch.no_grad(), (args.run_dir / "benchmark_raw.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for size in RESOLUTIONS:
            hidden = replay_input(directory, size, torch, device)
            side = size // LATENT_SCALE
            processors = {"full": full, "sparse": WindowAttentionProcessor(side, side)}
            for processor in processors.values():
                attn.set_processor(processor)
                for _ in range(WARMUP):
                    attn(hidden)
            torch.cuda.synchronize(device)
            order_index = 0
            for pair_id, methods in enumerate(order[size], start=1):
                for method in methods:
                    order_index += 1
                    attn.set_processor(processors[method])
                    torch.cuda.synchronize(device)
                    entry = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start.record(stream)
                    output = attn(hidden)
                    end.record(stream)
                    torch.cuda.synchronize(device)
                    elapsed = start.elapsed_time(end)
                    peak = torch.cuda.max_memory_allocated(device)
                    del output
                    if not math.isfinite(elapsed) or elapsed <= 0 or peak < entry:
                        raise RuntimeError(f"Invalid timing or memory reading at {size}, pair {pair_id}, {method}.")
                    row = {
                        "resolution": size, "tokens": side**2, "method": method,
                        "pair_id": pair_id, "run_order": order_index, "elapsed_ms": elapsed,
                        "entry_alloc_bytes": entry, "peak_alloc_bytes": peak,
                        "extra_alloc_bytes": peak - entry,
                        "input_sha256": manifest["inputs"][str(size)]["hidden_sha256"],
                    }
                    writer.writerow(row)
                    file.flush()
                    rows.append(row)
            del hidden
            print(f"Measured {size}×{size}: {2 * REPEATS} timed forwards", flush=True)

    summary = summarize_benchmark(rows)
    with (args.run_dir / "benchmark_summary.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def generate(args):
    directory, manifest = qualified_capture(args.run_dir)
    torch, device = single_cuda()
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file

    model = Path(manifest["model"])
    check_sha(model / UNET_WEIGHTS, manifest["unet_sha256"])
    check_sha(model / "unet/config.json", manifest["unet_config_sha256"])
    check_sha(model / "scheduler/scheduler_config.json", manifest["scheduler_config_sha256"])
    pipe = pipeline_on_cuda(model, torch, device)
    attn = target_attention(pipe.unet)
    weights = load_file(directory / "attention_weights.safetensors")
    current = attn.state_dict()
    if weights.keys() != current.keys() or any(
        not torch.equal(value, current[name].detach().cpu()) for name, value in weights.items()
    ):
        raise ValueError("The pipeline target weights differ from the benchmark snapshot.")
    original_processors = dict(pipe.unet.attn_processors)
    side = 512 // LATENT_SCALE
    processors = {"full": attn.get_processor(), "sparse": WindowAttentionProcessor(side, side)}
    initial_noise = load_noise(noise_path(directory, 512), 512, torch)
    images = args.run_dir / "images"
    images.mkdir()
    results = {}
    for method, processor in processors.items():
        attn.set_processor(processor)
        current_processors = pipe.unet.attn_processors
        if current_processors.keys() != original_processors.keys():
            raise RuntimeError("The pipeline attention module set changed.")
        changed = [key for key in current_processors if current_processors[key] is not original_processors[key]]
        if changed != ([TARGET + ".processor"] if method == "sparse" else []):
            raise RuntimeError(f"Unexpected processor changes: {changed}")
        checks = []

        def verify_output(_module, _inputs, output):
            checks.append(torch.isfinite(output).all())

        with torch.no_grad(), attn.register_forward_hook(verify_output):
            result = pipe(
                **SAMPLING, height=512, width=512, latents=initial_noise.to(device),
                output_type="np",
            )
        timestep_list = pipe.scheduler.timesteps.tolist()
        layer_finite = bool(checks and torch.stack(checks).all().item())
        pixels = result.images[0]
        finite_image = bool(np.isfinite(pixels).all())
        correct_shape = pixels.shape == (512, 512, 3)
        nonconstant = False
        image_path = images / f"finetuned_{method}.png"
        if finite_image and correct_shape:
            decoded = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
            nonconstant = bool(np.ptp(decoded) > 0)
            Image.fromarray(decoded).save(image_path)
        safety_passed = result.nsfw_content_detected == [False]
        passed = (
            len(checks) == SAMPLING["num_inference_steps"] and layer_finite
            and timestep_list == TIMESTEPS and nonconstant and safety_passed
        )
        results[method] = {
            "file": image_path.name if image_path.is_file() else None,
            "changed_processors": changed, "target_calls": len(checks),
            "target_outputs_finite": layer_finite, "timesteps": timestep_list,
            "image_shape": list(pixels.shape), "image_finite": finite_image,
            "image_nonconstant": nonconstant, "safety_checker_passed": safety_passed,
            "automatic_checks_passed": passed, "visual_review_pending": image_path.is_file(),
        }
        write_json(args.run_dir / "generation.json", {
            "model": str(model), "unet_sha256": manifest["unet_sha256"],
            "target_weights_sha256": manifest["weights_sha256"],
            "initial_noise": str(noise_path(directory, 512)),
            "initial_noise_sha256": manifest["inputs"]["512"]["noise_sha256"],
            "sampling": SAMPLING, "resolution": 512, "seed": SEED, "methods": results,
        })
        print(f"{method}: {len(checks)} target calls, automatic checks = {passed}", flush=True)
    if not all(record["automatic_checks_passed"] for record in results.values()):
        raise RuntimeError("Image checks failed. Review generation.json and the saved images.")


def plot(args):
    from sparse_plots import plot_results

    plot_results(args.run_dir, REPEATS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("capture", capture), ("qualify", qualify), ("benchmark", benchmark),
        ("generate", generate), ("plot", plot),
    ):
        command = commands.add_parser(name, help={
            "capture": "Save three real first-step attention inputs",
            "qualify": "Check the window implementation against a FP32 reference",
            "benchmark": "Measure the isolated attention layer with CUDA Events",
            "generate": "Save the Full/Sparse 512×512 DDIM comparison",
            "plot": "Create report figures from saved results",
        }[name])
        command.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
        command.set_defaults(handler=function)
    commands.choices["capture"].add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    commands.choices["capture"].add_argument("--model", type=Path, help="Relocated copy of the baseline pipeline")
    commands.choices["qualify"].add_argument("--capture-dir", type=Path, help="Reuse captures in a new run")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.command == "capture":
        args.run_dir.mkdir(parents=True, exist_ok=False)
    elif args.command == "qualify" and args.capture_dir:
        if not (args.capture_dir / "capture_manifest.json").is_file():
            parser.error("--capture-dir must contain a completed capture")
        args.run_dir.mkdir(parents=True, exist_ok=True)
    elif not args.run_dir.is_dir():
        parser.error("Run capture first or qualify with --capture-dir")
    if args.command in ("benchmark", "generate") and not (args.run_dir / "qualification.json").is_file():
        parser.error("Run qualify before benchmarking or generating comparisons")
    if args.command == "plot":
        required = ("qualify.csv", "benchmark_raw.csv", "benchmark_summary.csv", "generation.json",
                    "images/finetuned_full.png", "images/finetuned_sparse.png")
        missing = [name for name in required if not (args.run_dir / name).is_file()]
        if missing:
            parser.error(f"The report inputs are missing: {', '.join(missing)}")
    with record_stage(args.run_dir, args.command):
        args.handler(args)


if __name__ == "__main__":
    main()
