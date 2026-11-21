from hashlib import sha256 as sha256_digest
from pathlib import Path
from tempfile import mkdtemp
import json
import os
import random
import shutil

from datasets import Image, load_dataset
from PIL import Image as PILImage, ImageOps

project = Path(__file__).resolve().parent
scratch = Path(os.environ.get("SD_LAB_CACHE", str(project / "runtime")))
parquet = Path(os.environ.get(
    "SD_LAB_PARQUET", str(project / "data/raw/data/train-00000-of-00001.parquet")
))
root = project / "data/butterflies"
repo = "huggan/smithsonian_butterflies_subset"
revision = "3cdedf844922ab40393d46d4c7f81c596e1c6d45"
caption = "a photo of a winged insect"

def sha256(path):
    digest = sha256_digest()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()

def verify_export(path, expected_indices):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["repo"] == repo and manifest["revision"] == revision
    records = manifest["records"]
    assert [item["source_index"] for item in records] == expected_indices
    for split, expected in (("train", expected_indices[:32]), ("val", expected_indices[32:])):
        entries = [row for row in records if row["split"] == split]
        assert [row["source_index"] for row in entries] == expected
        folder = path / split
        assert sorted(file.name for file in folder.glob("*.png")) == sorted(
            row["file_name"] for row in entries
        )
        metadata = [json.loads(line) for line in (folder / "metadata.jsonl").read_text().splitlines()]
        assert metadata == [{"file_name": row["file_name"], "text": caption} for row in entries]
        for row in entries:
            file = folder / row["file_name"]
            assert row["text"] == caption and sha256(file) == row["sha256"]
            with PILImage.open(file) as image:
                assert image.format == "PNG" and image.mode == "RGB" and image.size == (512, 512)
                image.load()

indices = random.Random(42).sample(range(1000), 40)
if root.exists():
    verify_export(root, indices)
else:
    dataset = load_dataset(
        "parquet", data_files=str(parquet), split="train", cache_dir=str(scratch / "datasets-cache")
    )
    assert len(dataset) == 1000
    if not isinstance(dataset.features["image"], Image):
        dataset = dataset.cast_column("image", Image())
    root.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(mkdtemp(prefix=".butterflies-", dir=root.parent))
    try:
        records = []
        for split, source_indices in (("train", indices[:32]), ("val", indices[32:])):
            folder = temp / split
            folder.mkdir()
            with (folder / "metadata.jsonl").open("w", encoding="utf-8") as metadata:
                for index in source_indices:
                    row = dataset[index]
                    original = ImageOps.exif_transpose(row["image"]).convert("RGB")
                    image = ImageOps.pad(
                        original, (512, 512), method=PILImage.Resampling.LANCZOS, color="white"
                    )
                    filename = f"{index:04d}.png"
                    file = folder / filename
                    image.save(file)
                    metadata.write(json.dumps({"file_name": filename, "text": caption}) + "\n")
                    records.append({
                        "split": split, "source_index": index, "file_name": filename, "text": caption,
                        "source_image_hash": row.get("image_hash"), "sha256": sha256(file),
                    })
        manifest = {
            "repo": repo, "revision": revision, "seed": 42, "train_size": 32, "val_size": 8,
            "upstream": "https://huggingface.co/datasets/ceyda/smithsonian_butterflies",
            "upstream_license": "CC0-1.0", "source_sha256": sha256(parquet), "records": records,
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verify_export(temp, indices)
        temp.rename(root)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
print(f"Data verified: 32 train + 8 val RGB PNG files at {root}")
