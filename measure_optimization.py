import gzip
import io
import os
import time
from pathlib import Path

import torch
from models.resnet_model import get_model

ROOT = Path(__file__).resolve().parent
MODEL_ORIGINAL = ROOT / "best_model.pth"
MODEL_STATE_DICT = ROOT / "best_model_state_dict.pth"
MODEL_GZ = ROOT / "best_model_state_dict.pth.gz"
EXCLUDE_DIRS = {"dataset", "dataset_all", "venv", ".venv", "heart_disease_prediction/myenv", "__pycache__"}
EXCLUDE_FILES = {"best_model.pth", "best_model_state_dict.pth", "best_model_state_dict.pth.gz", "streamlit_app.log", "temp_ecg_for_streamlit.png"}
EXCLUDE_EXTENSIONS = {".pyc", ".pyo", ".pyd", ".lnk", ".exe", ".tmp", ".bak"}


def human_mb(size_bytes: int) -> str:
    return f"{size_bytes / 1024 / 1024:.2f} MB"


def file_size(path: Path):
    return path.stat().st_size if path.exists() else 0


def folder_size(root: Path, exclude_dirs=None, exclude_files=None, exclude_exts=None):
    total = 0
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if exclude_dirs and any(str(path).replace('\\', '/').startswith(str(root / d).replace('\\', '/') + '/') for d in exclude_dirs):
            continue
        if exclude_files and path.name in exclude_files:
            continue
        if exclude_exts and path.suffix.lower() in exclude_exts:
            continue
        total += path.stat().st_size
    return total


def load_model_state_dict(path: Path):
    t0 = time.perf_counter()
    state = torch.load(path, map_location="cpu")
    return state, time.perf_counter() - t0


def load_compressed_state_dict(path: Path):
    t0 = time.perf_counter()
    with gzip.open(path, "rb") as f:
        buffer = io.BytesIO(f.read())
    state = torch.load(buffer, map_location="cpu")
    return state, time.perf_counter() - t0


def build_model_and_infer(state_dict, runs=10):
    model = get_model(num_classes=4)
    model.load_state_dict(state_dict)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        elapsed = time.perf_counter() - start
    return elapsed / runs


def print_section(title: str):
    print(f"\n=== {title} ===")


def main():
    print_section("Artifact Sizes")
    for path in [MODEL_ORIGINAL, MODEL_STATE_DICT, MODEL_GZ]:
        print(f"{path.name}: {human_mb(file_size(path))}")

    print_section("Load Times")
    if MODEL_ORIGINAL.exists():
        _, orig_time = load_model_state_dict(MODEL_ORIGINAL)
        print(f"Original load time: {orig_time:.3f} sec")
    if MODEL_GZ.exists():
        state, gz_time = load_compressed_state_dict(MODEL_GZ)
        print(f"Compressed load time: {gz_time:.3f} sec")

        infer_ms = build_model_and_infer(state) * 1000
        print(f"Compressed model inference latency: {infer_ms:.1f} ms (avg over 10 runs)")

    full_size = folder_size(ROOT)
    clean_size = folder_size(ROOT, exclude_dirs=EXCLUDE_DIRS, exclude_files=EXCLUDE_FILES, exclude_exts=EXCLUDE_EXTENSIONS)

    print_section("Workspace Size")
    print(f"Full workspace size: {human_mb(full_size)}")
    print(f"Clean deployable size: {human_mb(clean_size)}")
    print(f"Size reduction: {human_mb(full_size - clean_size)}")

    print_section("Notes")
    print("- The clean deployable size excludes datasets, virtual environments, large temp files, logs, and local model artifacts.")
    print("- The compressed model is the optimized artifact to store in Drive or include in repo if needed.")


if __name__ == '__main__':
    main()
