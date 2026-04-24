import gzip
import io
import os
import time
import torch
from models.resnet_model import get_model


def get_file_size_mb(file_path):
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        return round(size_bytes / (1024 * 1024), 2)
    return None


def load_full_model(path: str):
    start = time.perf_counter()
    model_data = torch.load(path, map_location="cpu")
    elapsed = time.perf_counter() - start
    return model_data, elapsed


def load_compressed_state_dict(path: str):
    start = time.perf_counter()
    with gzip.open(path, "rb") as f:
        buffer = io.BytesIO(f.read())
    state_dict = torch.load(buffer, map_location="cpu")
    elapsed = time.perf_counter() - start
    return state_dict, elapsed


def measure_model_sizes():
    print("=== Artifact Sizes ===")

    original_path = "best_model.pth"
    original_size = get_file_size_mb(original_path)
    if original_size is not None:
        print(f"{original_path}: {original_size} MB")
    else:
        print(f"{original_path}: Not found")

    intermediate_path = "best_model_state_dict.pth"
    intermediate_size = get_file_size_mb(intermediate_path)
    if intermediate_size is not None:
        print(f"{intermediate_path}: {intermediate_size} MB")
    else:
        print(f"{intermediate_path}: Not found")

    compressed_path = "best_model_state_dict.pth.gz"
    compressed_size = get_file_size_mb(compressed_path)
    if compressed_size is not None:
        print(f"{compressed_path}: {compressed_size} MB")
    else:
        print(f"{compressed_path}: Not found")

    return original_path, compressed_path


def measure_load_times(original_path: str, compressed_path: str):
    print("\n=== Load Times ===")

    if os.path.exists(original_path):
        _, original_time = load_full_model(original_path)
        print(f"Original load time: {original_time:.3f} sec")
    else:
        print(f"Original load time: file not found")
        original_time = None

    if os.path.exists(compressed_path):
        _, compressed_time = load_compressed_state_dict(compressed_path)
        print(f"Compressed load time: {compressed_time:.3f} sec")
    else:
        print(f"Compressed load time: file not found")
        compressed_time = None

    return compressed_path, compressed_time


def measure_inference_latency(model_path: str, runs: int = 10):
    if not os.path.exists(model_path):
        print("Compressed model inference latency: model file not found")
        return

    state_dict, _ = load_compressed_state_dict(model_path)
    model = get_model(num_classes=4)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            model(dummy_input)

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(dummy_input)
            timings.append(time.perf_counter() - start)

    avg_latency_ms = (sum(timings) / len(timings)) * 1000
    print(f"Compressed model inference latency: {avg_latency_ms:.1f} ms (avg over {runs} runs)")


def measure_app_size():
    print("\n=== App Repository Size (Deployable) ===")

    total_size = 0
    excluded_patterns = [
        '__pycache__',
        '.git',
        'dataset',
        'dataset_all',
        'heart_disease_prediction/myenv',
        'venv',
        '.venv',
        'myenv'
    ]

    excluded_extensions = ['.pth', '.pt', '.pth.gz', '.log']

    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not any(pattern in os.path.join(root, d) for pattern in excluded_patterns)]
        for file in files:
            if any(file.endswith(ext) for ext in excluded_extensions):
                continue
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)

    size_mb = round(total_size / (1024 * 1024), 2)
    print(f"Total deployable app size: {size_mb} MB")
    print("This excludes models, datasets, and temp files per .gitignore")


if __name__ == "__main__":
    original_path, compressed_path = measure_model_sizes()
    _, _ = measure_load_times(original_path, compressed_path)
    measure_inference_latency(compressed_path, runs=10)
    measure_app_size()
