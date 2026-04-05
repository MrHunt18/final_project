import argparse
import gzip
import torch


def save_minimal_state_dict(input_path: str, output_path: str, half_precision: bool = False) -> None:
    state = torch.load(input_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    if half_precision:
        state_dict = {
            key: value.half() if isinstance(value, torch.Tensor) and value.dtype == torch.float32 else value
            for key, value in state_dict.items()
        }

    torch.save(state_dict, output_path)
    print(f"Saved minimal state dict to {output_path}")


def compress_file(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as source, gzip.open(output_path, "wb") as target:
        target.writelines(source)
    print(f"Compressed {input_path} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress a PyTorch model for Streamlit deployment.")
    parser.add_argument("--input", default="best_model.pth", help="Path to the original model file.")
    parser.add_argument("--output", default="best_model_state_dict.pth.gz", help="Path to the compressed output file.")
    parser.add_argument("--temp", default="best_model_state_dict.pth", help="Temporary state dict file path.")
    parser.add_argument("--half", action="store_true", help="Convert float32 weights to float16.")
    args = parser.parse_args()

    save_minimal_state_dict(args.input, args.temp, half_precision=args.half)
    compress_file(args.temp, args.output)
