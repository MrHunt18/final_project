import argparse
import gzip
import torch


def save_minimal_state_dict(input_path: str, intermediate_path: str, half_precision: bool = False):
    state = torch.load(input_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    if half_precision:
        state_dict = {
            k: v.half() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v
            for k, v in state_dict.items()
        }

    torch.save(state_dict, intermediate_path)
    print(f"Saved minimal state dict to {intermediate_path}")
    return intermediate_path


def compress_file_gzip(src_path: str, dst_path: str):
    with open(src_path, "rb") as f_in, gzip.open(dst_path, "wb") as f_out:
        f_out.writelines(f_in)
    print(f"Compressed {src_path} to {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress a PyTorch model for Streamlit deployment")
    parser.add_argument("--input", default="best_model.pth", help="Original model file path")
    parser.add_argument("--intermediate", default="best_model_state_dict.pth", help="Intermediate state dict path")
    parser.add_argument("--output", default="best_model_state_dict.pth.gz", help="Compressed output path")
    parser.add_argument("--half", action="store_true", help="Convert float32 weights to float16")
    args = parser.parse_args()

    intermediate_path = save_minimal_state_dict(args.input, args.intermediate, half_precision=args.half)
    compress_file_gzip(intermediate_path, args.output)
